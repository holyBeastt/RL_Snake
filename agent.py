import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from model import DQN

# ------------------------------------------------------------------ Hyperparams
# BATCH_SIZE: Số lượng "kỷ niệm" lấy từ ReplayBuffer mỗi lần để dạy AI. 
# Ở đây user đã tăng lên 128 (phương sai lớn -> cần tăng lô mẫu để tính trung bình Gradient chuẩn xác hơn).
BATCH_SIZE     = 128

# GAMMA (Discount Factor): Mức độ quan tâm đến tương lai. 0.98 nghĩa là rất quan tâm đến phần thưởng xa.
GAMMA          = 0.98      
# Tốc độ học (Learning Rate): Mức độ tinh chỉnh trọng số mạng Neuron ở mỗi bước cập nhật. Càng nhỏ học càng chậm nhưng chắc.
LR             = 5e-4      
# Kích thước kho chứa kinh nghiệm quá khứ (lưu tối đa 100,000 bước đi).
MEMORY_SIZE    = 100_000   
# Cứ sau 20 games thì copy toàn bộ chất xám từ Não Chính (Policy Net) sang Não Đóng Băng (Target Net).
TARGET_UPDATE  = 20        

# Epsilon-greedy (Cơ chế Nhắm mắt đi bừa để Khám phá)
# Phần trăm ban đầu ép chơi ngẫu nhiên là 5% (0.05).
EPS_START = 0.05
# Không bao giờ để ngẫu nhiên về 0 hoàn toàn, giữ lại 1% tò mò.
EPS_END   = 0.01
# Tốc độ giảm dần Epsilon sau mỗi game (nhân với 0.997).
EPS_DECAY = 0.997

# ------------------------------------------------------------------ Replay Buffer
class ReplayBuffer:
    """
    Kho trí nhớ của AI. Dùng để lưu trữ những bước đi trong quá khứ 
    nhằm xáo trộn lúc học, giúp mạng không bị 'học vẹt' các chuỗi hành động liên tiếp.
    """
    def __init__(self, capacity: int = MEMORY_SIZE):
        # deque là dạng hàng đợi hai đầu. maxlen=capacity giúp tự đẩy dữ liệu cũ nhất ra ngoài nếu kho đã đầy.
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Lưu 1 dòng kỷ niệm (Transition) vào Kho.
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        # Bốc ngẫu nhiên `batch_size` (vd: 128) kỷ niệm từ kho
        batch      = random.sample(self.buf, batch_size)
        # Tách riêng rẽ từng mảng: states, actions, rewards... (hàm zip(*batch) lộn trục ma trận)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Chuyển đổi toàn bộ thành PyTorch Tensor (ma trận cho GPU tính toán)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self):
        # Lấy ra số lượng kỷ niệm hiện tại đang có trong kho
        return len(self.buf)

# ------------------------------------------------------------------ DQN Agent
class DQNAgent:
    """
    Bộ não chỉ huy. Cầm Trí Nhớ (Replay Buffer) và Cầm Mạng Thần Kinh (DQN) để ra lệnh cho Rắn.
    """
    def __init__(self, state_size: int = 11, action_size: int = 3):
        self.state_size  = state_size # Đầu vào (11 biến môi trường)
        self.action_size = action_size # Đầu ra (3 hành động: Thẳng, Trái, Phải)
        self.epsilon     = EPS_START # Đặt mức dũng cảm tò mò ban đầu
        self.n_games     = 0 # Đếm số trận đã chơi tự động

        # Nếu máy tính có card rời CUDA, dùng GPU để vắt kiệt sức mạnh. Không thì xài CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # KHỞI TẠO NÃO BỘ: Tạo 2 mạng Thần kinh (Não chính và Não Tĩnh)
        # Não chính học liên tục (policy_net).
        self.policy_net = DQN(state_size, 256, action_size).to(self.device)
        # Não tĩnh dùng để phán đoán Kì vọng (target_net), học gián tiếp từ não chính bằng cách copy.
        self.target_net = DQN(state_size, 256, action_size).to(self.device)
        # Copy nếp nhăn (trọng số) lần đầu tiên sao cho 2 não y hệt nhau.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Đóng băng Não Tĩnh (eval mode), bảo với PyTorch là không tính gradient cho nó.
        self.target_net.eval()

        # Optimizer: Khai báo người sửa lỗi mạng (Adam), nhiệm vụ là thay đổi trọng số policy sao cho khớp Loss.
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        # Khởi tạo sọt rác Kỷ niệm
        self.memory    = ReplayBuffer(MEMORY_SIZE)

    # ----------------------------------------------------------- Chốt hành động
    def select_action(self, state: np.ndarray) -> int:
        """ AI suy nghĩ: Đi đâu bây giờ? """
        # Tung đồng xu dũng cảm (Epsilon). Nếu ra số nhỏ hơn Epsilon -> Chơi ngẫu nhiên.
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Nếu không ngẫu nhiên, đưa Data "State" (Mắt rắn) vào Não Chính để phân tích
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        # no_grad() = Khóa tính năng Học, vì bước này chỉ là Ứng dụng để chơi (Inference), không phải luyện tập.
        with torch.no_grad():
            q = self.policy_net(s) # Đầu ra là 3 Điểm Q-Value cho 3 hành động.
        
        # Hàm argmax = Chọn chiêu nào có Q-Value cao nhất (Hành động có vẻ khôn khéo nhất lúc này)
        return int(q.argmax().item())

    # ------------------------------------------------------ Dạy học AI
    def remember(self, state, action, reward, next_state, done):
        """ Ép AI lưu lại kỷ niệm vào kho sau khi lỡ đâm đầu vào tường (hoặc lỡ ăn được hạt). """
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """ Đây là hàm đốt nhân CPU/GPU nhiều nhất, chính là lúc Học Backpropagation """
        # Khi kho chưa đủ 128 (Batch size) dữ liệu, đi về không học, thu thập data đi đã.
        if len(self.memory) < BATCH_SIZE:
            return

        # Rút ngẫu nhiên 128 thẻ Cảnh quen thuộc trong não ra 
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        # Đẩy dữ liệu này vào Cạc màn hình (GPU/CPU)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # BUỚC 1: HỎI NÃO CHÍNH
        # Đưa 128 cảnh (states) vào não chính. Bốc ra đúng Cột Q-Value tương ứng với Action trong quá khứ đã được chọn.
        # Nghĩa là nó tính thử xem: Ồ, ngày xưa mình làm hành động đó thì Não bây giờ chấm điểm Q thế nào?
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # BUỚC 2: HỎI NÃO TĨNH (Kỳ Vọng Thực Tế)
        with torch.no_grad(): # Não Tĩnh ko liên quan việc cập nhật trọng số
            # Lấy Cảnh ngay khúc giây sau (next_states) đưa cho Não tĩnh, coi xem tương lai có gì tươi sáng không (max)
            next_q = self.target_net(next_states).max(1)[0]
            
            # THEO PHƯƠNG TRÌNH BELLMAN: Q_Lý_Tưởng = Phần thưởng nhận được NGAY + Kì vọng tương lai do Não Tĩnh đoán.
            # Nếu done=1 (Game Over) thì tương lai đã tắt, next_q sẽ bị nhân với (1-1=0). Vứt mẹ tương lai đi!
            target = rewards + GAMMA * next_q * (1 - dones)

        # BƯỚC 3: DÙNG THƯỚC KẺ ĐẬP VÀO TAY NÃO CHÍNH (TÍNH LOSS)
        # Chênh lệch Bình Phương Gắn Nghĩa (MSE). Đem Q-do_não_chính_ước_lượng chọi với Target_lý_tưởng_tính_được.
        loss = nn.MSELoss()(q_values, target)

        # BƯỚC 4: HỌC TẬP THỰC SỰ
        self.optimizer.zero_grad() # Xóa sổ điểm rác trong bộ nhớ Gradient cũ
        loss.backward()            # Lan truyền ngược (Đạo hàm tìm gốc rễ lỗi sai đang nằm ở cái Nơ-ron nào)
        self.optimizer.step()      # Mở khóa vặn Cờ-lê (Cập nhật các trọng số nơ-ron). NÃO ĐÃ KHÔN LÊN!

        return loss.item()

    # ------------------------------------------------- Dọn Dẹp Cuối Game
    def on_episode_end(self):
        """ Kích hoạt mỗi khi Rắn Chết -> Dấu chấm hết một màn game """
        self.n_games += 1 # Đếm số màn chơi
        
        # Bớt Dũng cảm (tò mò khám phá ngẫu nhiên) đi, và bám vào trí thông minh nhiều hơn.
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        
        # Nết N_GAME chia hết cho 20, vát toàn bộ Trí thông minh mới của Não Chính tạt sang Não Tĩnh (Đồng bộ)
        if self.n_games % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str = "checkpoints/model.pth"):
        """ Khóa lưu model ra ổ cứng tránh mất dữ kiện (Gọi bên hàm Model.save) """
        self.policy_net.save(path)

    def load(self, path: str = "checkpoints/model.pth"):
        """ Bốc ổ cứng nạp ngược lên Não. Não Tĩnh cũng copy y xì Não chính để chuẩn bị sẵn """
        self.policy_net.load(path)
        self.target_net.load_state_dict(self.policy_net.state_dict())

