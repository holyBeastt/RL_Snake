# Hiểu Sâu Về Deep Q-Learning (DQN) Toàn Tập 🧠

Deep Q-Learning là thuật toán do Google DeepMind tạo ra, nổi tiếng nhờ khả năng chơi game Atari từ con số 0 chỉ bằng cách "nhìn" vào màn hình pixel như con người. Thuật toán này kết hợp giữa **Q-Learning** (RL cổ điển) và **Deep Learning** (Mạng Nơ-ron).

Dưới đây là mổ xẻ chi tiết từng nấc của thuật toán:

---

## 1. Tiền Xử Lý Dữ Liệu (Preprocessing)
Ngoại trừ các game siêu đơn giản chỉ có vài con số tọa độ (như file rắn `snake_env.py` của chúng ta, truyền thẳng 11 con số), đối với môi trường phức tạp (như xử lý nguyên màn hình ảnh), việc đưa toàn bộ pixel 1080p full-color vào AI là quá sức tưởng tượng và lãng phí tính toán. Hệ thống cần "cắt gọt" dữ liệu:

*   **Grayscaling (Làm xám):** Chuyển ảnh RGB (3 kênh màu đỏ/xanh/lục) thành đen trắng (1 kênh). Màu sắc thường không quá quan trọng cho logic game, việc làm xám giảm $2/3$ khối lượng data.
*   **Cropping (Cắt cúp màn hình):** Cắt bỏ UI vướng víu không liên quan như thanh máu, menu viền, điểm số góc màn hình... Chỉ giữ lại "đấu trường" chính.
*   **Downsampling / Resizing:** Thu nhỏ độ phân giải ảnh (thường thu gọn thành ma trận hình vuông nhỏ khoảng $84 \times 84$ pixel).
*   **Frame Stacking (Chồng khung hình):** Một tấm ảnh đơn lẻ không thể cho AI biết cái xe đang chạy tới hay lùi lại. Hệ thống thường gộp 4 khung hình liên tiếp cuối cùng (4 frames) chập lại làm 1 State duy nhất đưa vào mạng nơ-ron (đầu vào thành mảng $84 \times 84 \times 4$). Việc này giúp AI cảm nhận được "Tốc độ" và "Hướng di chuyển" của vạn vật.

---

## 2. Trí Nhớ & Kinh Nghiệm (Experience Replay)
Nếu con người chỉ học từ hiện hành, ta dễ bị thiên kiến bởi vài cái chết vừa xảy ra và quên sạch các kỹ năng sống xa xưa. Học máy cũng vậy. Lập trình viên thiết kế một kho trí nhớ (Replay Buffer/Memory) với kích cỡ khoảng 100,000 đến 1 triệu vòng lặp.

Mỗi khi AI đi 1 bước, hệ thống gói gọn lại quá khứ dưới dạng cú pháp `Transition`:
> `(State_cũ, Hành_động_đã_làm, Phần_thưởng_đạt_được, State_mới, Chết_hay_sống)`

Từng cục kinh nghiệm này được đẩy vào kho lưu trữ (nếu kho đầy, cứ đẩy cái mới vào thì xóa cái cũ nhất văng ra). Khi Model cần "ôn bài" (Train), nó không học theo thứ tự tuyến tính thời gian chơi, mà **bốc ra ngẫu nhiên một mẻ (ví dụ Batch Size = 32 hoặc 64 kinh nghiệm bất kì)** để học. 
**Tại sao phải trộn ngẫu nhiên?** Để phá vỡ tính liên kết thời gian (correlation). Tránh cho việc xe đang ôm một đường cong chữ C quá dài khiến mạng tẩu hỏa nhập ma tin rằng "Rẽ Trái mãi mãi là chân lý".

---

## 3. Chính Sách Hành Động: Epsilon-Greedy ($\epsilon$-greedy)
Làm sao AI biết nó nên ấn nút gì?
*   Một hệ số gọi là Epsilon $\epsilon$ được thiết lập từ $1.0 = 100\%$.
*   Máy tung đồng xu. Nếu ra xác suất nằm trong khoảng $\epsilon$, nó **nhắm mắt bấm nút Ngẫu Nhiên (Explore)**. Tại sao? Để nó dám đi vào những ngõ ngách chưa từng biết, thử những chiến thuật liều mạng chưa từng chơi, lỡ nó tìm được rương kho báu giấu kín thì sao?
*   Ngược lại, nếu nằm ngoài $\epsilon$, nó sẽ ném `State` cho bộ não quét để lấy kết quả (tức Q-Value cao nhất) -> **Đưa ra Hành động Khôn ngoan nhất (Exploit)** .
*   Sau mỗi vài ngàn tập, Epsilon này bị ép giảm dần (decay) xuống xấp xỉ $0$. Nghĩa là khi bé tôi thử sai búa xua (chết liên tục chục ván đầu), càng lớn não có nếp nhăn rồi thì tôi không thử ngu nữa, tôi chơi hoàn toàn dựa dẫm vào trí thông minh của mình.

---

## 4. Q-Value & Phương Trình Bellman
Toàn bộ AI tính toán dựa trên việc: "Phần thưởng mong đợi ở tương lai (Q-value) nếu ta dùng hành động cục diện $a$ tại hoàn cảnh $s$".

Nó dựa trên ĐỊNH LÝ TÀI CHÍNH LỚN: Điểm số lúc này + ($0.99 \times$ dự đoán phần thưởng to nhất mà ta có thể giật được trong tương lai dài hạn). Cái $0.99$ gọi là Discount Gamma $\gamma$, nó trừ hao lạm phát: "1 con cá bây giờ giá trị hơn 10 con cá có hứa hẹn tìm được vào thế kỷ sau, nên chim ưng ưu tiên phần thưởng sát sườn hiện tại hơn một chút xíu".

Mạng nơ ron phải đoán con số Q-value này. Quá trình train bản chất là: Thầy giáo vác thước đi đập vào tay mạng nơ ron bắt nó sửa sao cho (Phần đoán) phải khớp với (Điểm số thật + điểm kỳ vọng bước sau).

`Loss (Mean Squared Error) = ( (Reward + Gamma * MaxQ_tuong_lai) - Q_du_doan )^2`

---

## 5. Mánh Lới Khủng Nhất: Target Network Đóng Băng
Trong cái `Loss` bên trên, để đoán MaxQ tương lai, nếu ta dùng ngay mạng nơ ron đang học (policy net) thì chẳng khác nào chó đuổi theo cái đuôi của chính nó. *Tại vì nó vừa học thông minh lên, cái kỳ vọng tương lai cũng bị xoay dịch đi ngay tắp lự.* Quá trình train sẽ giật lag rung lắc liên hồi không thể hội tụ (Học lệch).

**Google DeepMind giải điều kiện này cực khôn ngoan:** 
Họ clone mạng Nơ-ron làm 2 bản sao độc lập, một cái là `Policy Network` (não chính), và `Target Network` (não cũ).
*   Cái não chính `Policy Network` (hoạt động nhộn nhịp) sẽ tính toán hành động và không ngừng sửa chữa trọng số (Backpropagation/Gradient Descent) qua TỪNG MILI GIÂY lúc học. 
*   **Bí kíp:** Còn cái máy chấm điểm dự đoán tương lai `Target Network` sẽ bị **ĐÓNG BĂNG ĐÔNG LẠNH hoàn toàn**.
*   Khi não chính học và chạy tung tăng, nó bám vào cái mục tiêu tĩnh do Não Băng giá đưa ra, nên nó không bị giật lag mục tiêu (Chó dễ dàng đuổi theo cục xương bị ném đứng yên).
*   Sau mỗi chu kỳ cực lâu (Ví dụ: sau mỗi 1000 màn, hoặc 10000 bước đi - biến `TARGET_UPDATE` trong code), hệ thống mới cho phép **Cập nhật một cục copy paste toàn bộ (Copy Weights)** của Não Chính vừa khôn lên chuyển sang đè vào Não Bức tượng cũ, thiết lập một chuẩn mức tĩnh lặng xịn xò mới cho chặng đường rèn luyện tỷ năm tiếp theo.

---

## Tổng Kết Luồng (Workflow) một chu kỳ DQN:

1. Game quăng State cho AI.
2. Tung xúc xắc Epsilon: Chơi ngẫu nhiên, hoặc dựa bộ não `Policy Net` mà bấm nút.
3. Nhân vật di chuyển trúng độc chết, Game quăng lại State mới + Lệnh Game Over + Điểm Reward Trừ cực mạnh.
4. Hành trình đâm tường đó bị đóng gói nhét tống vào sọt rác kho `Replay Buffer` (nhưng sọt rác này chứa vàng).
5. Bot bốc ngẫu nhiên 64 cục rác trong quá khứ ra sân khấu tính đạo hàm.
6. Lấy bộ não đóng băng `Target Net` chấm điểm đoán trước Q-Value (Kỳ vọng).
7. Lấy bộ não năng động `Policy Net` đang chạy tính `Loss` so với điểm thực tế + kỳ vọng, từ Loss truyền Lan truyền ngược sinh cập nhật Adam Optimizer làm thông minh Não năng động lên $0.001\%$.
8. Qua n ván, đè bản copy Não khôn sang Não đóng băng để cập nhật mặt bằng chung. Lặp lại 40 tiếng đồng hồ liên tục.
