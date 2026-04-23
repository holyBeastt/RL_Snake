import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DQN(nn.Module):
    """
    Simple 3-layer fully-connected DQN.
    Input : 11-dim state vector
    Output: Q-values for 3 actions (straight, right, left)
    """

    def __init__(self, input_size: int = 11, hidden_size: int = 256, output_size: int = 3):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def save(self, path: str = "model.pth"):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[Model] Saved → {path}")

    def load(self, path: str = "model.pth"):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"[Model] Loaded ← {path}")
        else:
            print(f"[Model] No checkpoint found at {path}, starting fresh.")
