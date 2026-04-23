"""
train.py — Train DQN agent on Snake game.

Usage:
  python train.py               # silent mode (no window)
  python train.py --render      # show pygame window while training
  python train.py --episodes 500
"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless matplotlib
import matplotlib.pyplot as plt

from snake_env import SnakeGame
from agent import DQNAgent

# ------------------------------------------------------------------ args
parser = argparse.ArgumentParser()
parser.add_argument("--episodes",  type=int,  default=1000)
parser.add_argument("--render",    action="store_true")
parser.add_argument("--speed",     type=int,  default=60,  help="FPS when rendering")
parser.add_argument("--save",      type=str,  default="checkpoints/model.pth")
parser.add_argument("--load",      type=str,  default=None)
args = parser.parse_args()

# ------------------------------------------------------------------ init
game  = SnakeGame(render=args.render, speed=args.speed)
agent = DQNAgent(state_size=11, action_size=3)

if args.load:
    agent.load(args.load)

os.makedirs("checkpoints", exist_ok=True)

# ------------------------------------------------------------------ logging
scores        = []
mean_scores   = []
best_score    = 0
total_reward  = 0.0
start_time    = time.time()

print(f"\n{'='*55}")
print(f"  RL Snake – DQN Training")
print(f"  Episodes: {args.episodes}  |  Device: {agent.device}")
print(f"{'='*55}\n")

# ------------------------------------------------------------------ helper
def _plot(scores, means, ep, final=False):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(scores, alpha=0.4, label="Score", color="#4fc3f7")
    ax.plot(means,  linewidth=2, label="Mean-100", color="#f06292")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title(f"DQN Snake – Training Progress (ep {ep})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = "checkpoints/training_curve.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    if final:
        print(f"  Plot saved : {path}")


# ------------------------------------------------------------------ training loop
for ep in range(1, args.episodes + 1):
    state = game.reset()
    episode_reward = 0.0

    while True:
        action = agent.select_action(state)
        next_state, reward, done = game.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.train_step()

        state          = next_state
        episode_reward += reward

        if done:
            break

    agent.on_episode_end()

    ep_score = game.score
    scores.append(ep_score)
    total_reward += episode_reward
    mean_score = np.mean(scores[-100:])  # rolling mean
    mean_scores.append(mean_score)

    # save best model
    if ep_score > best_score:
        best_score = ep_score
        agent.save(args.save)

    # print progress
    if ep % 10 == 0 or ep == 1:
        elapsed = time.time() - start_time
        print(
            f"  Ep {ep:>5}/{args.episodes}"
            f"  Score: {ep_score:>4}"
            f"  Best: {best_score:>4}"
            f"  Mean100: {mean_score:>6.2f}"
            f"  ε: {agent.epsilon:.3f}"
            f"  Time: {elapsed:>6.1f}s"
        )

    # periodic checkpoint + plot every 100 episodes
    if ep % 100 == 0:
        agent.save(f"checkpoints/model_ep{ep}.pth")
        _plot(scores, mean_scores, ep)

# ------------------------------------------------------------------ final save & plot
agent.save(args.save)
_plot(scores, mean_scores, args.episodes, final=True)

print(f"\n{'='*55}")
print(f"  Training complete!")
print(f"  Best score  : {best_score}")
print(f"  Final mean  : {mean_scores[-1]:.2f}")
print(f"  Model saved : {args.save}")
print(f"{'='*55}\n")

game.close()


