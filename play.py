"""
play.py — Watch trained DQN agent play Snake live.

Usage:
  python play.py                        # load best checkpoint
  python play.py --model checkpoints/model_ep500.pth
  python play.py --speed 15             # slow it down to watch
  python play.py --human                # you play with WASD / arrow keys
"""

import argparse
import pygame
import torch
import numpy as np

from snake_env import SnakeGame, BLOCK, WIDTH, HEIGHT, WHITE, GREEN, RED, BLACK, GRAY
from agent import DQNAgent

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="checkpoints/model.pth")
parser.add_argument("--speed", type=int, default=20)
parser.add_argument("--human", action="store_true")
args = parser.parse_args()

# ------------------------------------------------------------------ setup
game  = SnakeGame(render=True, speed=args.speed)
agent = DQNAgent(state_size=11, action_size=3)

if not args.human:
    agent.load(args.model)
    agent.epsilon = 0.0   # pure greedy

# direction map for human mode
from snake_env import UP, DOWN, LEFT, RIGHT
KEY_DIR = {
    pygame.K_UP: UP,    pygame.K_w: UP,
    pygame.K_DOWN: DOWN, pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT, pygame.K_a: LEFT,
    pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
}

# ------------------------------------------------------------------ game loop
state   = game.reset()
running = True
episode = 0
scores  = []

while running:
    # --- event handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if args.human and event.type == pygame.KEYDOWN:
            if event.key in KEY_DIR:
                new_dir = KEY_DIR[event.key]
                # prevent reverse
                cur = game.direction
                if (new_dir[0] + cur[0], new_dir[1] + cur[1]) != (0, 0):
                    game.direction = new_dir

    if not running:
        break

    # --- action ---
    if args.human:
        action = 0   # keep current direction (already updated above)
    else:
        action = agent.select_action(state)

    state, reward, done = game.step(action)

    if done:
        episode += 1
        scores.append(game.score)
        avg = sum(scores) / len(scores)
        print(f"Episode {episode:>4}  Score: {game.score:>4}  Avg: {avg:.2f}")
        state = game.reset()

game.close()
print(f"\nSessions: {episode}  Best: {max(scores) if scores else 0}  Avg: {sum(scores)/max(len(scores),1):.2f}")
