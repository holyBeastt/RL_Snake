import pygame
import random
import numpy as np

# --- Constants ---
BLOCK = 20
WIDTH = 400
HEIGHT = 400
GRID_W = WIDTH // BLOCK
GRID_H = HEIGHT // BLOCK

# Directions (dx, dy)
UP    = (0, -BLOCK)
DOWN  = (0,  BLOCK)
LEFT  = (-BLOCK, 0)
RIGHT = ( BLOCK, 0)

DIRS = [UP, RIGHT, DOWN, LEFT]  # clockwise order

# Colors
BLACK  = (0,   0,   0)
GREEN  = (0,   200, 80)
DGREEN = (0,   150, 50)
RED    = (220, 50,  50)
WHITE  = (255, 255, 255)
GRAY   = (40,  40,  40)


class SnakeGame:
    """
    Snake game environment, RL-ready.

    State  : 11-dim numpy array (collision danger + direction + food direction)
    Action : 0 = straight | 1 = turn right | 2 = turn left
    Reward : +10 eat food | -10 die | +0.1 survive step
    """

    def __init__(self, render: bool = False, speed: int = 30):
        self.render_mode = render
        self.speed = speed

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("🐍 RL Snake")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("monospace", 16, bold=True)

        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        cx = (GRID_W // 2) * BLOCK
        cy = (GRID_H // 2) * BLOCK
        self.snake     = [(cx, cy), (cx - BLOCK, cy), (cx - 2 * BLOCK, cy)]
        self.direction = RIGHT
        self.score     = 0
        self.steps     = 0
        self.food      = self._spawn_food()
        return self.get_state()

    # -------------------------------------------------------------- spawn food
    def _spawn_food(self):
        while True:
            pos = (
                random.randint(0, GRID_W - 1) * BLOCK,
                random.randint(0, GRID_H - 1) * BLOCK,
            )
            if pos not in self.snake:
                return pos

    # ------------------------------------------------------------------- step
    def step(self, action: int):
        """
        action: 0=straight, 1=turn right, 2=turn left
        returns: (state, reward, done)
        """
        self.steps += 1

        # update direction
        self._apply_action(action)

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # insert new head
        self.snake.insert(0, new_head)

        reward = 0.0
        done   = False

        # --- collision check ---
        if self._is_collision(new_head):
            reward = -10.0
            done   = True
            if self.render_mode:
                self._draw()
            return self.get_state(), reward, done

        # --- food check ---
        if new_head == self.food:
            self.score += 1
            reward     = 10.0
            self.food  = self._spawn_food()
        else:
            self.snake.pop()
            reward = 0.1   # survived

        # prevent infinite loops: kill if no food eaten for too long
        if self.steps > 100 * len(self.snake):
            done   = True
            reward = -10.0

        if self.render_mode:
            self._draw()

        return self.get_state(), reward, done

    # ----------------------------------------------------------- apply action
    def _apply_action(self, action: int):
        idx = DIRS.index(self.direction)
        if action == 1:   # right turn
            idx = (idx + 1) % 4
        elif action == 2: # left turn
            idx = (idx - 1) % 4
        self.direction = DIRS[idx]

    # ----------------------------------------------------------- is_collision
    def _is_collision(self, point):
        x, y = point
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        if point in self.snake[1:]:
            return True
        return False

    # --------------------------------------------------------------- get_state
    def get_state(self):
        """
        11-dim state:
          [0..2]  danger straight / right / left
          [3..6]  current direction (up/down/left/right)
          [7..10] food relative position (left/right/up/down)
        """
        head = self.snake[0]
        hx, hy = head

        # points in each absolute direction
        pt_u = (hx, hy - BLOCK)
        pt_d = (hx, hy + BLOCK)
        pt_l = (hx - BLOCK, hy)
        pt_r = (hx + BLOCK, hy)

        dir_u = self.direction == UP
        dir_d = self.direction == DOWN
        dir_l = self.direction == LEFT
        dir_r = self.direction == RIGHT

        # danger: straight, right, left  (relative to current direction)
        if dir_r:
            straight, turn_r, turn_l = pt_r, pt_d, pt_u
        elif dir_l:
            straight, turn_r, turn_l = pt_l, pt_u, pt_d
        elif dir_u:
            straight, turn_r, turn_l = pt_u, pt_r, pt_l
        else:   # down
            straight, turn_r, turn_l = pt_d, pt_l, pt_r

        fx, fy = self.food

        state = [
            # danger
            int(self._is_collision(straight)),
            int(self._is_collision(turn_r)),
            int(self._is_collision(turn_l)),
            # direction
            int(dir_u),
            int(dir_d),
            int(dir_l),
            int(dir_r),
            # food location
            int(fx < hx),   # food left
            int(fx > hx),   # food right
            int(fy < hy),   # food up
            int(fy > hy),   # food down
        ]
        return np.array(state, dtype=np.float32)

    # ------------------------------------------------------------------ render
    def _draw(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill(BLACK)

        # grid
        for x in range(0, WIDTH, BLOCK):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, BLOCK):
            pygame.draw.line(self.screen, GRAY, (0, y), (WIDTH, y))

        # food
        fx, fy = self.food
        pygame.draw.rect(self.screen, RED,    (fx + 2, fy + 2, BLOCK - 4, BLOCK - 4))

        # snake
        for i, (sx, sy) in enumerate(self.snake):
            color = GREEN if i > 0 else DGREEN
            pygame.draw.rect(self.screen, color, (sx + 1, sy + 1, BLOCK - 2, BLOCK - 2))
            if i == 0:
                pygame.draw.rect(self.screen, WHITE, (sx + 1, sy + 1, BLOCK - 2, BLOCK - 2), 2)

        # score
        txt = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(txt, (6, 4))

        pygame.display.flip()
        self.clock.tick(self.speed)

    def close(self):
        if self.render_mode:
            pygame.quit()
