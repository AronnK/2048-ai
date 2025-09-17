import gymnasium as gym
from gymnasium import spaces
import numpy as np
import game

WEIGHT_MATRIX = np.array([
    [16, 12, 8, 4],
    [ 8,  6,  4, 2],
    [ 4,  2,  1, 0],
    [ 2,  1,  0, 0]
])

class Game2048Env(gym.Env):
    """
    2048 Gym environment with theoretically sound potential-based reward shaping.
    The final reward is the sum of the base game score and the shaped heuristic reward.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None,
                 kappa_invalid=10.0,      # Penalty for invalid moves
                 w_empty=2.5,            # Weight for empty cells
                 w_mono=1.0,             # Weight for monotonicity
                 w_merge=1.0,            # Weight for merge potential
                 w_snake=5.0,            # Weight for snake score
                 gamma=0.99,             # Discount factor for shaping
                 alpha=0.5,              # Shaping scale
                 shaping_clip=5.0,       # Clip the magnitude of the SHAPED part of the reward
                 debug=False):           # Debug logging flag
        super(Game2048Env, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=16,
            shape=(game.BOARD_SIZE, game.BOARD_SIZE),
            dtype=np.float32
        )

        self.board = None
        self.render_mode = render_mode

        # Reward shaping params
        self.kappa_invalid = kappa_invalid
        self.w_empty = w_empty
        self.w_mono = w_mono
        self.w_merge = w_merge
        self.gamma = gamma
        self.alpha = alpha
        self.w_snake = w_snake
        self.shaping_clip = shaping_clip
        self.debug = debug

    def _get_obs(self):
        """Returns the log2 representation of the board state for the agent."""
        with np.errstate(divide='ignore'):
            return np.where(self.board > 0, np.log2(self.board), 0.0).astype(np.float32)

    def _get_valid_moves(self, board):
        """Returns a boolean array indicating which moves are valid."""
        valid_moves = [False] * 4 # [up, right, down, left]
        for i, direction in enumerate(['up', 'right', 'down', 'left']):
            next_board, _ = game.move_board(board, direction)
            if not np.array_equal(board, next_board):
                valid_moves[i] = True
        return np.array(valid_moves)

    def _log_board(self, board):
        """Helper to get log2 of the board, mapping zeros to 0.0"""
        with np.errstate(divide='ignore'):
            return np.where(board > 0, np.log2(board), 0.0)

    def _heuristic_score(self, board):
        """
        Calculates a score representing the strategic quality of the board.
            This is the CORRECTED version with non-conflicting heuristics.
        """
        log_board = self._log_board(board)
        empty_score = np.count_nonzero(board == 0)
        merge_potential = 0
        for i in range(game.BOARD_SIZE):
            for j in range(game.BOARD_SIZE):
                if board[i, j] != 0:
                    if j < game.BOARD_SIZE - 1 and board[i, j] == board[i, j+1]:
                        merge_potential += log_board[i, j]
                    if i < game.BOARD_SIZE - 1 and board[i, j] == board[i+1, j]:
                        merge_potential += log_board[i, j]

        mono_score = 0
        mono_score += np.sum(log_board[:, :-1] - log_board[:, 1:])
        mono_score += np.sum(log_board[:-1, :] - log_board[1:, :])
        snake_score = np.sum(log_board * WEIGHT_MATRIX)

        return (self.w_empty * empty_score) + \
                (self.w_merge * merge_potential) + \
                (self.w_mono * mono_score) + \
                (self.w_snake * snake_score)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = game.get_empty_board()
        self.board = game.add_random_tile(self.board)
        self.board = game.add_random_tile(self.board)

        observation = self._get_obs()
        info = {'valid_moves': self._get_valid_moves(self.board)}
        
        if self.render_mode == 'human':
            self.render()
        return observation, info

    def step(self, action):
        action_map = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
        direction = action_map[int(action)]

        current_board = np.copy(self.board)
        h_old = self._heuristic_score(current_board)

        next_board, base_reward = game.move_board(self.board, direction)

        shaped_reward = 0.0
        if not np.array_equal(current_board, next_board):
            # Valid move
            self.board = game.add_random_tile(next_board)
            h_new = self._heuristic_score(self.board)

            norm = game.BOARD_SIZE * game.BOARD_SIZE
            phi_old = h_old / norm
            phi_new = h_new / norm

            shaped_reward = self.alpha * (self.gamma * phi_new - phi_old)
            shaped_reward = float(np.clip(shaped_reward, -self.shaping_clip, self.shaping_clip))

            total_reward = base_reward + shaped_reward
        else:
            # Invalid move
            self.board = current_board
            total_reward = -self.kappa_invalid

        terminated = game.is_game_over(self.board)
        truncated = False
        observation = self._get_obs()
        info = {
            "base_reward": base_reward,
            "shaped_reward": shaped_reward,
            "total_reward": total_reward,
            "valid_moves": self._get_valid_moves(self.board)
        }

        if self.debug:
            print(f"[DEBUG] Action={direction}, Base={base_reward:.2f}, Shaped={shaped_reward:.2f}, Total={total_reward:.2f}")

        if self.render_mode == 'human':
            self.render()
        return observation, float(total_reward), bool(terminated), bool(truncated), info

    def render(self, mode='human'):
        if mode == 'human':
            print("\nBoard:")
            print(self.board)
            print(f"Max tile: {int(np.max(self.board))}")