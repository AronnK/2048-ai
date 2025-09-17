# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import game

# class Game2048Env(gym.Env):
#     metadata = {'render_modes': ['human']}
    
#     def __init__(self, render_mode = None):
#         super(Game2048Env, self).__init__()
        
#         self.action_space = spaces.Discrete(4)
        
#         self.observation_space = spaces.Box(low=0, high=16, shape=(game.BOARD_SIZE, game.BOARD_SIZE), dtype=np.float32)
        
#         self.board = None
#         self.render_mode = render_mode

#     def _get_obs(self):
#         return np.log2(self.board, out=np.zeros_like(self.board, dtype=float), where=(self.board!=0)).astype(np.float32)

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.board = game.get_empty_board()
#         self.board = game.add_random_tile(self.board)
#         self.board = game.add_random_tile(self.board)
        
#         observation = self._get_obs()
#         info = {} 
#         if self.render_mode == 'human':
#             self.render()
#         return observation, info

#     def step(self, action):
#         action_map = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
#         direction = action_map[action]
        
#         current_board = np.copy(self.board)
        
#         next_board, reward = game.move_board(self.board, direction)
        
#         if not np.array_equal(current_board, next_board):
#             self.board = game.add_random_tile(next_board)
#             valid_move = True
#         else:
#             reward = -2
#             valid_move = False

#         terminated = game.is_game_over(self.board)
        
#         truncated = False
        
#         observation = self._get_obs()
#         info = {'valid_move': valid_move}

#         if self.render_mode == 'human':
#             self.render()
        
#         return observation, float(reward), terminated, truncated, info

#     def render(self, mode='human'):
#         if mode == 'human':
#             print(self.board)


'''
"Shaped Reward = (Heuristic Score of New State) - (Heuristic Score of Old State)
A common mistake is to give a reward for simply being in a good state.
This can be exploited. Instead, you should reward the agent for transitioning to a better state."

Gonna try shaping the reward system better to get better results.

1. Max tile in corner:
R
corner
={
+β if max tile is in a corner  
0 otherwise
}

2. Monotonicity reward:
Rmono=-∑|log2(board[i,j])-log2(board[i,j+1])|

3. Smoothness reward:
Rsmooth=-∑|log2(board[i,j])-log2(board[ni,nj])|

4. Max tile growth reward:
Rmax=η⋅(new max tile-old max tile)

5. Invalid move penalty:
Rinvalid=−κ

6. 2048 reached reward:
Rwin=+μ
'''

# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import game

# class Game2048Env(gym.Env):
#     """
#     2048 Gym environment with reward shaping:
#       - shaped reward = heuristic(new_state) - heuristic(old_state)
#       - plus explicit terms:
#           * corner bonus (if max tile sits in a corner)
#           * max-tile growth bonus
#           * invalid-move penalty
#           * win reward (first time reaching 2048)
#       - returns observation, reward, terminated, truncated, info (Gymnasium API)
#     """

#     metadata = {'render_modes': ['human']}

#     def __init__(self, render_mode=None,
#                  beta_corner=2.0,
#                  eta_max_growth=0.05,
#                  kappa_invalid=2.0,
#                  mu_win=10.0,
#                  w_mono=1.0,
#                  w_smooth=0.1,
#                  w_tilecount=0.5,
#                  w_maxlog=0.5):
#         super(Game2048Env, self).__init__()

#         self.action_space = spaces.Discrete(4)
#         self.observation_space = spaces.Box(low=0, high=16, shape=(game.BOARD_SIZE, game.BOARD_SIZE), dtype=np.float32)

#         self.board = None
#         self.render_mode = render_mode

#         self.beta_corner = beta_corner
#         self.eta_max_growth = eta_max_growth
#         self.kappa_invalid = kappa_invalid
#         self.mu_win = mu_win
#         self.w_mono = w_mono
#         self.w_smooth = w_smooth
#         self.w_tilecount = w_tilecount
#         self.w_maxlog = w_maxlog
#         self._last_max_tile = 0

#     def _get_obs(self):
#         with np.errstate(divide='ignore'):
#             obs = np.where(self.board > 0, np.log2(self.board), 0.0).astype(np.float32)
#         return obs

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.board = game.get_empty_board()
#         self.board = game.add_random_tile(self.board)
#         self.board = game.add_random_tile(self.board)

#         self._last_max_tile = int(np.max(self.board))
#         observation = self._get_obs()
#         info = {}
#         if self.render_mode == 'human':
#             self.render()
#         return observation, info

#     def _log_board(self, board):
#         """Return log2(board) with zeros mapped to 0.0"""
#         with np.errstate(divide='ignore'):
#             return np.where(board > 0, np.log2(board), 0.0)

#     def _max_tile_in_corner(self, board):
#         """Return True if the maximum tile sits in one of the four corners."""
#         max_tile = np.max(board)
#         n = board.shape[0]
#         corners = [board[0, 0], board[0, n-1], board[n-1, 0], board[n-1, n-1]]
#         return max_tile in corners

#     def _monotonicity(self, board):
#         """
#         Monotonicity measure: sum of absolute differences between adjacent tiles along rows and columns.
#         Lower is more monotonic. We'll return the total absolute adjacent diff (without the negative sign).
#         The reward will be -w_mono * monotonicity.
#         """
#         logb = self._log_board(board)
#         n = board.shape[0]
#         row_diff = np.sum(np.abs(logb[:, 1:] - logb[:, :-1]))
#         col_diff = np.sum(np.abs(logb[1:, :] - logb[:-1, :]))
#         return row_diff + col_diff

#     def _smoothness(self, board):
#         """
#         Smoothness: sum of absolute differences between neighbors (each neighbor pair counted once).
#         We'll compute right and down differences to avoid double-counting.
#         """
#         logb = self._log_board(board)
#         right_diff = np.sum(np.abs(logb[:, 1:] - logb[:, :-1]))
#         down_diff = np.sum(np.abs(logb[1:, :] - logb[:-1, :]))
#         return right_diff + down_diff

#     def _nonzero_count(self, board):
#         return int(np.count_nonzero(board))

#     def _heuristic_score(self, board):
#         """
#         Combined heuristic score for a state (higher is better).
#         We construct so that monotonicity and smoothness contribute negatively (we want small diffs).
#         Also add a (positive) contribution from the log of the max tile (encourages building large tiles).
#         And penalize number of tiles (encourages emptier board).
#         """
#         max_tile = np.max(board)
#         max_log = 0.0 if max_tile == 0 else float(np.log2(max_tile))

#         mono = self._monotonicity(board)
#         smooth = self._smoothness(board)
#         nonzero = self._nonzero_count(board)

#         # heuristic composition (higher better)
#         score = 0.0
#         score += - self.w_mono * mono
#         score += - self.w_smooth * smooth
#         score += - self.w_tilecount * nonzero
#         score += self.w_maxlog * max_log
#         return float(score)

#     def step(self, action):
#         action_map = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
#         direction = action_map[int(action)]

#         current_board = np.copy(self.board)
#         old_max = int(np.max(current_board))

#         next_board, base_reward = game.move_board(self.board, direction)

#         if not np.array_equal(current_board, next_board):
#             new_board = game.add_random_tile(next_board)
#             valid_move = True
#         else:
#             new_board = np.copy(self.board)  
#             valid_move = False
#             base_reward = -1.0
#         invalid_penalty = -float(self.kappa_invalid) if not valid_move else 0.0

#         h_old = self._heuristic_score(current_board)
#         h_new = self._heuristic_score(new_board)
#         shaped_delta = (h_new - h_old)

#         new_max = int(np.max(new_board))
#         max_growth_bonus = self.eta_max_growth * float(new_max - old_max)

#         corner_bonus = 0.0
#         if self._max_tile_in_corner(new_board):
#             corner_bonus = float(self.beta_corner)

#         win_bonus = 0.0
#         if new_max >= 2048 and old_max < 2048:
#             win_bonus = float(self.mu_win)

#         total_reward = float(base_reward) + shaped_delta + max_growth_bonus + corner_bonus + win_bonus + invalid_penalty

#         self.board = np.copy(new_board)
#         self._last_max_tile = new_max

#         terminated = game.is_game_over(self.board)

#         truncated = False

#         observation = self._get_obs()
#         info = {
#             'valid_move': valid_move,
#             'shaped_components': {
#                 'base_reward': float(base_reward),
#                 'heuristic_old': float(h_old),
#                 'heuristic_new': float(h_new),
#                 'shaped_delta': float(shaped_delta),
#                 'max_growth_bonus': float(max_growth_bonus),
#                 'corner_bonus': float(corner_bonus),
#                 'win_bonus': float(win_bonus),
#                 'invalid_move_penalty': float(invalid_penalty),
#                 'total_reward': float(total_reward),
#                 'old_max': int(old_max),
#                 'new_max': int(new_max),
#                 'nonzero_tiles': int(self._nonzero_count(self.board))
#             }
#         }

#         if self.render_mode == 'human':
#             self.render()

#         return observation, float(total_reward), bool(terminated), bool(truncated), info

#     def render(self, mode='human'):
#         if mode == 'human':
#             print("Board:")
#             print(self.board)
#             print(f"Max tile: {int(np.max(self.board))}")

'''
Nvm.... 
The model just collapsed with this compicated reward function...
Episode 0, reward: 620.3000000000004, epsilon: 1.00
Model saved to 2048_cnn_ddqn.pth
Episode 100, reward: 1100.9999999999995, epsilon: 0.48
Model saved to 2048_cnn_ddqn.pth
Episode 200, reward: 1041.7999999999993, epsilon: 0.21
Model saved to 2048_cnn_ddqn.pth
Episode 300, reward: 870.7000000000002, epsilon: 0.05
Model saved to 2048_cnn_ddqn.pth
Episode 400, reward: -167.39999999999998, epsilon: 0.01
Model saved to 2048_cnn_ddqn.pth
Episode 500, reward: -0.2999999999999119, epsilon: 0.01
Model saved to 2048_cnn_ddqn.pth
Episode 600, reward: -181.6999999999998, epsilon: 0.01
Model saved to 2048_cnn_ddqn.pth

And now Gemini is telling me that "Yes your reward shaping is too complex...
Like bro I asked you if it was good and you buttered me up and now you're telling it's complex... Urgh...
Anyways, gotta simplify it.
I can let the model learn some stuff on its own instead of teaching it at every turn as to what's good and bad.
Gotta teach it only the most important stuff...
'''

# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import game


# class Game2048Env(gym.Env):
#     """
#     2048 Gym environment with theoretically sound potential-based reward shaping.
#     The final reward is the sum of the base game score and the shaped heuristic reward.
#     """
#     metadata = {'render_modes': ['human']}

#     def __init__(self, render_mode=None,
#                  kappa_invalid=10.0,      # Penalty for invalid moves
#                  w_empty=2.0,            # Weight for empty cells
#                  w_mono=1.0,             # Weight for monotonicity
#                  w_merge=1.0,            # Weight for merge potential
#                  gamma=0.99,             # Discount factor for shaping
#                  alpha=0.5,              # Shaping scale
#                  shaping_clip=5.0,       # Clip the magnitude of the SHAPED part of the reward
#                  debug=False):           # Debug logging flag
#         super(Game2048Env, self).__init__()

#         self.action_space = spaces.Discrete(4)
#         self.observation_space = spaces.Box(
#             low=0, high=16,
#             shape=(game.BOARD_SIZE, game.BOARD_SIZE),
#             dtype=np.float32
#         )

#         self.board = None
#         self.render_mode = render_mode

#         # Reward shaping params
#         self.kappa_invalid = kappa_invalid
#         self.w_empty = w_empty
#         self.w_mono = w_mono
#         self.w_merge = w_merge
#         self.gamma = gamma
#         self.alpha = alpha
#         self.shaping_clip = shaping_clip
#         self.debug = debug

#     def _get_obs(self):
#         """Returns the log2 representation of the board state for the agent."""
#         with np.errstate(divide='ignore'):
#             return np.where(self.board > 0, np.log2(self.board), 0.0).astype(np.float32)

#     def _get_valid_moves(self, board):
#         """Returns a boolean array indicating which moves are valid."""
#         valid_moves = [False] * 4 # [up, right, down, left]
#         for i, direction in enumerate(['up', 'right', 'down', 'left']):
#             next_board, _ = game.move_board(board, direction)
#             if not np.array_equal(board, next_board):
#                 valid_moves[i] = True
#         return np.array(valid_moves)

#     def _log_board(self, board):
#         """Helper to get log2 of the board, mapping zeros to 0.0"""
#         with np.errstate(divide='ignore'):
#             return np.where(board > 0, np.log2(board), 0.0)

#     def _heuristic_score(self, board):
#         """
#         Calculates a score representing the strategic quality of the board.
#         Higher score = better board.
#         """
#         empty_score = np.count_nonzero(board == 0)
#         merge_potential = 0
#         log_board = self._log_board(board)
#         for i in range(game.BOARD_SIZE):
#             for j in range(game.BOARD_SIZE):
#                 if board[i, j] != 0:
#                     if j < game.BOARD_SIZE - 1 and board[i, j] == board[i, j+1]:
#                         merge_potential += log_board[i, j]
#                     if i < game.BOARD_SIZE - 1 and board[i, j] == board[i+1, j]:
#                         merge_potential += log_board[i, j]

#         mono_score = 0
#         for i in range(game.BOARD_SIZE):
#             row = log_board[i, :]
#             mono_score -= np.sum(np.maximum(0, row[:-1] - row[1:]))
#             col = log_board[:, i]
#             mono_score -= np.sum(np.maximum(0, col[:-1] - col[1:]))

#         return (self.w_empty * empty_score) + \
#                (self.w_merge * merge_potential) + \
#                (self.w_mono * mono_score)

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.board = game.get_empty_board()
#         self.board = game.add_random_tile(self.board)
#         self.board = game.add_random_tile(self.board)

#         observation = self._get_obs()
#         info = {'valid_moves': self._get_valid_moves(self.board)}
        
#         if self.render_mode == 'human':
#             self.render()
#         return observation, info

#     def step(self, action):
#         action_map = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
#         direction = action_map[int(action)]

#         current_board = np.copy(self.board)
#         h_old = self._heuristic_score(current_board)

#         next_board, base_reward = game.move_board(self.board, direction)

#         shaped_reward = 0.0
#         if not np.array_equal(current_board, next_board):
#             # Valid move
#             self.board = game.add_random_tile(next_board)
#             h_new = self._heuristic_score(self.board)

#             norm = game.BOARD_SIZE * game.BOARD_SIZE
#             phi_old = h_old / norm
#             phi_new = h_new / norm

#             shaped_reward = self.alpha * (self.gamma * phi_new - phi_old)
#             shaped_reward = float(np.clip(shaped_reward, -self.shaping_clip, self.shaping_clip))

#             total_reward = base_reward + shaped_reward
#         else:
#             # Invalid move
#             self.board = current_board
#             total_reward = -self.kappa_invalid

#         terminated = game.is_game_over(self.board)
#         truncated = False
#         observation = self._get_obs()
#         info = {
#             "base_reward": base_reward,
#             "shaped_reward": shaped_reward,
#             "total_reward": total_reward,
#             "valid_moves": self._get_valid_moves(self.board)
#         }

#         if self.debug:
#             print(f"[DEBUG] Action={direction}, Base={base_reward:.2f}, Shaped={shaped_reward:.2f}, Total={total_reward:.2f}")

#         if self.render_mode == 'human':
#             self.render()

#         # DELETED the final reward clipping line that was here.
#         return observation, float(total_reward), bool(terminated), bool(truncated), info

#     def render(self, mode='human'):
#         if mode == 'human':
#             print("\nBoard:")
#             print(self.board)
#             print(f"Max tile: {int(np.max(self.board))}")

'''
20k steps later we're consistently reaching 256, but monotonicity and max tile in corner is not being followed everytime, so gotta add those
in the reward. Model architecture can also be changed to incorporate PER and other stuff, but I want to push another 20-30k more steps with
this implementation, see how far I can push with this Vanilla DDQN CNN approach and then maybe go for a expectimax or PER approach...
'''


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
        valid_moves = [False] * 4 
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