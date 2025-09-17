import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import time
from env import Game2048Env
import game 

MODEL_FILE = "2048_best_final.pth" 
NUM_EPISODES = 10
ACTION_DELAY = 0.0001

class CNN_DuelingDQN(nn.Module):
    def __init__(self, in_channels, action_dim):
        super(CNN_DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2), nn.ReLU()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 4, 4)
            conv_out_size = self.conv(dummy_input).flatten().shape[0]
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class ExpectiMaxAgent:
    """
    This agent uses a trained neural network to evaluate future board states
    and an Expectimax search to choose the best immediate move.
    """
    def __init__(self, model_path, device):
        self.device = device
        print("Loading model for ExpectiMax Agent...")

        self.model = CNN_DuelingDQN(1, 4).to(self.device)
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            exit()
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['q_net'])
        self.model.eval()
        print("Model loaded successfully.")
        
    def _preprocess_state(self, board):
        """Preprocesses the raw board to the log2 format for the network."""
        with np.errstate(divide='ignore'):
            processed = np.where(board > 0, np.log2(board), 0.0)
        return processed.astype(np.float32)

    def _evaluate_board(self, board):
        """Uses the trained NN to assign a single score to a board state."""
        with torch.no_grad():
            state_processed = self._preprocess_state(board)
            state_tensor = torch.FloatTensor(state_processed).unsqueeze(0).unsqueeze(0).to(self.device)
            value = self.model(state_tensor).mean().item()
        return value

    def select_move(self, env):
        """
        Performs a 1-ply Expectimax search to find the best move.
        """
        board = env.board
        valid_moves = env._get_valid_moves(board)
        best_move = -1
        best_score = -float('inf')

        for move_idx, is_valid in enumerate(valid_moves):
            if is_valid:
                action_str = ['up', 'right', 'down', 'left'][move_idx]

                board_after_move, _ = game.move_board(np.copy(board), action_str)
 
                empty_cells = list(zip(*np.where(board_after_move == 0)))
                
                if not empty_cells:

                    score = self._evaluate_board(board_after_move)
                else:
                    expected_score = 0
                    for r, c in empty_cells:
                        temp_board = np.copy(board_after_move)
                        temp_board[r, c] = 2
                        expected_score += 0.9 * self._evaluate_board(temp_board)
                    for r, c in empty_cells:
                        temp_board = np.copy(board_after_move)
                        temp_board[r, c] = 4
                        expected_score += 0.1 * self._evaluate_board(temp_board)

                    score = expected_score / len(empty_cells)

                if score > best_score:
                    best_score = score
                    best_move = move_idx
                    
        return best_move if best_move != -1 else np.random.choice(np.where(valid_moves)[0])

device = torch.device("cpu")
print(f"Using device: {device}")

env = Game2048Env(render_mode='human')
agent = ExpectiMaxAgent(MODEL_FILE, device=device)

all_scores = []
all_max_tiles = []

for epi in range(NUM_EPISODES):
    state, info = env.reset()
    done = False
    game_score = 0
    
    print(f"\n--- Starting Evaluation Episode {epi + 1}/{NUM_EPISODES} ---")

    while not done:

        action = agent.select_move(env)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        state = next_state
        game_score += info['base_reward']
        
        time.sleep(ACTION_DELAY)

    max_tile = int(np.max(env.board))
    all_scores.append(game_score)
    all_max_tiles.append(max_tile)
    
    print(f"Episode {epi + 1} finished.")
    print(f"  > Final Score (from merges): {int(game_score)}")
    print(f"  > Max Tile Reached: {max_tile}")

env.close()

print("\n--- Evaluation Summary ---")
print(f"Average Score: {np.mean(all_scores):.2f}")
print(f"Average Max Tile: {np.mean(all_max_tiles):.2f}")
print(f"Highest Score: {np.max(all_scores):.0f}")
print(f"Highest Tile Reached: {np.max(all_max_tiles)}")
print("\nEvaluation finished.")