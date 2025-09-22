import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import time
from env import Game2048Env

MODEL_FILE = "2048_best_final.pth"
NUM_EPISODES = 10
ACTION_DELAY = 0.00001

class CNN_DuelingDQN(nn.Module):
    """The Dueling DQN architecture for better state-value estimation."""
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
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

device = torch.device("cpu")
print(f"Using device: {device}")

env = Game2048Env(render_mode='human')
action_space_size = env.action_space.n

q_net = CNN_DuelingDQN(1, action_space_size).to(device)

if not os.path.exists(MODEL_FILE):
    print(f"Error: Model file not found at {MODEL_FILE}. Please run the training script first.")
    exit()

print(f"Loading model from {MODEL_FILE}...")
checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)
q_net.load_state_dict(checkpoint['q_net'])
q_net.eval()
print("Model loaded successfully!")

all_scores = []
all_max_tiles = []

for epi in range(NUM_EPISODES):
    state, info = env.reset()
    valid_moves = info['valid_moves']
    done = False
    
    total_shaped_reward = 0
    game_score = 0
    
    print(f"\n--- Starting Evaluation Episode {epi + 1}/{NUM_EPISODES} ---")

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_net(state_tensor)
            q_values[0, ~valid_moves] = -float('inf')
            action = torch.argmax(q_values).item()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        state = next_state
        valid_moves = info['valid_moves']

        total_shaped_reward += reward
        game_score += info['base_reward']
        time.sleep(ACTION_DELAY)

    max_tile = int(np.max(env.board))
    all_scores.append(game_score)
    all_max_tiles.append(max_tile)
    
    print(f"Episode {epi + 1} finished.")
    print(f"  > Final Score (from merges): {int(game_score)}")
    print(f"  > Max Tile Reached: {max_tile}")
    print(f"  > Total Shaped Reward: {total_shaped_reward:.2f}")

env.close()

print("\n--- Evaluation Summary ---")
print(f"Average Score: {np.mean(all_scores):.2f}")
print(f"Average Max Tile: {np.mean(all_max_tiles):.2f}")
print(f"Highest Score: {np.max(all_scores):.0f}")
print(f"Highest Tile Reached: {np.max(all_max_tiles)}")
print("\nEvaluation finished.")