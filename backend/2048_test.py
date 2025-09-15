# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import time
# from env import Game2048Env

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def preprocess_state(state):
#     state_nz = state.copy()
#     state_nz[state_nz == 0] = 1 # Avoid log2(0)
#     processed = np.log2(state_nz) / 16.0 
#     return processed.reshape(1, 4, 4).astype(np.float32)

# class CNN_DQN(nn.Module):
#     def __init__(self, in_channels, action_dim):
#         super(CNN_DQN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=2),
#             nn.ReLU()
#         )
#         grid_size = 4
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, in_channels, grid_size, grid_size)
#             conv_out_size = self.conv(dummy_input).flatten().shape[0]
#         self.fc = nn.Sequential(
#             nn.Linear(conv_out_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_dim)
#         )
#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)


# MODEL_FILE = "2048_cnn_ddqn.pth"
# NUM_EPISODES = 10
# ACTION_DELAY = 0.5

# env = Game2048Env(render_mode='human')
# action_space_size = env.action_space.n
# in_channels = 1

# q_net = CNN_DQN(in_channels, action_space_size).to(device)

# print(f"Loading model from {MODEL_FILE}...")
# checkpoint = torch.load(MODEL_FILE, map_location=device)
# q_net.load_state_dict(checkpoint['q_net'])

# q_net.eval()
# print("Model loaded successfully!")

# for epi in range(NUM_EPISODES):
#     state, _ = env.reset()
#     done = False
#     total_reward = 0
    
#     print(f"\n--- Starting Episode {epi + 1}/{NUM_EPISODES} ---")

#     while not done:
#         state_processed = preprocess_state(state)
#         state_tensor = torch.FloatTensor(state_processed).unsqueeze(0).to(device)
       
#         with torch.no_grad():
#             q_values = q_net(state_tensor)
        
#         action = torch.argmax(q_values).item()

#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
        
#         state = next_state
#         total_reward += reward
        
#         time.sleep(ACTION_DELAY)
        
#     print(f"Episode {epi + 1} finished with a total score of: {total_reward}")

# env.close()
# print("\nEvaluation finished.")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import time
from env import Game2048Env


MODEL_FILE = "2048_cnn_ddqn.pth"
NUM_EPISODES = 10
ACTION_DELAY = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CNN_DQN(nn.Module):
    """
    The same CNN architecture used for training.
    """
    def __init__(self, in_channels, action_dim):
        super(CNN_DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2),
            nn.ReLU()
        )
        grid_size = 4
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, grid_size, grid_size)
            conv_out_size = self.conv(dummy_input).flatten().shape[0]
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

env = Game2048Env(render_mode='human')
action_space_size = env.action_space.n
in_channels = 1

q_net = CNN_DQN(in_channels, action_space_size).to(device)

try:
    print(f"Loading model from {MODEL_FILE}...")
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    q_net.load_state_dict(checkpoint['q_net'])
    q_net.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_FILE}. Please ensure the file exists.")
    exit()


for epi in range(NUM_EPISODES):
    state, info = env.reset()
    valid_moves = info['valid_moves']
    
    done = False
    total_game_score = 0
    
    print(f"\n--- Starting Episode {epi + 1}/{NUM_EPISODES} ---")

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values = q_net(state_tensor)
            valid_moves_tensor = torch.BoolTensor(valid_moves).to(device)
            q_values[0, ~valid_moves_tensor] = -float('inf')
        action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        valid_moves = info['valid_moves']
        total_game_score += info['base_reward']
        time.sleep(ACTION_DELAY)
        
    print(f"Episode {epi + 1} finished with a total game score of: {int(total_game_score)}")

env.close()
print("\nEvaluation finished.")