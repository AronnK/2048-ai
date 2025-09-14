import os
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from env import Game2048Env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = Game2048Env()
action_space_size = env.action_space.n
state_space_size = 16

# class DQN(nn.Module):
#     def __init__(self, state_space_size, action_space_size):
#         super(DQN, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(state_space_size, 256),
#             nn.ReLU(),
#             nn.Linear(256,128),
#             nn.ReLU(),
#             nn.Linear(128,64),
#             nn.ReLU(),
#             nn.Linear(64, action_space_size)
#         )
#     def forward(self, x):
#         return self.fc(x)

class CNN_DQN(nn.Module):
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
        # print("Tensor shape entering the model:", x.shape)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

in_channels = 1

qnet = CNN_DQN(in_channels, action_space_size).to(device)
tnet = CNN_DQN(in_channels, action_space_size).to(device)
tnet.load_state_dict(qnet.state_dict())

optimizer = optim.Adam(qnet.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
loss_fn = nn.SmoothL1Loss()

memory = deque(maxlen=100000)

batch_size = 64
gamma = 0.99
epsilon = 1.0
eps_min = 0.02
eps_decay = 0.9999998
episodes = 50000
# target_update = 200
best_avg_reward = -float('inf')
reward_window = deque(maxlen=100)
model_file = "2048_cnn_ddqn.pth"

if os.path.exists(model_file):
    print("Loading existing model to resume training...")
    checkpoint = torch.load(model_file, map_location=device)
    qnet.load_state_dict(checkpoint['q_net'])
    tnet.load_state_dict(checkpoint['target_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # epsilon = checkpoint.get('epsilon', epsilon)

for epi in range(episodes):
    curr_state, info = env.reset()
    valid_moves = info["valid_moves"]
    done = False
    tot_rew = 0

    while not done:
        if random.random() < epsilon:
            action = np.random.choice(np.where(valid_moves)[0])
            # action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(curr_state).unsqueeze(0).unsqueeze(0).to(device)
                q_values = qnet(state_tensor)
                q_values[0, ~valid_moves] = -float('inf')
                action = torch.argmax(q_values).item()

        next_state, rew, term, trun, info = env.step(action)
        valid_moves = info['valid_moves']
        done = term or trun

        memory.append((curr_state, action, rew, next_state, done))

        curr_state = next_state
        tot_rew += rew

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            curr_states, actions, rewards, next_states, dones = zip(*batch)

            curr_states = torch.FloatTensor(np.array(curr_states)).unsqueeze(1).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_values = qnet(curr_states).gather(1, actions)
            next_q_values_online = qnet(next_states).clone()
            best_next_actions = next_q_values_online.argmax(1).unsqueeze(1)
            with torch.no_grad():
                next_q_from_target = tnet(next_states).gather(1, best_next_actions)
                q_target = rewards + gamma * next_q_from_target * (1 - dones)

            loss = loss_fn(q_values, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tau = 0.005
            for target_param, online_param in zip(tnet.parameters(), qnet.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
        
        epsilon = max(eps_min, epsilon * eps_decay)
    scheduler.step()
    reward_window.append(tot_rew)
    if len(reward_window) == 100:
        avg_reward = np.mean(reward_window)
        print(f"Episode {epi}, Avg Reward (100 epi): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save({
                'q_net': qnet.state_dict(),
                'target_net': tnet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epsilon': epsilon,
            }, model_file)
            print(f"*** New best model saved with avg reward: {best_avg_reward:.2f} ***")
    # if epi % target_update == 0:
    #     tnet.load_state_dict(qnet.state_dict())

    if epi % 100 == 0:
        print(f"Episode {epi}, reward: {tot_rew}, epsilon: {epsilon:.2f}")
