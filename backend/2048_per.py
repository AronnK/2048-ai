import os
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from env import Game2048Env 

class SumTree:
    """A SumTree is a binary tree where each leaf is a priority, and each parent
    is the sum of its children. This allows for efficient, weighted sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PERMemory:
    """The memory buffer that uses a SumTree to implement PER."""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.e = 0.01

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.alpha

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, torch.FloatTensor(is_weight).to(device)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries

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

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
env = Game2048Env(w_snake=7.5) 
action_space_size = env.action_space.n
qnet = CNN_DuelingDQN(1, action_space_size).to(device)
tnet = CNN_DuelingDQN(1, action_space_size).to(device)
tnet.load_state_dict(qnet.state_dict())

optimizer = optim.Adam(qnet.parameters(), lr=1e-4)
memory = PERMemory(100000)
batch_size = 64
gamma = 0.99
# epsilon = 1.0
epsilon = 0.4
eps_min = 0.01
eps_decay = 0.999995
episodes = 100000
model_file = "2048_best_new.pth"
best_avg_reward = -float('inf')
reward_window = deque(maxlen=100)
start_episode = 0
model_file_new = "2048_best_final.pth"

if os.path.exists(model_file):
    print("Loading checkpoint to resume training...")
    checkpoint = torch.load(model_file, map_location=device)
    qnet.load_state_dict(checkpoint['q_net'])
    tnet.load_state_dict(checkpoint['target_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_episode = checkpoint['episode'] + 1
    # epsilon = checkpoint.get('epsilon', epsilon) 
    print(f"Resuming from episode {start_episode} with epsilon {epsilon:.4f}")

for epi in range(start_episode, episodes):
    curr_state, info = env.reset()
    valid_moves = info["valid_moves"]
    done = False
    tot_rew = 0

    while not done:
        if random.random() < epsilon:
            action = np.random.choice(np.where(valid_moves)[0])
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(curr_state).unsqueeze(0).unsqueeze(0).to(device)
                q_values = qnet(state_tensor)
                q_values[0, ~valid_moves] = -float('inf')
                action = torch.argmax(q_values).item()

        next_state, rew, term, trun, info = env.step(action)
        valid_moves = info['valid_moves']
        done = term or trun
        
        with torch.no_grad():
            state_t = torch.FloatTensor(curr_state).unsqueeze(0).unsqueeze(0).to(device)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(device)
            q_value = qnet(state_t)[0][action]
            next_q_values_online = qnet(next_state_t)
            best_next_action = next_q_values_online.argmax(1).item()
            next_q_from_target = tnet(next_state_t)[0][best_next_action]
            q_target_val = rew + gamma * next_q_from_target * (1 - done)
            error = abs(q_value - q_target_val).item()
        
        memory.add(error, (curr_state, action, rew, next_state, done))

        curr_state = next_state
        tot_rew += rew

        if len(memory) >= batch_size:
            mini_batch, idxs, is_weights = memory.sample(batch_size)
            curr_states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*mini_batch)

            curr_states_b = torch.FloatTensor(np.array(curr_states_b)).unsqueeze(1).to(device)
            actions_b = torch.LongTensor(actions_b).unsqueeze(1).to(device)
            rewards_b = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
            next_states_b = torch.FloatTensor(np.array(next_states_b)).unsqueeze(1).to(device)
            dones_b = torch.FloatTensor(dones_b).unsqueeze(1).to(device)

            q_values = qnet(curr_states_b).gather(1, actions_b)
            
            next_q_values_online = qnet(next_states_b)
            best_next_actions = next_q_values_online.argmax(1).unsqueeze(1)
            with torch.no_grad():
                next_q_from_target = tnet(next_states_b).gather(1, best_next_actions)
                q_target = rewards_b + gamma * next_q_from_target * (1 - dones_b)
            
            td_errors = q_target - q_values
            loss = (is_weights.unsqueeze(1) * td_errors.pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for i in range(batch_size):
                memory.update(idxs[i], td_errors[i].item())

            tau = 0.005
            for target_param, online_param in zip(tnet.parameters(), qnet.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
        
        epsilon = max(eps_min, epsilon * eps_decay)
    
    reward_window.append(tot_rew)
    if (epi + 1) % 100 == 0:
        avg_reward = np.mean(reward_window)
        print(f"Episode {epi+1}, Avg Reward (100 epi): {avg_reward:.2f}, Epsilon: {epsilon:.4f}, Beta: {memory.beta:.4f}")
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            
            print(f"*** New best model saved at episode {epi+1} with avg reward: {best_avg_reward:.2f} ***")
            torch.save({
                'episode': epi,
                'q_net': qnet.state_dict(),
                'target_net': tnet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epsilon': epsilon,
            }, model_file_new)


'''
After around 35k training episodes I feel like my model has reached a plateau cuz for the last 3-4k episodes there have been no improvements...
The current best 2048_best_new can reach 1024 10% of the time, 512 (with like another 512 or 2-3 256s and 128s just sitting around) 
80% of the time and ends in 256 or lower itself 10% of the time.

I'm gonna try to run another 5k episodes with 0.4 epsilon and 1e-4 learning rate, hopefully I see some improvements. 
(Gemini said it was a bad idea tho...)

Also I'm gonna try to pair this current best model with a ExpectiMax algorithm and see how it goes.

If adding exploration rate doesn't help, I'm gonna redesign the architecture and the rewards and whatnot one last time...
Maybe add in a few more conv layers, do something new with the state representation or something... 
Idk I'll just fiddle with it, hopefully something works... If it doesn't, then, it will be the end of 2048 project.... :)
'''