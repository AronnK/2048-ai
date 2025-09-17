import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import torch.optim as optim
import random

env = gym.make("CartPole-v1")
action_space_size = env.action_space.n
state_space_size = 4

class DQN(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_space_size, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, action_space_size)
        )
    def forward(self, x):
        return self.fc(x)

qnet = DQN(state_space_size, action_space_size)
tnet = DQN(state_space_size, action_space_size)
tnet.load_state_dict(qnet.state_dict())

optimizer = optim.Adam(qnet.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

memory = deque(maxlen=10000)

batch_size = 64
gamma = 0.99
epsilon = 1.0
eps_min = 0.01
eps_decay = 0.99995
episodes = 2000
target_update = 50


for epi in range(episodes):
    curr_state, _ = env.reset()
    curr_state = curr_state
    done = False
    tot_rew = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = qnet(torch.FloatTensor(curr_state))
                action = torch.argmax(q_values).item()


        next_state, rew, term, trun, _ = env.step(action)
        done = term or trun
        next_state = next_state

        memory.append((curr_state, action, rew, next_state, done))

        curr_state = next_state
        tot_rew += rew

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            curr_states, actions, rewards, next_states, dones = zip(*batch)

            curr_states = torch.FloatTensor(curr_states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            q_values = qnet(curr_states).gather(1, actions)

            with torch.no_grad():
                max_next_q = tnet(next_states).max(1)[0].unsqueeze(1)
                q_target = rewards + gamma * max_next_q * (1 - dones)

            loss = loss_fn(q_values, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epsilon = max(eps_min, epsilon * eps_decay)

    if epi % target_update == 0:
        tnet.load_state_dict(qnet.state_dict())

    if epi % 100 == 0:
        print(f"Episode {epi}, reward: {tot_rew}, epsilon: {epsilon:.2f}")

'''
Okay so apprently CartPole is easier than my implementation of CNN+DDQN arch Frozen Lake as the agent gets most of the information needed.
I can build more on this but I feel like it's not that worth the time and so do Gemini and ChatGPT...
So I guess we will be moving on the Atari games from here, or just a straight jump to why I even started learning RL, the 2048 game.

'''