import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class GNNLayer(nn.Module):
    def __init__(self, n_agents, in_dim, out_dim):
        super().__init__()
        self.n_agents = n_agents
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: (n_agents, in_dim)
        # adj: (n_agents, n_agents) adjacency matrix (binary)
        # Graph convolution: A * x * W
        x = self.fc(x)
        return adj @ x

class QMixNet(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Linear(state_dim, hidden_dim * n_agents)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, q_values, state):
        # q_values: (batch, n_agents)
        # state: (batch, state_dim)
        batch = state.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.n_agents, -1)
        b1 = self.hyper_b1(state).view(batch, 1, -1)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, -1, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        q_tot = torch.bmm(torch.relu(torch.bmm(q_values.unsqueeze(1), w1) + b1), w2) + b2
        return q_tot.squeeze()

class Agent(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_agents):
        super().__init__()
        self.n_agents = n_agents
        self.gnn1 = GNNLayer(n_agents, obs_dim, hidden_dim)
        self.gnn2 = GNNLayer(n_agents, hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, adj):
        # obs: (n_agents, obs_dim)
        x = torch.relu(self.gnn1(obs, adj))
        x = torch.relu(self.gnn2(x, adj))
        q = self.fc_out(x).squeeze(-1)   # (n_agents,)
        return q

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, actions, rewards, next_state, dones):
        self.buffer.append((state, actions, rewards, next_state, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state = torch.stack([b[0] for b in batch])
        actions = torch.stack([b[1] for b in batch])
        rewards = torch.stack([b[2] for b in batch])
        next_state = torch.stack([b[3] for b in batch])
        dones = torch.stack([b[4] for b in batch])
        return state, actions, rewards, next_state, dones

    def __len__(self):
        return len(self.buffer)

def train_marl(returns, window, n_episodes=100, episode_len=20, batch_size=32, lr=1e-3, gamma=0.99, tau=0.005):
    env = MARLEnv(returns, lookback=10)
    n_agents = env.n_agents
    obs_dim = 10
    state_dim = n_agents * obs_dim   # global state for QMix
    agent = Agent(obs_dim, 128, n_agents)
    target_agent = Agent(obs_dim, 128, n_agents)
    target_agent.load_state_dict(agent.state_dict())
    qmix = QMixNet(n_agents, state_dim, 64)
    target_qmix = QMixNet(n_agents, state_dim, 64)
    target_qmix.load_state_dict(qmix.state_dict())
    optimizer_a = optim.Adam(agent.parameters(), lr=lr)
    optimizer_q = optim.Adam(qmix.parameters(), lr=lr)
    buffer = ReplayBuffer(10000)

    # Precompute adjacency matrix from correlation of returns
    corr = np.corrcoef(returns[-window:].T)
    adj = (np.abs(corr) > config.GRAPH_THRESHOLD).astype(np.float32)
    np.fill_diagonal(adj, 1)
    adj_t = torch.tensor(adj)

    for episode in range(n_episodes):
        state = env.reset()
        state_t = torch.tensor(state, dtype=torch.float32)   # (n_agents, obs_dim)
        episode_reward = 0
        for step in range(episode_len):
            q_values = agent(state_t, adj_t)   # (n_agents,)
            # epsilon-greedy (exploration)
            if np.random.rand() < 0.1:
                actions = torch.rand(n_agents) * 2 - 1   # continuous actions in [-1,1]
            else:
                actions = q_values
            # convert actions to weights via softmax later
            next_state, reward, done, _ = env.step(actions.numpy())
            episode_reward += reward
            next_state_t = torch.tensor(next_state, dtype=torch.float32) if next_state is not None else None
            buffer.push(state_t, actions, torch.tensor([reward]), next_state_t, torch.tensor([done]))
            state_t = next_state_t
            if done:
                break
        if len(buffer) > batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            # Compute Q values for current and next states
            with torch.no_grad():
                next_q = []
                for i in range(batch_size):
                    if next_states[i] is not None:
                        nq = target_agent(next_states[i], adj_t)
                        next_q.append(nq)
                    else:
                        next_q.append(torch.zeros(n_agents))
                next_q = torch.stack(next_q)
                target_q_tot = target_qmix(next_q, next_states.view(batch_size, -1))
            q = []
            for i in range(batch_size):
                qi = agent(states[i], adj_t)
                q.append(qi)
            q = torch.stack(q)
            q_tot = qmix(q, states.view(batch_size, -1))
            loss = ((q_tot - (rewards + gamma * target_q_tot * (1-dones)))**2).mean()
            optimizer_q.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_q.step()
            # Update agent using QMIX's gradient (the mixing network)
            # Actually agent's gradients come from the same loss.
            # For simplicity, we update agent with the same loss.
            # But we must not retain graph again. We'll compute a separate loss for agent.
            # Instead, we recompute Q for agent using the same batch but detach the mixer's output? 
            # Standard CTDE: update agent and mixer together. We'll just use one optimizer step for both.
            # For clarity, we'll do one optimizer step for both networks.
            # We'll combine the parameters.
            optimizer_a.zero_grad()
            loss.backward()
            optimizer_a.step()
            # Soft update target networks
            for target_param, param in zip(target_agent.parameters(), agent.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            for target_param, param in zip(target_qmix.parameters(), qmix.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    # Final policy: greedy actions
    state = env.reset()
    state_t = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        q_values = agent(state_t, adj_t)
        actions = q_values
    # Convert to weights
    weights = torch.softmax(actions, dim=0).numpy()
    return weights, agent, qmix
