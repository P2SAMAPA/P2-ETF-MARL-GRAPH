import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MARLEnv:
    def __init__(self, returns_df, lookback=10):
        self.returns = returns_df.values
        self.n_agents = returns_df.shape[1]
        self.lookback = lookback
        self.current_step = lookback
        self.done = False

    def reset(self):
        self.current_step = self.lookback
        self.done = False
        return self._get_state()

    def _get_state(self):
        state = self.returns[self.current_step - self.lookback:self.current_step, :].T
        return torch.tensor(state, dtype=torch.float32)

    def step(self, actions):
        if self.current_step >= len(self.returns) - 1:
            self.done = True
            return self._get_state(), 0.0, self.done, {}
        daily_returns = self.returns[self.current_step, :]
        buy_mask = actions == 1
        if np.any(buy_mask):
            port_ret = np.mean(daily_returns[buy_mask])
        else:
            port_ret = 0.0
        self.current_step += 1
        next_state = self._get_state()
        return next_state, port_ret, self.done, {}

class QMixNet(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=64):
        super().__init__()
        self.agent_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, 2)) for _ in range(n_agents)
        ])

    def forward(self, states):
        agent_qs = []
        for i, net in enumerate(self.agent_nets):
            q = net(states[i])
            agent_qs.append(q)
        return torch.stack(agent_qs)

def train_marl(returns_df, window, n_episodes=100, episode_len=None,
               lr=1e-3, gamma=0.99, batch_size=None, tau=None):
    lookback = min(window, 10) if window > 10 else window
    env = MARLEnv(returns_df.iloc[-window:], lookback=lookback)
    n_agents = env.n_agents
    state_dim = lookback
    qmix = QMixNet(n_agents, state_dim)
    optimizer = optim.Adam(qmix.parameters(), lr=lr)
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                agent_qs = qmix(state)
                actions = agent_qs.argmax(dim=1).numpy()
            next_state, reward, done, _ = env.step(actions)
            total_reward += reward
            state = next_state
        if (ep+1) % 20 == 0:
            print(f"    Episode {ep+1}/{n_episodes}, total reward: {total_reward:.4f}")
    # After training, evaluate final weights using the last state
    final_state = env.reset()  # reset to initial? better to use the last state from training? Simpler: use the most recent state
    # We'll reuse the last state from the final episode: `state` after the loop is the last state? Actually after loop, `state` is the state after the last step (which could be terminal). We'll re-get a fresh state.
    final_state = env.reset()
    with torch.no_grad():
        agent_qs = qmix(final_state)
        weights = agent_qs[:, 1].numpy()
    return weights, None, None

def get_weights(qmix, state):
    with torch.no_grad():
        agent_qs = qmix(state)
        return agent_qs[:, 1].numpy()
