# Multi-Agent RL (MARL) with Graph Communication

QMIX with graph neural network communication. Each ETF is an agent that observes its own recent returns and communicates with correlated ETFs via a graph (adjacency from correlation >0.5). The mixing network QMIX learns to decompose the global portfolio return into individual agent contributions using centralised training with decentralised execution (CTDE). Outputs are the learned policy weights (softmax of Q-values).

- **Algorithm:** QMIX + GNN
- **Graph:** correlation matrix thresholded at 0.5
- **Environment:** multi‑agent portfolio allocation
- **Training:** CTDE with replay buffer and soft updates
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs by weight

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
