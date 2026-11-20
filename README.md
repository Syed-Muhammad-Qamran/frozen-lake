# ❄️ FrozenLake Q-Learning Agent & Interactive Streamlit Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-brightgreen.svg)](https://gymnasium.farama.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end **Reinforcement Learning** project implementing **Tabular Q-Learning** from scratch to solve the classical **FrozenLake-v1** environment from Farama Gymnasium. Features an interactive, feature-rich **Streamlit web dashboard** to train, visualize, and simulate the agent in real time.

---

## 📌 Project Overview

**FrozenLake** is a classic discrete Reinforcement Learning puzzle. The agent controls an entity on a $4 \times 4$ grid:
```
S F F F       S : Start (Safe)
F H F H       F : Frozen surface (Safe)
F F F H       H : Hole (Game Over, Reward = 0)
H F F G       G : Goal (Reward = +1)
```

The objective is to navigate from the starting point (`S`) to the goal (`G`) without falling into any fatal holes (`H`).

### Key Highlights
* **Pure Python Implementation:** Built from first principles without heavy high-level wrappers.
* **Deterministic & Stochastic Modes:** Supports both non-slippery ice (100% success rate) and slippery stochastic ice (~74% optimal limit).
* **Interactive Streamlit Web App:** Real-time agent training, step-by-step game visualization, policy maps, and full Q-table inspection.
* **Rigorous Mathematical Foundations:** Complete formulation with Markov Decision Processes (MDPs) and Bellman optimality updates.

---

## 🧠 Reinforcement Learning Theory

### 1. Markov Decision Process (MDP)
FrozenLake is modeled as an MDP $(S, A, P, R, \gamma)$:
* **State Space ($S$):** 16 discrete tiles ($0$ through $15$).
* **Action Space ($A$):** 4 directional moves:
  * `0`: Left ($\leftarrow$)
  * `1`: Down ($\downarrow$)
  * `2`: Right ($\rightarrow$)
  * `3`: Up ($\uparrow$)
* **Reward Function ($R$):**
  * $+1.0$ upon reaching Goal (`G`).
  * $0.0$ for all other transitions (including falling into a hole).
* **Discount Factor ($\gamma$):** Balances immediate vs. future returns ($\gamma \in [0, 1)$).

---

### 2. The Q-Learning Algorithm
Q-Learning is a **Model-Free, Off-Policy Temporal Difference (TD) Control** algorithm. The agent maintains a table $Q(s, a)$ of size $16 \times 4$.

#### Update Equation
At each step $(S_t, A_t, R_{t+1}, S_{t+1})$, the Q-value is updated via:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[ \underbrace{R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)}_{\text{TD Target}} - \underbrace{Q(S_t, A_t)}_{\text{Current Estimate}} \Big]$$

* **$\alpha$ (Learning Rate):** Determines how much newly acquired information overrides old knowledge.
* **$\gamma \max_{a} Q(S_{t+1}, a)$:** The discounted value of the best possible action in the subsequent state.
* **TD Error ($\delta_t$):** The difference between the target and the current estimate.

---

### 3. Exploration vs. Exploitation ($\epsilon$-Greedy)
To balance trying new actions vs. exploiting known good moves:
* With probability $\epsilon$, choose a completely random action (**Explore**).
* With probability $1 - \epsilon$, choose $\arg\max_a Q(s, a)$ (**Exploit**).
* $\epsilon$ decays after each episode:
  $$\epsilon \leftarrow \max(\epsilon_{\text{min}}, \epsilon \times \epsilon_{\text{decay}})$$

---

## 🚀 Streamlit Web Dashboard Features

| Tab | Description |
| :--- | :--- |
| **🎮 Interactive Game Visualizer** | Live 4x4 board rendering with emoji graphics. Step the agent move-by-move or simulate an entire game. |
| **📊 Training Performance** | Real-time charts showing rolling 100-episode success rates, cumulative rewards, and epsilon decay. |
| **🧭 Q-Table & Policy Map** | Arrow-annotated grid showing the optimal directional move per tile, alongside an interactive heatmap of all 16 states. |
| **📖 Theory & Math** | In-app mathematical formulation and algorithmic walkthrough. |

---

## 🛠️ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Syed-Muhammad-Qamran/frozen-lake.git
cd frozen-lake
```

### 2. Set Up Virtual Environment & Install Dependencies
#### Using standard `pip`:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

#### Or using `uv` (ultra-fast):
```bash
uv pip install -r requirements.txt
```

---

## 💻 Running the Project

### Launch the Streamlit App:
```bash
streamlit run app.py
```
Your default web browser will automatically open:
```
Local URL: http://localhost:8501
```

---

## 📊 Benchmark Results

| Mode | Training Episodes | Evaluation Success Rate (100 Games) | Behavior |
| :--- | :---: | :---: | :--- |
| **Deterministic (`is_slippery=False`)** | 1,000 | **100.0%** | Finds optimal 6-step path without errors. |
| **Stochastic (`is_slippery=True`)** | 5,000 | **~74.0%** | Reaches theoretical optimal policy under random slip dynamics. |

### Learned Optimal Policy Grid (Deterministic):
```
| ↓  →  ↓  ← |
| ↓  🕳️  ↓  🕳️ |
| →  ↓  ↓  🕳️ |
| 🕳️  →  →  🏆 |
```

---

## 📁 Repository Structure

```text
frozen-lake/
│
├── app.py               # Streamlit web application & interactive UI
├── q_learning.py        # Core Q-Learning Agent class & training engine
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore patterns
└── README.md            # Comprehensive project documentation
```

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

## 👤 Author
* **Syed Muhammad Qamran**
* GitHub: [@Syed-Muhammad-Qamran](https://github.com/Syed-Muhammad-Qamran)
