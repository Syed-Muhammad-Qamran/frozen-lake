import time
import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd
from q_learning import QLearningAgent

st.set_page_config(
    page_title="FrozenLake RL - Q-Learning Agent",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #334155;
        text-align: center;
    }
    .grid-tile {
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        border: 1px solid #475569;
        margin: 2px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# FrozenLake 4x4 Map definitions
MAP_TILES = [
    'S', 'F', 'F', 'F',
    'F', 'H', 'F', 'H',
    'F', 'F', 'F', 'H',
    'H', 'F', 'F', 'G'
]
ACTION_NAMES = {0: "Left (←)", 1: "Down (↓)", 2: "Right (→)", 3: "Up (↑)"}
ACTION_ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}

# Initialize session state variables
if "agent" not in st.session_state:
    st.session_state.agent = None
if "training_stats" not in st.session_state:
    st.session_state.training_stats = None
if "eval_stats" not in st.session_state:
    st.session_state.eval_stats = None
if "current_sim_state" not in st.session_state:
    st.session_state.current_sim_state = 0
if "sim_history" not in st.session_state:
    st.session_state.sim_history = [0]
if "sim_status" not in st.session_state:
    st.session_state.sim_status = "Ready to play"

# ----------------- SIDEBAR -----------------
st.sidebar.title("❄️ Agent Configuration")
st.sidebar.markdown("Configure and train your **Q-Learning** agent on **FrozenLake-v1**.")

st.sidebar.subheader("1. Environment Settings")
is_slippery = st.sidebar.checkbox(
    "Slippery Ice (Stochastic)",
    value=False,
    help="When unchecked (deterministic), actions succeed 100% of the time. When checked, the agent slips with 66% probability."
)

st.sidebar.subheader("2. Hyperparameters")
episodes = st.sidebar.slider("Training Episodes", min_value=100, max_value=5000, value=1000, step=100)
alpha = st.sidebar.slider("Learning Rate (α)", min_value=0.01, max_value=1.0, value=0.8 if not is_slippery else 0.1, step=0.05)
gamma = st.sidebar.slider("Discount Factor (γ)", min_value=0.50, max_value=0.99, value=0.95, step=0.01)
epsilon = st.sidebar.slider("Initial Exploration (ε)", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
epsilon_decay = st.sidebar.slider("Epsilon Decay Rate", min_value=0.990, max_value=0.9999, value=0.999, step=0.0005, format="%.4f")

if st.sidebar.button("🚀 Train Agent", type="primary", use_container_width=True):
    with st.spinner("Training Q-Learning agent..."):
        env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
        agent = QLearningAgent(
            n_states=16,
            n_actions=4,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=0.01,
            epsilon_decay=epsilon_decay
        )
        
        progress_bar = st.sidebar.progress(0.0)
        def on_progress(current, total, success, eps):
            progress_bar.progress(current / total)

        stats = agent.train(env, num_episodes=episodes, callback=on_progress)
        eval_results = agent.evaluate(env, num_episodes=100)
        env.close()

        st.session_state.agent = agent
        st.session_state.training_stats = stats
        st.session_state.eval_stats = eval_results
        st.session_state.current_sim_state = 0
        st.session_state.sim_history = [0]
        st.session_state.sim_status = "Training complete! Ready to simulate."
        st.sidebar.success(f"Trained in {episodes} episodes!")

if st.sidebar.button("🔄 Reset Agent", use_container_width=True):
    st.session_state.agent = None
    st.session_state.training_stats = None
    st.session_state.eval_stats = None
    st.session_state.current_sim_state = 0
    st.session_state.sim_history = [0]
    st.session_state.sim_status = "Reset."
    st.rerun()

# ----------------- MAIN PAGE -----------------
st.title("🧊 FrozenLake Q-Learning Dashboard")
st.markdown(
    "Explore how an autonomous Reinforcement Learning agent masters the classical **FrozenLake** puzzle from scratch through trial and error."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎮 Interactive Game Visualizer",
    "📊 Training Performance",
    "🧭 Q-Table & Learned Policy",
    "📖 RL Theory & Math"
])

# ----------------- TAB 1: GAME VISUALIZER -----------------
with tab1:
    st.subheader("Interactive 4x4 Lake Board")
    
    col_board, col_ctrl = st.columns([1.2, 1])
    
    with col_board:
        # Render 4x4 Grid
        for row in range(4):
            cols = st.columns(4)
            for col in range(4):
                state_idx = row * 4 + col
                tile_type = MAP_TILES[state_idx]
                is_agent = (state_idx == st.session_state.current_sim_state)
                
                # Determine display label
                if is_agent:
                    label = "🤖 Agent"
                    bg_color = "#38bdf8" # bright sky blue
                    text_color = "#0f172a"
                elif tile_type == 'S':
                    label = "🟦 Start"
                    bg_color = "#1e3a8a"
                    text_color = "#93c5fd"
                elif tile_type == 'H':
                    label = "🕳️ Hole"
                    bg_color = "#450a0a"
                    text_color = "#f87171"
                elif tile_type == 'G':
                    label = "🏆 Goal"
                    bg_color = "#78350f"
                    text_color = "#fde047"
                else:
                    label = "❄️ Ice"
                    bg_color = "#0f172a"
                    text_color = "#cbd5e1"
                    
                cols[col].markdown(
                    f"""<div class="grid-tile" style="background-color: {bg_color}; color: {text_color};">
                        {label}<br><span style="font-size: 0.75rem; opacity: 0.8;">Tile {state_idx}</span>
                    </div>""",
                    unsafe_allow_html=True
                )
        
        st.markdown(f"**Game Status:** `{st.session_state.sim_status}`")

    with col_ctrl:
        st.markdown("### Simulation Controls")
        if st.session_state.agent is None:
            st.info("👈 Please train the agent first using the sidebar button **'🚀 Train Agent'**.")
        else:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("▶️ Step Agent", use_container_width=True):
                    curr = st.session_state.current_sim_state
                    if MAP_TILES[curr] in ['H', 'G']:
                        st.session_state.current_sim_state = 0
                        st.session_state.sim_history = [0]
                        st.session_state.sim_status = "Restarted from start."
                    else:
                        action = st.session_state.agent.get_action(curr, greedy=True)
                        # Step environment
                        env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
                        env.reset()
                        # advance to current state
                        env.unwrapped.s = curr
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        env.close()
                        
                        st.session_state.current_sim_state = next_state
                        st.session_state.sim_history.append(next_state)
                        
                        if terminated:
                            if reward == 1.0:
                                st.session_state.sim_status = "🎉 SUCCESS: Reached the Goal!"
                            else:
                                st.session_state.sim_status = "💥 FAILED: Fell into a Hole!"
                        else:
                            st.session_state.sim_status = f"Moved {ACTION_NAMES[action]} to Tile {next_state}"
                    st.rerun()

            with col_b2:
                if st.button("🔄 Reset Position", use_container_width=True):
                    st.session_state.current_sim_state = 0
                    st.session_state.sim_history = [0]
                    st.session_state.sim_status = "Reset to start."
                    st.rerun()
            
            st.markdown(f"**Current State:** `Tile {st.session_state.current_sim_state} ({MAP_TILES[st.session_state.current_sim_state]})`")
            st.markdown(f"**Path History:** `{' -> '.join(map(str, st.session_state.sim_history))}`")
            
            if st.session_state.agent:
                best_act = st.session_state.agent.get_action(st.session_state.current_sim_state, greedy=True)
                st.markdown(f"**Agent's Preferred Move:** `{ACTION_NAMES[best_act]}`")

# ----------------- TAB 2: TRAINING PERFORMANCE -----------------
with tab2:
    st.subheader("Training Analytics & Evaluation Metrics")
    if st.session_state.training_stats is None:
        st.info("Train the agent to view learning curves and performance metrics.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Episodes Trained", episodes)
        m2.metric("Test Success Rate (100 games)", f"{st.session_state.eval_stats['success_rate']:.1f}%")
        m3.metric("Avg Steps to Goal", f"{st.session_state.eval_stats['avg_steps']:.1f}")
        m4.metric("Final Epsilon (ε)", f"{st.session_state.agent.epsilon:.4f}")

        st.markdown("#### 100-Episode Rolling Success Rate (%)")
        df_chart = pd.DataFrame({
            "Success Rate (%)": st.session_state.training_stats["success_rates"],
            "Epsilon Exploration": st.session_state.training_stats["epsilons"]
        })
        st.line_chart(df_chart["Success Rate (%)"])
        
        st.markdown("#### Epsilon Decay Curve")
        st.line_chart(df_chart["Epsilon Exploration"])

# ----------------- TAB 3: Q-TABLE & POLICY -----------------
with tab3:
    st.subheader("Learned Optimal Policy & Q-Table Values")
    if st.session_state.agent is None:
        st.info("Train the agent to inspect the learned Q-table.")
    else:
        col_p1, col_p2 = st.columns([1, 1.5])
        
        with col_p1:
            st.markdown("#### Optimal Policy Direction Map")
            arrows = st.session_state.agent.get_policy_arrows()
            for r in range(4):
                cols = st.columns(4)
                for c in range(4):
                    tile_idx = r * 4 + c
                    if MAP_TILES[tile_idx] == 'H':
                        disp = "🕳️"
                    elif MAP_TILES[tile_idx] == 'G':
                        disp = "🏆"
                    else:
                        disp = arrows[r, c]
                    cols[c].button(f"{disp}\n({tile_idx})", key=f"pol_{tile_idx}", use_container_width=True)
                    
        with col_p2:
            st.markdown("#### Full Q-Table Matrix (16 States × 4 Actions)")
            df_q = pd.DataFrame(
                st.session_state.agent.Q,
                columns=["Left (←)", "Down (↓)", "Right (→)", "Up (↑)"],
                index=[f"Tile {i} ({MAP_TILES[i]})" for i in range(16)]
            )
            st.dataframe(
                df_q.style.highlight_max(axis=1, color="#0284c7"),
                use_container_width=True,
                height=380
            )

# ----------------- TAB 4: RL THEORY & MATH -----------------
with tab4:
    st.subheader("Reinforcement Learning Foundations")
    st.markdown("""
    ### 1. Markov Decision Process (MDP)
    FrozenLake is formulated as a discrete MDP defined by $(S, A, P, R, \gamma)$:
    * **States ($S$):** 16 positions on a $4 \\times 4$ grid ($0$ to $15$).
    * **Actions ($A$):** 4 possible moves: Left ($0$), Down ($1$), Right ($2$), Up ($3$).
    * **Rewards ($R$):** $+1$ for reaching the Goal (`G`), $0$ otherwise.
    * **Discount Factor ($\gamma$):** Weighs immediate vs. future rewards ($\gamma \\in [0, 1)$).

    ### 2. The Q-Learning Update Equation
    Q-Learning is a **Model-Free, Off-Policy Temporal Difference (TD)** control algorithm. It updates state-action values using:
    """)
    st.latex(r"Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \Big]")
    st.markdown("""
    * **$\alpha$ (Learning Rate):** Controls how fast new information overrides old estimates.
    * **$R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$:** The **TD Target** (observed reward plus discounted optimal future value).
    * **$\text{TD Target} - Q(S_t, A_t)$:** The **TD Error** (the surprise factor).

    ### 3. Exploration vs. Exploitation ($\epsilon$-Greedy)
    * With probability $\epsilon$, the agent chooses a random action (**Exploration**).
    * With probability $1 - \epsilon$, the agent picks $\\arg\\max_a Q(s, a)$ (**Exploitation**).
    * Over time, $\epsilon$ decays so the agent transitions from random exploration to exploiting its learned policy.
    """)
