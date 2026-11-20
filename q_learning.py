import gymnasium as gym
import numpy as np

class QLearningAgent:
    """
    Q-Learning Agent for Discrete State and Action Spaces.
    Implements model-free, off-policy Temporal Difference (TD) control.
    """
    def __init__(
        self,
        n_states=16,
        n_actions=4,
        alpha=0.8,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-Table with zeros
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)
        
    def get_action(self, state, greedy=False):
        """
        Select an action using epsilon-greedy policy.
        If greedy=True, selects best action with random tie-breaking.
        """
        if not greedy and np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.n_actions)
        
        max_val = np.max(self.Q[state, :])
        best_actions = np.where(self.Q[state, :] == max_val)[0]
        return int(np.random.choice(best_actions))
    
    def update(self, state, action, reward, next_state):
        """
        Perform the Q-learning update step:
        Q(s, a) <- Q(s, a) + alpha * [R + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        best_next_q = np.max(self.Q[next_state, :])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error
    
    def decay_epsilon(self):
        """Decay exploration rate over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def train(self, env, num_episodes=1000, max_steps=100, callback=None):
        """
        Train the Q-learning agent on the given Gymnasium environment.
        """
        rewards_history = []
        rolling_success_rates = []
        epsilons = []

        for ep in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            for _ in range(max_steps):
                action = self.get_action(state, greedy=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                
                if terminated or truncated:
                    break
                    
            self.decay_epsilon()
            rewards_history.append(total_reward)
            epsilons.append(self.epsilon)
            
            # 100-episode rolling success rate
            recent_success = np.mean(rewards_history[-min(100, len(rewards_history)):]) * 100
            rolling_success_rates.append(recent_success)
            
            if callback and (ep % max(1, num_episodes // 20) == 0 or ep == num_episodes - 1):
                callback(ep + 1, num_episodes, recent_success, self.epsilon)
                
        return {
            "rewards": rewards_history,
            "success_rates": rolling_success_rates,
            "epsilons": epsilons
        }

    def evaluate(self, env, num_episodes=100, max_steps=100):
        """
        Evaluates the agent using purely greedy policy (no exploration).
        """
        successes = 0
        total_steps = 0
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            steps = 0
            for _ in range(max_steps):
                steps += 1
                action = self.get_action(state, greedy=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = next_state
                if terminated or truncated:
                    if reward == 1.0:
                        successes += 1
                    break
            total_steps += steps
            
        return {
            "success_rate": (successes / num_episodes) * 100.0,
            "avg_steps": total_steps / num_episodes
        }

    def get_policy_matrix(self):
        """
        Extracts the greedy policy for all states: argmax_a Q(s, a).
        """
        return np.argmax(self.Q, axis=1)

    def get_policy_arrows(self):
        """
        Returns a 4x4 array of arrow characters representing the optimal policy.
        """
        symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        policy = self.get_policy_matrix()
        return np.array([symbols[a] for a in policy]).reshape((4, 4))
