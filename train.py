import gymnasium
import flappy_bird_gymnasium
import numpy as np
import random
import utils
import pickle
import os
from collections import defaultdict

# Create Environment
env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)

# Hyperparameters
episodes = 20000
alpha = 0.1         # Learning rate
gamma = 0.98        # Discount factor
epsilon = 0.0       # Greedy policy (exploration via optimistic init)
init_q = 0.1        # Optimistic Initialization
lam = 0.7           # Lambda for eligibility traces
trace_min = 0.01    # Threshold to prune traces

# Q-Table as defaultdict
# Returns a list of [Q(s,0), Q(s,1)] initialized to init_q
q_table = defaultdict(lambda: [init_q, init_q])

# Training Metrics
scores_history = []
best_score = 0

print(f"Starting Training: Q-Learning with Traces (Alpha={alpha}, Gamma={gamma}, InitQ={init_q}, Lambda={lam})")

try:
    for episode in range(episodes):
        state, _ = env.reset()
        current_state_key = utils.get_discrete_state(state)
        
        # Eligibility Traces: Map state -> [trace_action_0, trace_action_1]
        traces = defaultdict(lambda: [0.0, 0.0])
        
        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0
        
        while not (terminated or truncated):
            # Select Action: Greedy (due to epsilon=0.0)
            # Break ties randomly
            q_values = q_table[current_state_key]
            if q_values[0] == q_values[1]:
                action = random.choice([0, 1])
            else:
                action = np.argmax(q_values)
                
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Custom Reward Logic
            if terminated:
                reward = -1000
            elif reward >= 1.0: # Passed a pipe
                reward = 5.0
            else:
                reward = 0.5 # Survival reward per frame
                
            total_reward += reward
            
            # Discretize Next State
            next_state_key = utils.get_discrete_state(next_state)
            
            # --- Q-Learning with Eligibility Traces ---
            
            # 1. Calculate TD Error
            # delta = R + gamma * max_a(Q(s', a)) - Q(s, a)
            current_q = q_table[current_state_key][action]
            max_next_q = np.max(q_table[next_state_key])
            delta = reward + gamma * max_next_q - current_q
            
            # 2. Update Eligibility Trace for current state (Replacing Traces)
            traces[current_state_key][action] = 1.0
            
            # 3. Update Q-values and Decay Traces for ALL active states
            # To optimize, we iterate over the copy of keys or items to allow modification/deletion
            # OR we create a list of keys to remove
            keys_to_remove = []
            
            for state_key, trace_values in traces.items():
                # We update both actions for the state if they have traces
                for a in range(2):
                    if trace_values[a] > trace_min:
                        # Update Q-Value
                        q_table[state_key][a] += alpha * delta * trace_values[a]
                        
                        # Decay Trace
                        traces[state_key][a] *= (gamma * lam)
                    else:
                        # Just ensure it's zero if it fell below threshold logic previously (or was 0)
                        traces[state_key][a] = 0.0
                
                # If both traces are practically zero, mark for removal to keep dict small
                if traces[state_key][0] <= trace_min and traces[state_key][1] <= trace_min:
                     keys_to_remove.append(state_key)
                     
            # Cleanup
            for key in keys_to_remove:
                del traces[key]
            
            # Move to next state
            current_state_key = next_state_key
            step_count += 1
            
        scores_history.append(step_count)
        if step_count > best_score:
            best_score = step_count
        
        # Logging
        if episode % 1000 == 0:
            avg_score = np.mean(scores_history[-100:]) if scores_history else 0
            print(f"Episode: {episode}, Score: {step_count}, Best Score: {best_score}, Avg Score: {avg_score:.2f}, Q-Table Size: {len(q_table)}")
            
except KeyboardInterrupt:
    print("Training Interrupted by User")

# Final Save
with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(q_table), f)
print("Training Completed & Saved to q_table.pkl")
env.close()
