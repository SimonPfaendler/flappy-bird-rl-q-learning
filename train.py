import gymnasium
import flappy_bird_gymnasium
import numpy as np
import random
import utils

env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)

q_table = np.full(list(utils.STATE_DIMS) + [env.action_space.n], 0.1)

episodes = 60000
epsilon = 1.0          
epsilon_min = 0.0
epsilon_decay = 0.9995 
alpha = 0.1
gamma = 0.99   
lam = 0.8

eligibility_trace = np.zeros_like(q_table)

scores_history = []

print("Training")

for episode in range(episodes):
    state, _ = env.reset()
    state_disc = utils.get_discrete_state(state)
    
    eligibility_trace.fill(0)
    
    # Epsilon Greedy
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state_disc])

    terminated = False
    step_count = 0
    
    while not terminated:
        # --- REWARD SYSTEM ---
        next_state, reward_env, terminated, truncated, info = env.step(action)
        
        # Frame Skipping: If we flapped (action 1), coast for a few frames
        if action == 1 and not terminated:
            frames_to_skip = 2 
            for _ in range(frames_to_skip):
                if terminated: break
                # Skip frame with action 0 (do nothing)
                next_state_s, reward_s, terminated_s, truncated_s, info_s = env.step(0)
                reward_env += reward_s
                next_state = next_state_s # Update next_state to the latest
                terminated = terminated_s or terminated
                truncated = truncated_s or truncated
                
                # Decay traces for the skipped step
                eligibility_trace *= (gamma * lam)
                eligibility_trace[eligibility_trace < 0.01] = 0

        next_state_disc = utils.get_discrete_state(next_state)

        if terminated:
            reward = -1000 
        elif reward_env > 0.9: # This might need adjustment if we sum rewards
            reward = 500  
        else:
            reward = 0.1
        
        if random.random() < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(q_table[next_state_disc])

        # --- SARSA(Lambda) UPDATE ---
        
        # 1. TD-Error
        current_q = q_table[state_disc + (action,)]
        if terminated:
            target = reward
        else:
            target = reward + gamma * q_table[next_state_disc + (next_action,)]
        
        delta = target - current_q

        # 2. Eligibility Trace Update (REPLACING TRACES)
        eligibility_trace[state_disc + (action,)] = 1.0

        # 3. Q-Table Update
        q_table += alpha * delta * eligibility_trace

        # 4. Trace Decay
        eligibility_trace *= (gamma * lam)
        
        # Performance
        if step_count % 10 == 0:
             eligibility_trace[eligibility_trace < 0.01] = 0

        state_disc = next_state_disc
        action = next_action
        step_count += 1
    
    scores_history.append(step_count)
    if len(scores_history) > 100:
        scores_history.pop(0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if alpha > 0.01:
        alpha *= 0.99995

    if episode % 1000 == 0:
        avg_score = np.mean(scores_history)
        print(f"Episode: {episode}, Score: {step_count}, Avg Score: {avg_score:.2f}, Epsilon: {epsilon:.4f}, Alpha: {alpha:.2f}")

np.save("brain.npy", q_table)
print("Training finish")
env.close()