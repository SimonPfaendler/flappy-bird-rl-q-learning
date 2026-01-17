import gymnasium
import flappy_bird_gymnasium
import numpy as np
import time


horiz_bins = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5])
vert_bins = np.array([-0.4, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.4])
vel_bins = np.array([-3.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0])

def get_discrete_state(state):
    horiz_dist = state[3]
    vert_dist = state[5] - state[9] 
    velocity = state[10]
    
    x = np.digitize(horiz_dist, horiz_bins)
    y = np.digitize(vert_dist, vert_bins)
    v = np.digitize(velocity, vel_bins)
    return (x, y, v)


try:
    q_table = np.load("brain.npy")
    print("Q-Table created")
except:
    print("First Run train.py")
    exit()


env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

state, _ = env.reset()
state_disc = get_discrete_state(state)

while True:
    
    action = np.argmax(q_table[state_disc])

    state, reward, terminated, truncated, info = env.step(action)
    state_disc = get_discrete_state(state)
    
    time.sleep(0.03)

    if terminated:
        state, _ = env.reset()
        state_disc = get_discrete_state(state)