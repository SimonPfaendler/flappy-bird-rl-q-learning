import gymnasium
import flappy_bird_gymnasium
import numpy as np
import time


horiz_bins = np.array([-0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
vert_bins = np.array([
    -0.5, -0.3, -0.2, 
    -0.15, -0.1, -0.075, -0.05, -0.025,  # Sehr fein UNTER der Mitte (Vogel fällt oft hier)
    0.0, 
    0.025, 0.05, 0.075, 0.1, 0.15,      # Sehr fein ÜBER der Mitte
    0.2, 0.3, 0.5
])
vel_bins = np.array([-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.5, 3.0])

def get_discrete_state(state):
    horiz_dist = state[3]
    gap_center_y = (state[5] + 0.1) - 0.04 
    
    vert_dist = gap_center_y - state[9] 
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