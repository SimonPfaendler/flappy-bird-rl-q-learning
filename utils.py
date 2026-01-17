import numpy as np

horiz_bins = np.array([-0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
vert_bins = np.array([
    -0.5, -0.3, -0.2, 
    -0.15, -0.1, -0.075, -0.05, -0.025,  # Sehr fein UNTER der Mitte (Vogel fällt oft hier)
    0.0, 
    0.025, 0.05, 0.075, 0.1, 0.15,      # Sehr fein ÜBER der Mitte
    0.2, 0.3, 0.5
])
vel_bins = np.array([-9.0, -7.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0])

STATE_DIMS = (len(horiz_bins)+1, len(vert_bins)+1, len(vel_bins)+1)


def get_discrete_state(state):
    horiz_dist = state[3]
    target_y = (state[5] + 0.1) - 0.09 
    vert_dist = target_y - state[9] 
    velocity = state[10]
    
    x = np.digitize(horiz_dist, horiz_bins)
    y = np.digitize(vert_dist, vert_bins)
    v = np.digitize(velocity, vel_bins)
    
    return (x, y, v)
