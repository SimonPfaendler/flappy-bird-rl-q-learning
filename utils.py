import math

def get_discrete_state(state):
    """
    Discretizes the state using the FlapAI Bird paper methodology:
    discretized = rounding * math.floor(value / rounding)
    
    State mapping from FlappyBird-v0:
    0: last_pipe_x
    1: last_pipe_y
    2: last_pipe_y_bottom
    3: next_pipe_x (dist_x)
    4: next_pipe_y_top
    5: next_pipe_y_bottom
    6: next_next_pipe_x
    7: next_next_pipe_y_top
    8: next_next_pipe_y_bottom
    9: player_y
    10: player_vel
    11: player_rot
    """
    
    # Extract raw features
    dist_x = state[3]
    
    dist_y = state[9] - state[5]
    
    velocity = state[10]
    
    # Discretization parameters
    round_x = 0.15
    round_y = 0.1
    round_v = 1.0
    
    # Apply formula
    d_x = round_x * math.floor(dist_x / round_x)
    d_y = round_y * math.floor(dist_y / round_y)
    d_v = round_v * math.floor(velocity / round_v)
    
    # Return as tuple for dictionary key
    # Rounding to avoids floating point weirdness in keys (optional but good practice)
    return (round(d_x, 2), round(d_y, 2), round(d_v, 2))
