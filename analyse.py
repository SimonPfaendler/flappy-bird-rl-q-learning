import gymnasium
import flappy_bird_gymnasium
import time
import numpy as np
import pygame


env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

feature_names = [
    "0. Last Pipe Horizontal ",
    "1. Last Top Pipe Y      ",
    "2. Last Bottom Pipe Y   ",
    "3. Next Pipe Horizontal ",
    "4. Next Top Pipe Y      ",
    "5. Next Bottom Pipe Y   ",
    "6. Next Next Pipe Horiz ",
    "7. Next Next Top Y      ",
    "8. Next Next Bottom Y   ",
    "9. Player Y Position    ",
    "10. Player Velocity     ", 
    "11. Player Rotation     "
]

state, _ = env.reset()

print("--- START ANALYSE ---")
print("Press SPACE to flap!")

while True:
    # Handle PyGame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()

    keys = pygame.key.get_pressed()
    
    action = 0 
    if keys[pygame.K_SPACE]:
        action = 1
        
    state, reward, terminated, truncated, info = env.step(action)
    

    
    print("\n------------------------------------------------")
    print(f"Aktueller State (Reward: {reward})")
    print("------------------------------------------------")
    
    for i in range(12):
        val = state[i]
        name = feature_names[i]
        
        # Markiere die wichtigen Werte fett oder mit Pfeil
        marker = ""
        if i in [3, 5, 9, 10]:
            marker = "  <-- RELEVANT FÜR SARSA"
            
        print(f"{name}: {val:.4f}{marker}")


    time.sleep(0.05)

    if terminated:
        print("\n--- GESTORBEN ---")
        print("Press SPACE to restart")
        while True:
            # Handle PyGame events to allow closing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    exit()
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                break
            
            time.sleep(0.1)

        state, _ = env.reset()

