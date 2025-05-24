"""
Script to run the AuralNav environment with visualization enabled.
"""
import sys
sys.path.append("../src/")

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

import room_types
import audio_room
import constants
from visualization import EnvironmentVisualizer

# Register the environment if not already registered
if "audio-room-v0" not in gym.envs.registry:
    gym.register(
        id="audio-room-v0",
        entry_point="src.audio_room.envs.audio_env:AudioEnv",
        max_episode_steps=1000
    )

# Create a Shoebox Room
room = room_types.ShoeBox(x_length=8, y_length=8)

# Set up source folders
source_folders_dict = {
    '../sounds/phone/': 1,
    '../sounds/siren/': 1
}

# Create the environment
env = gym.make(
    "audio-room-v0",
    room_config=room.generate(),
    source_folders_dict=source_folders_dict,
    corners=room.corners,
    max_order=10,
    step_size=.5,
    acceptable_radius=1.0,
    absorption=1.0,
    play_audio_on_step=False,
    show_room_on_step=False,
)

# Initialize the visualizer
visualizer = EnvironmentVisualizer()

# Run a few episodes with visualization
num_episodes = 3
steps_per_episode = 200

for episode in range(num_episodes):
    print(f"\nEpisode {episode+1}/{num_episodes}")
    
    # Reset the environment
    observation, info = env.reset()
    
    # Setup visualization for this episode
    if hasattr(env.unwrapped, 'corners') and env.unwrapped.corners:
        room_bounds = [env.unwrapped.room_config[0], env.unwrapped.room_config[1]]
    else:
        room_bounds = [env.unwrapped.x_min, env.unwrapped.y_min, 
                       env.unwrapped.x_max, env.unwrapped.y_max]
    
    source_locs = env.unwrapped.source_locs
    
    # Reset the visualizer with new room configuration
    visualizer.reset(room_boundaries=room_bounds, source_locs=source_locs)
    
    # Add initial agent position
    if hasattr(env.unwrapped, 'agent_loc'):
        visualizer.update(env.unwrapped.agent_loc, env.unwrapped.cur_angle)
    
    # Run one episode
    episode_reward = 0
    for step in range(steps_per_episode):
        # Take a random action
        action = env.action_space.sample()
        
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # Update visualization with new agent position
        if hasattr(env.unwrapped, 'agent_loc'):
            visualizer.update(env.unwrapped.agent_loc, env.unwrapped.cur_angle)
        
        # Print step information
        if step % 20 == 0:
            print(f"Step {step}, Reward: {reward:.2f}, Cumulative: {episode_reward:.2f}")
        
        # Break if episode is done
        if terminated or truncated:
            print(f"Episode finished after {step+1} steps with reward {episode_reward:.2f}")
            break
    
    # End episode in visualizer and generate visualization
    visualizer.end_episode()
    
    # For the last episode, also show an animation
    if episode == num_episodes - 1:
        visualizer.animate_episode(interval=50, save=True)

# At the end, visualize all episodes together
visualizer.visualize_all_episodes(save=True)

print("\nVisualization complete! Check the '../visualizations' directory for saved images and animations.")
