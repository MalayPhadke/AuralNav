import sys
sys.path.append("../src/")
from datetime import datetime
import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging 
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import room_types
import agent
import audio_room
import utils
import constants
import nussl_utils
from datasets import BufferData
import time
import audio_processing
from models import RnnAgent
import transforms

import warnings
warnings.filterwarnings("ignore")

"""
Multi-source experiment for AuralNav
This experiment allows configuring and testing with variable numbers of audio sources
"""

if "audio-room-v0" not in gym.envs.registry:
    gym.register(
        id="audio-room-v0",
        entry_point="src.audio_room.envs.audio_env:AudioEnv",
        max_episode_steps=1000  # Or your preferred max steps
    )

# Room configuration
# Shoebox Room - use a simple rectangular room
room = room_types.ShoeBox(x_length=10, y_length=10)  # Larger room for multi-source scenario

# Configure audio sources - add more source types for multi-source experiments
source_folders_dict = {
    '../sounds/phone/': 1,
    '../sounds/siren/': 1,
    '../sounds/car/': 1,
    '../sounds/samples/male': 1,
}

# Multi-source configuration
NUM_SOURCES = 4  # Try with 3, 4, or more sources
MAX_DETECTABLE = None  # Set to a specific number to allow partial success (e.g., 2)
                       # or None to require finding all sources

# Verify that each path in source_folders_dict points to valid wav files
import os
import glob

valid_audio_files = 0
missing_folders = []
missing_files = []

for folder_path, count in source_folders_dict.items():
    # Get absolute path if it's relative
    abs_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), folder_path))
    
    # Check if folder exists
    if not os.path.exists(abs_folder_path):
        missing_folders.append(folder_path)
        continue
        
    # Count wav files in the folder
    wav_files = glob.glob(os.path.join(abs_folder_path, "*.wav"))
    
    if len(wav_files) == 0:
        missing_files.append(folder_path)
    else:
        valid_audio_files += min(count, len(wav_files))  # Count only up to requested number

# Report any issues found
if missing_folders or missing_files or valid_audio_files < NUM_SOURCES:
    error_msg = ["Audio source validation failed:"]
    
    if missing_folders:
        error_msg.append(f"- Missing folders: {', '.join(missing_folders)}")
    
    if missing_files:
        error_msg.append(f"- No .wav files in folders: {', '.join(missing_files)}")
        
    if valid_audio_files < NUM_SOURCES:
        error_msg.append(f"- Not enough audio files: found {valid_audio_files}, need {NUM_SOURCES}")
        
    raise ValueError("\n".join(error_msg))

# Set up the gym environment
# Get room configuration
room_config = room.generate()

# Create the environment
env = gym.make(
    "audio-room-v0",
    room_config=room_config,
    source_folders_dict=source_folders_dict,
    corners=False,  # Using ShoeBox room, not a polygon
    max_order=10,
    step_size=0.5,
    acceptable_radius=1.0,
    absorption=1.0,
    visualize_pygame=True,
    play_audio_on_step=True,
    num_sources=NUM_SOURCES,
    max_detectable_sources=MAX_DETECTABLE
)
observation, info = env.reset(seed=0)

# Create buffer data folders
utils.create_buffer_data_folders()

# Transform configuration
tfm = transforms.Compose([
    transforms.GetAudio(mix_key=['prev_state', 'new_state']),
    transforms.ToSeparationModel(),
    transforms.GetExcerpt(excerpt_length=32000, tf_keys=['mix_audio_prev_state'], time_dim=1),
    transforms.GetExcerpt(excerpt_length=32000, tf_keys=['mix_audio_new_state'], time_dim=1)
])

# Create dataset object
dataset = BufferData(
    folder=constants.DIR_DATASET_ITEMS, 
    to_disk=True, 
    transform=tfm
)

# Configure experiment name and tensorboard writer
exp_name = f'multi-source-{NUM_SOURCES}'
exp_id = '{}_{}'.format(exp_name, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
writer = SummaryWriter('runs/{}'.format(exp_id))

# Environment configuration
env_config = {
    'env': env, 
    'dataset': dataset, 
    'episodes': 200,  # More episodes for learning with multiple sources 
    'max_steps': 1500,  # More steps per episode for finding multiple sources
    'stable_update_freq': 150,
    'save_freq': 10, 
    'writer': writer,
    'dense': True,
    'decay_rate': 0.0001,  # Slower decay for more exploration with multiple sources
    'decay_per_ep': True,
    'num_sources': NUM_SOURCES,
    'sample_rate': constants.RESAMPLE_RATE
}

# Store num_sources separately to use when creating the agent
num_sources_for_agent = NUM_SOURCES

# Set up model save path
save_path = os.path.join(constants.MODEL_SAVE_PATH, exp_name)
dataset_config = {
    'batch_size': 16,  # Larger batch size for better gradient estimates
    'num_updates': 3,  # More updates per step 
    'save_path': save_path
}

# Clear save folder for clean experiment
utils.clear_models_folder(save_path)

# RNN configuration
rnn_config = {
    'bidirectional': True,
    'dropout': 0.3,
    'filter_length': 256,
    'hidden_size': 64,  # Increased hidden size for more capacity
    'hop_length': 64,
    'mask_activation': ['softmax'],
    'mask_complex': False,
    'mix_key': 'mix_audio',
    'normalization_class': 'BatchNorm',
    'num_audio_channels': 1,
    'num_filters': 256,
    'num_layers': 2,  # Increased to 2 layers for more capacity
    'num_sources': NUM_SOURCES,
    'rnn_type': 'lstm',
    'window_type': 'sqrt_hann',
}

# STFT configuration
stft_config = {
    'hop_length': 64,
    'num_filters': 256,
    'direction': 'transform',
    'window_type': 'sqrt_hann'
}

# Initialize the agent
rnn_agent = RnnAgent(
    env_config=env_config, 
    dataset_config=dataset_config,
    rnn_config=rnn_config,
    stft_config=stft_config,
    learning_rate=0.0005,  # Lower learning rate for stability
    pretrained=False,
    num_sources=num_sources_for_agent
)

# Run the training
print(f"\nStarting training with {NUM_SOURCES} sources...")
print(f"Max detectable sources: {MAX_DETECTABLE if MAX_DETECTABLE else 'All'}")
torch.autograd.set_detect_anomaly(True)
rnn_agent.fit(visualize=True)

# Create directory for plots
plot_dir = '../models/evaluation_data/'
os.makedirs(plot_dir, exist_ok=True)

# Generate performance plots
print("\nGenerating performance plots...")

# Plot of mean reward per episode
plt.figure(figsize=(10, 6))
plt.plot(rnn_agent.mean_episode_reward, 'b-')
plt.title(f'Mean Reward per Episode ({NUM_SOURCES} Sources)')
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.grid(True, alpha=0.3)
plt.savefig(f'{plot_dir}/mean_reward_{exp_name}.png', dpi=300)
plt.close()
print(f"Mean reward plot saved to: {plot_dir}/mean_reward_{exp_name}.png")

# Plot of cumulative reward
plt.figure(figsize=(10, 6))
plt.plot(rnn_agent.cumulative_reward, 'r-')
plt.title(f'Cumulative Reward ({NUM_SOURCES} Sources)')
plt.xlabel('Training Step')
plt.ylabel('Cumulative Reward')
plt.grid(True, alpha=0.3)
plt.savefig(f'{plot_dir}/cumulative_reward_{exp_name}.png', dpi=300)
plt.close()
print(f"Cumulative reward plot saved to: {plot_dir}/cumulative_reward_{exp_name}.png")

# Plot of steps per episode
finished_steps = []
for i, episode in enumerate(rnn_agent.episode_summary):
    if 'finished_at_step' in episode:
        finished_steps.append(episode['finished_at_step'])
    elif hasattr(rnn_agent, 'episode_steps') and i < len(rnn_agent.episode_steps):
        finished_steps.append(rnn_agent.episode_steps[i])

if finished_steps:
    plt.figure(figsize=(10, 6))
    plt.plot(finished_steps, 'g-')
    plt.title(f'Steps to Complete Episode ({NUM_SOURCES} Sources)')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{plot_dir}/steps_per_episode_{exp_name}.png', dpi=300)
    plt.close()
    print(f"Steps per episode plot saved to: {plot_dir}/steps_per_episode_{exp_name}.png")

# Plot sources detected per episode
if hasattr(rnn_agent, 'episode_summary'):
    sources_detected = []
    for episode in rnn_agent.episode_summary:
        if 'sources_found' in episode:
            sources_detected.append(episode['sources_found'])
        else:
            sources_detected.append(0)  # Default if not found
            
    if sources_detected:
        plt.figure(figsize=(10, 6))
        plt.plot(sources_detected, 'purple')
        plt.title(f'Sources Detected per Episode (Max: {NUM_SOURCES})')
        plt.xlabel('Episode')
        plt.ylabel('Sources Detected')
        plt.yticks(np.arange(0, NUM_SOURCES + 1, 1))
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{plot_dir}/sources_detected_{exp_name}.png', dpi=300)
        plt.close()
        print(f"Sources detected plot saved to: {plot_dir}/sources_detected_{exp_name}.png")

print("\nMulti-source training and visualization complete!")
