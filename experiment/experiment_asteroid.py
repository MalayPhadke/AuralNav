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
from models_asteroid import RnnAgent
import transforms

import warnings
warnings.filterwarnings("ignore")

"""
One of our main experiments for OtoWorld introductory paper 
"""

if "audio-room-v0" not in gym.envs.registry:
    gym.register(
        id="audio-room-v0",
        entry_point="src.audio_room.envs.audio_env:AudioEnv",
        max_episode_steps=1000  # Or your preferred max steps
    )


# Shoebox Room
room = room_types.ShoeBox(x_length=8, y_length=8)

# Uncomment for Polygon Room
#room = room_types.Polygon(n=6, r=2, x_center=5, y_center=5)

# Configure audio sources
# For multi-source experiment, add more source types or increase count
# Format: {folder_path: number_of_files_to_use}
source_folders_dict = {
    '../sounds/phone/': 1,
    '../sounds/siren/': 1,
    '../sounds/car/': 1,
    '../sounds/samples/male/': 1,
}

# Configure number of sources for the experiment
NUM_SOURCES = 4  # Change this to 3, 4, etc. for multi-source experiments
MAX_DETECTABLE = None  # Set to a number less than NUM_SOURCES to allow partial success

# Set up the gym environment
env = gym.make(
    "audio-room-v0",
    room_config=room.generate(),
    source_folders_dict=source_folders_dict,
    corners=room.corners,
    max_order=10,
    step_size=.5,
    acceptable_radius=1.0,
    absorption=1.0,
    visualize_pygame=True,
    play_audio_on_step=False,
    num_sources=NUM_SOURCES,
    max_detectable_sources=MAX_DETECTABLE
)
observation, info = env.reset(seed=0)

# create buffer data folders
utils.create_buffer_data_folders()

# fixing lengths
tfm = transforms.Compose([
    transforms.GetAudio(mix_key=['prev_state', 'new_state']),
    transforms.ToSeparationModel(),
    transforms.GetExcerpt(excerpt_length=32000,
                            tf_keys=['mix_audio_prev_state'], time_dim=1),
    transforms.GetExcerpt(excerpt_length=32000,
                            tf_keys=['mix_audio_new_state'], time_dim=1)
])

# create dataset object (subclass of nussl.datasets.BaseDataset)
dataset = BufferData(
    folder=constants.DIR_DATASET_ITEMS, 
    to_disk=True, 
    transform=tfm
)

# define tensorboard writer, name the experiment!
exp_name = f'multi-source-asteroid-{NUM_SOURCES}-150eps'
exp_id = '{}_{}'.format(exp_name, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
writer = SummaryWriter('runs/{}'.format(exp_id))


# Define the relevant dictionaries
env_config = {
    'env': env, 
    'dataset': dataset, 
    'episodes': 150, 
    'max_steps': 1000,
    'stable_update_freq': 150,
    'save_freq': 1, 
    'writer': writer,
    'dense': True,
    'decay_rate': 0.0002,  # trial and error
    'decay_per_ep': True,
    'sample_rate': constants.RESAMPLE_RATE,
    'num_sources': NUM_SOURCES  # Pass the number of sources to the agent
}

save_path = os.path.join(constants.MODEL_SAVE_PATH, exp_name)
dataset_config = {
    'batch_size': 3, 
    'num_updates': 2, 
    'save_path': save_path
}

# clear save_path folder for each experiment
utils.clear_models_folder(save_path)

# asteroid_config for DPRNNTasNet (using defaults from RnnAgent as a base)
asteroid_config = {
    'n_filters': 64,    # Encoder/decoder number of filters
    'n_kernel': 16,     # Encoder/decoder kernel size
    'n_repeat': 4,      # Repeats of DPRNN block in a stack
    'n_basis': 512,     # Number of basis functions in encoder/decoder (like STFT window size)
    'hid_size': 128,    # Hidden units in LSTM within DPRNN
    'chunk_size': 100,  # Processing chunk size
    'hop_size': 50,     # Hop size for chunk processing
    'n_blocks': 8,      # Number of DPRNN blocks in a stack
    'causal': False,
    'mask_act': 'relu',
}

stft_config = {
    'hop_length': 64,
    'num_filters': 256,
    'direction': 'transform',
    'window_type': 'sqrt_hann'
}

rnn_agent = RnnAgent(
    env_config=env_config, 
    dataset_config=dataset_config,
    asteroid_config=asteroid_config,  # Use asteroid_config here
    stft_config=stft_config,
    learning_rate=.001,
    pretrained=False,
    num_sources=NUM_SOURCES  # Explicitly set the number of sources
)
# Run the training
torch.autograd.set_detect_anomaly(True)
rnn_agent.fit(visualize=False)

# Create directory for plots if it doesn't exist
plot_dir = './models/evaluation_data/'
os.makedirs(plot_dir, exist_ok=True)

# Generate plots after training
print("\nGenerating performance plots...")

# 1. Create plot of mean reward per episode
plt.figure(figsize=(10, 6))
plt.plot(rnn_agent.mean_episode_reward, 'b-')
plt.title(f'Mean Reward per Episode ({NUM_SOURCES} Sources)')
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.grid(True, alpha=0.3)
plt.savefig(f'{plot_dir}/mean_reward_{exp_name}.png', dpi=300)
plt.close()
print(f"Mean reward plot saved to: {plot_dir}/mean_reward_{exp_name}.png")

# 2. Create plot of cumulative reward
plt.figure(figsize=(10, 6))
plt.plot(rnn_agent.cumulative_reward, 'r-')
plt.title(f'Cumulative Reward ({NUM_SOURCES} Sources)')
plt.xlabel('Training Step')
plt.ylabel('Cumulative Reward')
plt.grid(True, alpha=0.3)
plt.savefig(f'{plot_dir}/cumulative_reward_{exp_name}.png', dpi=300)
plt.close()
print(f"Cumulative reward plot saved to: {plot_dir}/cumulative_reward_{exp_name}.png")

# 3. Create plot showing steps per episode
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

print("\nTraining and visualization complete!")
