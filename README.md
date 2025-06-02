<h1 align="center">AuralNav</h1>

AuralNav is an interactive research environment designed for reinforcement learning agents that learn to navigate complex acoustic scenes. Agents are tasked with identifying, localizing, and moving towards various sound sources within a simulated room. This platform facilitates research at the intersection of computer audition, deep learning, and reinforcement learning.

**Core Philosophy:** To enable agents to *learn to listen* and make intelligent decisions based on auditory input.

AuralNav leverages powerful open-source libraries:
- **OpenAI [`gymnasium`](https://gymnasium.farama.org/):** For standardized RL environment and agent interaction.
- **[`pyroomacoustics`](https://github.com/LCAV/pyroomacoustics):** For realistic acoustic simulations, including reverberation and sound propagation using ray-tracing.
- **[`asteroid`](https://github.com/asteroid-team/asteroid):** For state-of-the-art deep learning models for audio source separation (e.g., DPRNNTasNet).

In a typical AuralNav episode, an agent receives a multi-channel audio input representing the current soundscape of the room. It must then navigate towards each active sound source and 'interact' with it (e.g., turn it off). The number and types of sources can vary, creating diverse and challenging scenarios.

<br>

<!-- Consider updating the image if it doesn't reflect recent visualization changes -->
![AuralNav Environment](otoworld.png)

## Key Features

- **Advanced Audio Source Separation**: Utilizes **Asteroid's DPRNNTasNet** for high-fidelity time-domain separation of multiple overlapping sound sources.
- **Dynamic Multi-Source Environments**: Supports a configurable number of audio sources, allowing for complex and scalable auditory scenes.
- **Deep Q-Network (DQN) Agent**: Employs a DQN with experience replay and target networks for learning navigation policies based on auditory features.
- **Sophisticated Auditory Feature Extraction**: A novel feature pipeline processes separated sources and the mixture to derive rich spatial and acoustic cues (IPD, ILD, Volume) for the DQN.
- **Pygame-Powered Visualization**: Offers clear, real-time visualization of the agent's trajectory, source locations, detected sources, and environmental layout, greatly aiding in debugging and understanding agent behavior.
- **Flexible Environment Configuration**: Easily customize room acoustics, source types, agent properties, and task objectives through configuration files.
- **TensorBoard Integration**: Comprehensive logging of training metrics (rewards, losses, episode statistics) for monitoring and analysis.


## Installation 
Clone the repository
```bash
git clone https://github.com/yourusername/AuralNav.git
```
Create a virtual environment: 
```bash
python -m venv venv
``` 
Activate the environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install requirements:
```bash
pip install -r requirements.txt
```
Install ffmpeg (if not already installed):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS with Homebrew
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```
If using a **CUDA-enabled GPU (highly recommended)**, install PyTorch with CUDA support:  
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```
otherwise: 
```bash
pip install torch torchvision torchaudio
```


## Additional Installation Notes - Linux
* Linux users may need to install the sound file library if it is not present in the system:
```bash
sudo apt-get install libsndfile1
```

## Visualizing Training Progress
AuralNav provides two ways to visualize training progress:

1. **Real-time episode visualizations** showing the agent's trajectory and source detections
```bash
python experiment/experiment1.py
```
Visualizations are automatically generated and saved to the `visualizations/` directory.

2. **TensorBoard metrics** for tracking rewards, losses, and other training metrics
```bash
tensorboard --logdir=runs --port=6006
```
Then open your browser and navigate to `http://localhost:6006`.

Alternatively, you can generate plots using the included utility:
```bash
python src/plot_runs.py
```

## Running Experiments
AuralNav includes two experiment configurations:

```bash
# Run experiment 1 (150 episodes)
cd experiment/
python experiment1.py

# Run experiment 2 (200 episodes)
python experiment2.py
```

You can customize these experiments or create your own by modifying the configuration parameters. A GPU running CUDA is highly recommended for meaningful experiments.

### Is It Running Properly?
You should see a message indicating the experiment is running, such as this:
```
------------------------------ 
- Starting to Fit Agent
------------------------------- 
```

The terminal will display information about source locations, agent movement, and rewards received. You'll also see messages when the agent successfully detects sources:

```
Agent has found source ../sounds/siren/siren.wav.
Agent loc: [5.48227103 5.91024018], Source loc: [5.635482199008357, 5.0]
Visualization: Recorded detection of source 2 at step 466
```

## Key Components

### Agent Architecture
The agent uses a recurrent neural network (RNN) to process audio inputs and extract features, which are then fed into a Deep Q-Network (DQN) to determine the optimal action. The system includes:

- Experience replay for stable learning
- Target networks for reducing overestimation bias
- Epsilon-greedy exploration strategy
- Audio source separation capabilities through Asteroid

### Visualization System
The visualization system provides detailed insights into the agent's behavior:

- Room layout with source positions
- Agent trajectory with color gradient to show progression
- Source detection events with connecting lines to sources
- Detection markers with step information
- Episode summary statistics

## Credits and Acknowledgments

This project is based on the original OtoWorld framework developed by Omkar Ranadive, Grant Gasser, David Terpay, and Prem Seetharaman. AuralNav extends and enhances the core functionality with improved visualizations, Gymnasium compatibility, and better audio processing capabilities using Asteroid.

### Original OtoWorld Paper:
```
@inproceedings {otoworld
    author = {Omkar Ranadive and Grant Gasser and David Terpay and Prem Seetharaman},
    title = "OtoWorld: Towards Learning to Separate by Learning to Move",
    journal = "Self Supervision in Audio and Speech Workshop, 37th International Conference on Machine Learning ({ICML} 2020), Vienna, Austria",
    year = 2020
}
```

### Original OtoWorld Repository:
https://github.com/pseeth/otoworld

## Future Roadmap and Research Directions

AuralNav is an ongoing research project with several planned enhancements and directions for future work:

### Core Environment Enhancements & Novel Tasks

#### Dynamic and Interactive Environments
- **Development**: Move beyond static sound sources to implement dynamic environments where sound sources move, change their characteristics (e.g., volume, pitch), or appear/disappear over time
- **Impact**: More realistic and challenging scenarios that allow for research into adaptive learning and real-time decision-making

#### Complex Soundscapes
- **Development**: Replace simple tones/sounds with complex, real-world soundscapes (e.g., urban environments, forests, offices)
- **Impact**: Significantly increases the difficulty of source separation and addresses the sim-to-real gap in audio research

#### Multi-Agent Scenarios
- **Development**: Extend to support multiple agents that can cooperate or compete in tasks requiring coordination based on auditory cues
- **Impact**: Explores the intersection of multi-agent reinforcement learning and computer audition

#### Source Matching
- **Development**: Implement a source matching task where agents must learn to match the correct source to the correct agent
- **Impact**: Addresses the problem of source identification and matching in multi-agent systems

#### Realistic Sensor Modeling
- **Development**: Implement more realistic sensor models that simulate the limitations of real-world microphones
- **Impact**: Bridges the gap between simulation and real-world deployment

### Algorithmic Innovations & Learning Paradigms

#### Self-Supervised and Unsupervised Learning
- **Development**: Explore self-supervised techniques to pre-train the agent's auditory perception module
- **Impact**: Addresses sample efficiency problems in reinforcement learning

#### Meta-Learning for Rapid Adaptation
- **Development**: Apply meta-learning techniques for rapid adaptation to new environments
- **Impact**: Enables quick deployment of agents in unseen environments

#### Neuro-Inspired Architectures
- **Development**: Design neural network architectures inspired by the human auditory system
- **Impact**: Leads to more efficient and biologically plausible audio processing algorithms

#### Curriculum Learning and Task Decomposition
- **Development**: Implement curriculum learning strategies where the agent gradually learns more complex tasks
- **Impact**: Improves learning efficiency and enables agents to solve more challenging tasks
