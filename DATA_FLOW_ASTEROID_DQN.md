# Data Flow: Asteroid DPRNNTasNet Separation and DQN for AuralNav

This document outlines the data flow for the audio source separation and reinforcement learning (RL) agent interaction within the AuralNav project, specifically focusing on the integration of the Asteroid library's DPRNNTasNet for source separation and the subsequent Deep Q-Network (DQN) for agent decision-making.

## Core Idea

The system uses DPRNNTasNet to separate audio sources from a stereo mixture in the time domain. The DQN then uses features derived from both the original stereo mixture and the separated mono sources to make decisions. A key aspect is the use of Time-Frequency (TF) masks, estimated from the separated sources, to attribute spatial cues (IPD, ILD, Volume) from the original mixture to individual sources.

## Key Components and Files

*   **`agent.py`**: Implements the `AgentBase` and `RnnAgent` classes. `RnnAgent` manages the main training loop, interaction with the environment, experience replay, and calls to the separation and DQN models.
*   **`models_asteroid.py`**: 
    *   `RnnSeparator`: A wrapper around the `AsteroidSeparationModel`. It's called by the `RnnAgent` to perform source separation.
    *   `DQN`: The Deep Q-Network. Its `forward` method implements the complex feature extraction pipeline (STFT, spatial cue calculation, TF masking, feature aggregation) and the subsequent MLP for Q-value estimation.
*   **`asteroid_separation_model.py`**:
    *   `AsteroidSeparationModel`: A wrapper for Asteroid's `DPRNNTasNet` model. It handles the direct call to the pre-trained Asteroid model for time-domain source separation.
*   **`DPRNNTasNet` (from Asteroid library)**: The core deep learning model performing time-domain audio source separation. It takes a multi-channel (stereo) audio signal and outputs estimated mono waveforms for each source.
*   **`torchaudio.transforms.Spectrogram`**: Used within `DQN.forward` to convert time-domain audio signals (both mixture and separated sources) into their Short-Time Fourier Transform (STFT) representations (complex-valued spectrograms).
*   **`experiment_asteroid.py`**: The main script used to configure and run training or evaluation experiments for the `RnnAgent` using the Asteroid-based models.
*   **`audio_env.py`**: Defines the `AudioEnv` class, which simulates the acoustic environment, generates audio mixtures based on agent and source positions, and provides observations and rewards to the agent.
*   **`rl_utils.py`**: Contains utilities like `RLDataset` for managing the experience replay buffer.

## Data Flow Detailed

The data flow can be primarily understood by looking at two main phases: action selection (`RnnAgent.choose_action`) and model updating (`RnnAgent.update`).

### 1. `RnnAgent.choose_action` (Agent Deciding Next Action)

1.  **Environment State**: The `AudioEnv` provides the current state, which includes `mix_audio_new_state` – a raw stereo audio tensor representing what the agent currently 'hears' (Shape: `(Channels, NumSamples)`, e.g., `(2, 32000)`).
2.  **Input Preparation (`RnnAgent`)**: 
    *   `mix_audio_new_state` is converted to a batched tensor (Shape: `(1, Channels, NumSamples)`).
3.  **Source Separation (`RnnSeparator.forward` in `models_asteroid.py`)**:
    *   The batched stereo audio is passed to `RnnSeparator`.
    *   This calls `AsteroidSeparationModel.forward` (in `asteroid_separation_model.py`).
    *   `AsteroidSeparationModel` passes the audio to the underlying `DPRNNTasNet` model.
    *   `DPRNNTasNet` separates the sources in the time domain.
    *   Output is a dictionary: `separator_output_dict = {'estimated_sources': es_tensor, 'mix_audio': mix_tensor}`.
        *   `es_tensor` (Estimated Sources): Mono audio for each source (Shape: `(1, NumSources, NumSamples)`).
        *   `mix_tensor` (Mixture Audio): The original input stereo mix (Shape: `(1, Channels, NumSamples)`).
4.  **Feature Extraction & Q-Value Estimation (`DQN.forward` in `models_asteroid.py`)**:
    *   Receives `separator_output_dict` and `agent_info` (e.g., agent's current position/orientation).
    *   **STFT Stage**:
        *   The `mix_audio` (stereo) is converted to its complex STFT: `stft_mix_stereo` (Shape e.g., `(1, Channels, FreqBins, TimeFrames)`).
        *   Each mono `estimated_source` is converted to its complex STFT: `stft_mono_source_i` (Shape e.g., `(1, 1, FreqBins, TimeFrames)`).
    *   **Primary Spatial Cue Calculation (from stereo mixture STFT)**:
        *   Interaural Phase Difference (IPD), Interaural Level Difference (ILD), and overall Volume (VOL) are calculated from `stft_mix_stereo`. After processing, these are typically shaped `(1, TimeFrames, FreqBins)`.
    *   **Time-Frequency (TF) Mask Estimation**:
        *   For each source `i`:
            *   Magnitude of `stft_mono_source_i` -> `magnitude_mono_source`.
            *   Magnitude of `stft_mix_stereo` (averaged over channels for a mono reference) -> `magnitude_mix_mono_ref`.
            *   `estimated_mask_i = magnitude_mono_source / (magnitude_mix_mono_ref + epsilon)` (Shape: `(1, TimeFrames, FreqBins)`).
    *   **Masked Feature Application**: 
        *   `masked_ipd_source_i = estimated_mask_i * ipd_from_mixture`.
        *   Similarly for `masked_ild_source_i` and `masked_vol_source_i`.
    *   **Feature Aggregation**:
        *   The three masked features for each source `i` are averaged over Time and Frequency dimensions, resulting in three scalar values per source (mean masked IPD, ILD, Vol).
        *   These scalar features are concatenated across all sources (e.g., `(1, 3 * NumSources)`).
    *   **Concatenate Agent Information**: 
        *   The agent's own state information (`agent_info`, e.g., `(1, NumAgentFeatures)`) is concatenated with the aggregated audio features.
        *   Final feature vector `X` (Shape e.g., `(1, 3 * NumSources + NumAgentFeatures)`).
    *   **MLP for Q-Values**: 
        *   `X` is passed through a Multi-Layer Perceptron (MLP) to produce Q-values for each possible action (Shape: `(1, NumActions)`).
5.  **Action Selection (`RnnAgent`)**: The action with the highest Q-value is typically chosen (or an action is chosen via an epsilon-greedy policy).

### 2. `RnnAgent.update` (Agent Learning from Experience)

1.  **Batch Sampling**: A batch of experiences (state, action, reward, next_state, agent_info) is sampled from the `RLDataset` (replay buffer).
    *   `data['mix_audio_prev_state']` (Shape: `(BatchSize, Channels, NumSamples)`)
    *   `data['mix_audio_new_state']` (Shape: `(BatchSize, Channels, NumSamples)`)
    *   `data['action']`, `data['reward']`, `data['agent_info']`.
2.  **Processing Previous State (for current Q-values)**:
    *   `data['mix_audio_prev_state']` is passed through `self.rnn_model` (the online `RnnSeparator`).
    *   The output dictionary and `data['agent_info']` are passed to `self.dqn_model` (the online `DQN`).
    *   The `DQN.forward` proceeds as described above, yielding `q_values` `(BatchSize, NumActions)`.
    *   The Q-value for the action actually taken (`data['action']`) is selected: `q_value = q_values.gather(1, data['action'])`.
3.  **Processing Next State (for target Q-values)**:
    *   `data['mix_audio_new_state']` is passed through `self.rnn_model_stable` (the target `RnnSeparator`).
    *   The output dictionary and `data['agent_info']` are passed to `self.dqn_model_stable` (the target `DQN`).
    *   The `DQN.forward` proceeds, yielding `next_q_values_all_actions` `(BatchSize, NumActions)`.
    *   The maximum Q-value for the next state is selected (e.g., `next_q_value = next_q_values_all_actions.max(1)[0].unsqueeze(-1)`).
4.  **Loss Calculation**: 
    *   The target for the Bellman equation is calculated: `expected_q_value = data['reward'] + gamma * next_q_value`.
    *   The loss (e.g., MSE loss) is computed: `loss = F.mse_loss(q_value, expected_q_value.detach())`.
5.  **Backpropagation and Optimization**: The loss is backpropagated through the online `dqn_model` and `rnn_model` (if its weights are being trained/fine-tuned), and the optimizer updates the weights.
6.  **Target Network Update**: Periodically, the weights of the online networks (`self.rnn_model`, `self.dqn_model`) are copied to the target networks (`self.rnn_model_stable`, `self.dqn_model_stable`).

## Summary of Key Tensor Transformations in `DQN.forward`

*   **Input Audio (Time Domain)**:
    *   Mixture: `(B, C, N_samples_mix)`
    *   Separated Source `i`: `(B, 1, N_samples_src)` (originally `(B, NumSources, N_samples_src)` from separator)
*   **STFT Output (Complex Frequency Domain)**:
    *   Mixture STFT: `(B, C, FreqBins, TimeFrames)`
    *   Source `i` STFT: `(B, 1, FreqBins, TimeFrames)`
*   **Spatial Features from Mixture (Real-valued, after processing)**:
    *   IPD_mix, ILD_mix, VOL_mix: Each `(B, TimeFrames, FreqBins)`
*   **TF Mask for Source `i`**: `(B, TimeFrames, FreqBins)`
*   **Masked Spatial Features for Source `i`**: Each `(B, TimeFrames, FreqBins)`
*   **Aggregated Features per Source `i` (after mean over T,F)**: `(B, 3)` (for IPD, ILD, Vol means)
*   **Concatenated Audio Features (all sources)**: `(B, 3 * NumSources)`
*   **Final Feature Vector (with agent_info)**: `(B, 3 * NumSources + NumAgentFeatures)`
*   **Output Q-Values**: `(B, NumActions)`

This detailed flow should provide a clear understanding of how audio is processed and utilized by the RL agent in the AuralNav system with Asteroid integration.
