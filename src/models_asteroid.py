import os
import gymnasium as gym
import numpy as np
import torch
import logging
import agent
import utils
import constants
import builders
import audio_processing
import agent
from datasets import BufferData, RLDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import STFT
from asteroid_separation_model import AsteroidSeparationModel


class RnnAgent(agent.AgentBase):
    def __init__(self, env_config, dataset_config, asteroid_config=None, stft_config=None, 
                 verbose=False, autoclip_percentile=10, learning_rate=.001, pretrained=False,
                 num_sources=None):
        """
        Args:
            env_config (dict): Dictionary containing the audio environment config
            dataset_config (dict): Dictionary consisting of dataset related parameters. List of parameters
            'batch_size' : Denotes the batch size of the samples
            'num_updates': Amount of iterations we run the training for in each pass.
             Ex - If num_updates = 5 and batch_size = 25, then we run the update process 5 times where in each run we
             sample 25 data points.
             'sampler': The sampler to use. Ex - Weighted Sampler, Batch sampler etc

            asteroid_config (dict): Dictionary containing the parameters for the Asteroid model (e.g., DPRNNTasNet).
            stft_config (dict):  Dictionary containing the parameters for STFT
        """

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('DEVICE:', self.device)
        # Get number of sources from environment if not specified, or from argument
        if num_sources is None and hasattr(env_config, 'get'):
            self.num_sources = env_config.get('num_sources', 2)
        elif num_sources is not None:
            self.num_sources = num_sources
        else:
            self.num_sources = 2 # Fallback default

        # Prepare Asteroid model configuration
        if asteroid_config is None:
            # Default parameters for DPRNNTasNet if no config provided
            processed_asteroid_config = {
                'n_filters': 64,    # Encoder/decoder number of filters
                'n_kernel': 16,     # Encoder/decoder kernel size
                'n_repeat': 4,      # Repeats of DPRNN block in a stack
                'n_basis': 512,     # Number of basis functions in encoder/decoder (like STFT window size)
                'hid_size': 128,    # Hidden units in LSTM within DPRNN
                'chunk_size': 100,  # Processing chunk size
                'hop_size': 50,     # Hop size for chunk processing
                'n_blocks': 8,      # Number of DPRNN blocks in a stack
                'causal': False,
                'mask_act': 'relu'
            }
        else:
            processed_asteroid_config = asteroid_config.copy()
        
        # Ensure n_src and sample_rate are set for AsteroidSeparationModel
        # These are passed directly to AsteroidSeparationModel, not as part of asteroid_kwargs dict
        # but asteroid_kwargs in RnnSeparator becomes model_opts in AsteroidSeparationModel
        # So, we ensure they are in the dict that AsteroidSeparationModel will receive.
        # AsteroidSeparationModel itself expects num_sources and sample_rate as direct args.
        # The asteroid_kwargs in RnnSeparator is for other DPRNNTasNet specific params.
        # The actual n_src and sample_rate for DPRNNTasNet are handled inside AsteroidSeparationModel init.
        # So, processed_asteroid_config is what we pass to RnnSeparator's asteroid_kwargs.

        if stft_config is None:
            self.stft_diff = STFT(
                hop_length=128, filter_length=512, 
                direction='transform', num_filters=512
            )
        else:
            self.stft_diff = STFT(**stft_config)

        # Extract sample_rate for RLDataset's use, as AgentBase.__init__ doesn't expect it.
        self.sample_rate = env_config['sample_rate']

        # Prepare config for AgentBase, excluding 'sample_rate' as it's not an explicit param for AgentBase.__init__
        agent_base_env_config = {k: v for k, v in env_config.items() if k != 'sample_rate'}
        
        # Initialize the Agent Base class
        super().__init__(**agent_base_env_config)

        # Uncomment this to find backprop errors
        # torch.autograd.set_detect_anomaly(True)

        # Initialize the rnn model (RnnSeparator wrapping AsteroidSeparationModel)
        self.rnn_model = RnnSeparator(num_sources=self.num_sources, 
                                        sample_rate=self.sample_rate, 
                                        asteroid_kwargs=processed_asteroid_config).to(self.device)
        self.rnn_model_stable = RnnSeparator(num_sources=self.num_sources, 
                                             sample_rate=self.sample_rate, 
                                             asteroid_kwargs=processed_asteroid_config).to(self.device)

        # Load pretrained model
        if pretrained:
            # self.rnn_model.rnn_model.load_state_dict(torch.load(os.path.join(self.SAVE_PATH, "rnn_separator.pth")))
            # The above line is for the old SeparationModel. AsteroidSeparationModel has its own load method.
            # We might need to adjust saving/loading logic if using pretrained Asteroid models or custom checkpoints.
            # For now, if pretrained is True, we assume the AsteroidSeparationModel handles its own pretrained weights if any,
            # or this needs to be adapted to load a state_dict directly into self.rnn_model.asteroid_model
            # Example: self.rnn_model.asteroid_model.load_state_dict(torch.load(PATH_TO_ASTEROID_WEIGHTS))
            # This part needs careful handling based on how pretrained models are managed.
            if os.path.exists(os.path.join(self.SAVE_PATH, "rnn_separator_asteroid.pth")):
                 self.rnn_model.load_state_dict(torch.load(os.path.join(self.SAVE_PATH, "rnn_separator_asteroid.pth")))
                 print(f"Loaded pretrained Asteroid RnnSeparator from {os.path.join(self.SAVE_PATH, 'rnn_separator_asteroid.pth')}")
            else:
                 print(f"Warning: Pretrained specified but rnn_separator_asteroid.pth not found in {self.SAVE_PATH}")

        # Initialize dataset related parameters from dataset_config
        self.bs = dataset_config['batch_size']
        # logging.info(f"RnnAgent.__init__: batch_size: {self.bs}")
        self.num_updates = dataset_config.get('num_updates', 1) # Default to 1 if not specified
        self.dynamic_dataset = RLDataset(buffer=self.dataset, sample_size=self.bs, sample_rate=self.sample_rate)
        self.dataloader = torch.utils.data.DataLoader(self.dynamic_dataset, batch_size=self.bs, shuffle=False) # Shuffle is usually True for training, but RLDataset handles sampling

        # Initialize the DQN model
        self.dqn_model = DQN(network_params={
            'filter_length': self.stft_diff.filter_length,
            'total_actions': self.env.action_space.n,
            'stft_diff': self.stft_diff,
            'num_sources': self.num_sources  # Pass num_sources to DQN
        }).to(self.device)
        self.dqn_model_stable = DQN(network_params={
            'filter_length': self.stft_diff.filter_length,
            'total_actions': self.env.action_space.n,
            'stft_diff': self.stft_diff,
            'num_sources': self.num_sources  # Pass num_sources to DQN
        }).to(self.device)
        self.dqn_model_stable.load_state_dict(self.dqn_model.state_dict())

        # Initialize dataset related parameters
        self.bs = dataset_config['batch_size']
        self.num_updates = dataset_config['num_updates']
        self.dynamic_dataset = RLDataset(buffer=self.dataset, sample_size=self.bs, sample_rate=self.sample_rate)
        self.dataloader = torch.utils.data.DataLoader(self.dynamic_dataset, batch_size=self.bs)

        # Initialize the optimizer
        params = list(self.rnn_model.parameters()) + list(self.dqn_model.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # Folder path where the model will be saved
        self.SAVE_PATH = dataset_config['save_path']
        self.grad_norms = None
        self.percentile = autoclip_percentile

    def update(self):
        """
        Runs the main training pipeline. Sends mix to RNN separator, then to the DQN. 
        Calculates the q-values and the expected q-values, comparing them to get the loss and then
        computes the gradient w.r.t to the entire differentiable pipeline.
        """
          # Run the update only if samples >= batch_size
        if len(self.dataset.items) < self.bs:
            # logging.info(f"Not enough samples in the dataset. Current size: {len(self.dataset.items)}, batch size: {self.bs}")
            return

        for index, data in enumerate(self.dataloader):
            if index > self.num_updates:
                break

            # Get the total number of time steps
            total_time_steps = data['mix_audio_prev_state'].shape[-1]

            # Pass the batched multi-channel audio directly to self.rnn_model
            # data['mix_audio_prev_state'] is (B, C, N)
            prev_state_audio_input = data['mix_audio_prev_state'].float().to(self.device)
            data['action'] = data['action'].to(self.device)
            data['reward'] = data['reward'].to(self.device)
            agent_info = data['agent_info'].to(self.device)

            # Get the separated sources by running through RNN separation model
            # rnn_model is an alias for rnn_separator
            output = self.rnn_model(prev_state_audio_input) 
            # Ensure the original stereo mix is available for DQN's feature calculation
            output['mix_audio'] = prev_state_audio_input 
            logging.debug(f"RnnAgent.update: prev_state_audio_input.shape: {prev_state_audio_input.shape}")
            logging.debug(f"RnnAgent.update: output after rnn_model: { {k: v.shape for k, v in output.items()} }")

            # Get q values from the online network
            q_values = self.dqn_model(
                output,
                agent_info,
                total_time_steps
            )
            # logging.info(f"RnnAgent.update: q_values before gather: {q_values.shape}, data['action']: {data['action'].shape}")
            q_values = q_values.gather(1, data['action']).squeeze(1)

            with torch.no_grad():
                # Now, get q-values for the next-state
                # Get the total number of time steps
                total_time_steps = data['mix_audio_new_state'].shape[-1]

                # Pass the batched multi-channel audio directly for next_state Q-values
            # data['mix_audio_new_state'] is (B, C, N)
            new_state_audio_input = data['mix_audio_new_state'].float().to(self.device)
            # rnn_model_stable is an alias for rnn_separator_stable
            stable_output = self.rnn_model_stable(new_state_audio_input)
            # Ensure the original stereo mix is available for DQN's feature calculation
            stable_output['mix_audio'] = new_state_audio_input
            logging.debug(f"RnnAgent.update: new_state_audio_input.shape: {new_state_audio_input.shape}")
            logging.debug(f"RnnAgent.update: stable_output after rnn_model_stable: { {k: v.shape for k, v in stable_output.items()} }")
            next_q_values = self.dqn_model_stable(stable_output, agent_info, total_time_steps).max(1)[0].unsqueeze(-1)
        
            # Calculate the expected q values
            expected_q_values = data['reward'] + self.gamma * next_q_values 

            # Calculate the loss
            loss = F.mse_loss(q_values, expected_q_values.detach())
            self.losses.append(loss)
            self.writer.add_scalar('Loss/train', loss, len(self.losses))

            # Perform backpropagation
            self.optimizer.zero_grad()
            # logging.info("RnnAgent.update: Before loss.backward()")
            loss.backward()
            # logging.info("RnnAgent.update: After loss.backward()")

            # Auto clipping
            # self.grad_norms = nn.utils.clip_grad_norm_(
            #     list(self.rnn_model.parameters()) + list(self.dqn_model.parameters()), 
            #     max_norm=constants.MAX_GRAD_NORM, 
            #     norm_type=2
            # )
            self.grad_norms = utils.autoclip(self.rnn_model, self.percentile, self.grad_norms)

            self.optimizer.step()

        # Log the loss and gradient norms
        # # logging.info(f"Loss: {loss.item()}, Grad Norms: {self.grad_norms.item()}")
        
        # Update the total time steps
        # self.dataset.total_time_steps += 1

        # return loss.item(), self.grad_norms.item()

    def choose_action(self):
        """
        Runs a forward pass though the RNN separator and then the Q-network. An action is choosen 
        by taking the argmax of the output vector of the network, where the output is a 
        probability distribution over the action space (via softmax).

        Returns:
            action (int): the argmax of the q-values vector 
        """
        with torch.no_grad():
            # Get the latest state from the buffer
            data = self.dataset[self.dataset.last_ptr]
            # logging.info(f"RnnAgent.choose_action: data['mix_audio_new_state'].shape: {data['mix_audio_new_state'].shape if 'mix_audio_new_state' in data else 'N/A'}")
            # logging.info(f"RnnAgent.choose_action: data.keys(): {data.keys()}")
        
            # Perform the forward pass (RNN separator => DQN)
            # data['mix_audio_new_state'] is (Channels, Time) from dataset for a single item, e.g., (2, 8000)
            total_time_steps = data['mix_audio_new_state'].shape[-1]
            data['mix_audio'] = data['mix_audio_new_state'].float().view(
                -1, 1, total_time_steps).to(self.device)
            # logging.info(f"RnnAgent.choose_action: data['mix_audio_new_state'].shape: {data['mix_audio_new_state'].shape if 'mix_audio_new_state' in data else 'N/A'}")
            # logging.info(f"RnnAgent.choose_action: next_state data['mix_audio'].shape: {data['mix_audio'].shape}")
            # Add batch dimension for model processing: (1, Channels, Time)
            # original_stereo_mix_gpu = data['mix_audio'].float().unsqueeze(0).to(self.device)
            # # logging.info(f"RnnAgent.choose_action: original_stereo_mix_gpu.shape: {original_stereo_mix_gpu.shape}")
            # self.rnn_model (RnnSeparator -> AsteroidSeparationModel) now expects a stereo mix
            # and returns a dictionary: 
            # {'estimated_sources': (mono_sources_tensor), 'mix_audio': (original_stereo_mix_tensor)}
            # # logging.info(f"Input shape: {original_stereo_mix_gpu.shape}")
            separator_output_dict = self.rnn_model(data['mix_audio'])
            # logging.info(f"RnnAgent.choose_action: separator_output_dict.keys(): {separator_output_dict.keys()}")
            # # logging.info(f"Output shape: {separator_output_dict['estimated_sources'].shape}")

            # separator_output_dict now contains the correct 'estimated_sources' (mono) 
            # and 'mix_audio' (original stereo) needed by DQN.forward.

            # Ensure agent_info also has a batch dimension and is on the correct device
            agent_info_cpu = data['agent_info'] # Assuming (N_features)
            agent_info_gpu = agent_info_cpu.to(self.device) # Shape (1, N_features)

            # action = argmax(q-values)
            # Pass the separator_output_dict directly to dqn_model
            q_values = self.dqn_model(separator_output_dict, agent_info_gpu, total_time_steps)
            action = q_values.max(1)[1].unsqueeze(-1)
            action = action[0].item()

            action = int(action)

            return action


    def update_stable_networks(self):
        self.dqn_model_stable.load_state_dict(self.dqn_model.state_dict())
        self.rnn_model_stable.load_state_dict(self.rnn_model.state_dict())

    def save_model(self, name):
        """
        Args:
            name (str): Name contains the episode information (To give saved models unique names)
        """
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        
        torch.save(self.rnn_model.state_dict(), os.path.join(self.SAVE_PATH, f"{name}_rnn_separator_asteroid.pth"))
        torch.save(self.dqn_model.state_dict(), os.path.join(self.SAVE_PATH, f"{name}_dqn_model.pth"))
        print(f"Models saved to {self.SAVE_PATH} with prefix {name}")


class RnnSeparator(nn.Module):
    def __init__(self, num_sources, sample_rate, verbose=False, asteroid_kwargs=None):
        super(RnnSeparator, self).__init__()
        self.num_sources = num_sources
        self.sample_rate = sample_rate
        if asteroid_kwargs is None:
            asteroid_kwargs = {}
        # Initialize AsteroidSeparationModel with num_sources, sample_rate, and other asteroid specific args
        self.rnn_model = AsteroidSeparationModel(num_sources=num_sources, 
                                                 sample_rate=sample_rate, 
                                                 **asteroid_kwargs)

    def forward(self, mix_audio_tensor):
        """
        Args:
            mix_audio_tensor (torch.Tensor): Raw mixture audio tensor of shape (batch, channels, time) or (batch, time).
        Returns:
            dict: A dictionary containing the separated sources and the original mix.
                  {'estimated_sources': tensor (batch, num_sources, time),
                   'mix_audio': tensor (batch, channels, time) or (batch, time) for DQN's STFT}
        """
        # logging.info(f"RnnSeparator.forward: mix_audio_tensor.shape: {mix_audio_tensor.shape}")
        output = self.rnn_model(mix_audio_tensor)
        # logging.info(f"RnnSeparator.forward: output: {output.keys()}")
        # logging.info(f"RnnSeparator.forward: output['estimated_sources'].shape: {output['estimated_sources'].shape}")
        # logging.info(f"RnnSeparator.forward: output['mix_audio'].shape: {output['mix_audio'].shape}")
        return output


class DQN(nn.Module):
    def __init__(self, network_params):
        """
        The main DQN class, which takes the output of the RNN separator and input and
        returns q-values (prob dist of the action space)

        Args:
            network_params (dict): Dict of network parameters including:
                - filter_length: Length of the filter for STFT
                - total_actions: Number of possible actions
                - stft_diff: STFT transformation object
                - num_sources: Number of audio sources to handle

        Returns:
            q_values (torch.Tensor): q-values 
        """
        super(DQN, self).__init__()

        self.stft_diff = network_params['stft_diff']
        self.num_sources = network_params.get('num_sources', 2)  # Default to 2 if not specified
        
        # Adjust input size based on number of sources (3 features per source + 3 for agent info)
        input_size = 3 * self.num_sources + 3
        
        # Create network layers with adjusted input size
        self.fc = nn.Linear(input_size, network_params['total_actions'])
        self.prelu = nn.PReLU()

    def forward(self, separator_output, agent_info, total_time_steps):
        # separator_output is a dict from RnnSeparator:
        # 'estimated_sources': tensor (batch, num_actual_sources, num_samples_src) - mono sources
        # 'mix_audio': tensor (batch, 2, num_samples_mix) - original stereo mix
        # logging.info(f"DQN.forward: input separator_output['mix_audio'].shape: {separator_output.get('mix_audio').shape if separator_output.get('mix_audio') is not None else 'N/A'}")
        # logging.info(f"DQN.forward: input separator_output['estimated_sources'].shape: {separator_output.get('estimated_sources').shape if separator_output.get('estimated_sources') is not None else 'N/A'}")
        # logging.info(f"DQN.forward: input agent_info.shape: {agent_info.shape}")
       
        mix_audio = separator_output['mix_audio']  # Shape: (B, 2, N_samples_mix)
        estimated_sources = separator_output['estimated_sources']  # Shape: (B, N_actual_src, N_samples_src)
        
        batch_size = mix_audio.shape[0]
        
        # 1. Perform STFT on the original stereo mixture (output is complex)
        # stft_data_mix shape: (B, 2, T, F) where T=num_frames, F=num_freq_bins
        stft_data_mix = self.stft_diff(mix_audio, direction='transform')
        
        # 2. Extract Magnitude and Phase from stereo STFT
        magnitude_mix_stereo = torch.abs(stft_data_mix)  # Shape: (B, 2, T, F)
        phase_mix_stereo = torch.angle(stft_data_mix)    # Shape: (B, 2, T, F)

        # 3. Calculate IPD, ILD, VOL features from STFT data
        num_channels_mix = stft_data_mix.shape[3] # Channel is last: (B,T,F,C)
        
        if num_channels_mix == 1:  # Mono mixture
            # IPD and ILD are conceptually zero for mono. Shape: (B, T, F)
            ipd_mix = torch.zeros_like(magnitude_mix_stereo.squeeze(3)) # Squeeze channel dim (C=1 for mono mix)
            ild_mix = torch.zeros_like(magnitude_mix_stereo.squeeze(3)) # Squeeze channel dim (C=1 for mono mix)
            # VOL is just the magnitude of the single channel. Shape: (B, T, F)
            vol_mix = magnitude_mix_stereo.squeeze(3) # Squeeze channel dim (C=1 for mono mix)
        elif num_channels_mix >= 2:  # Stereo or multi-channel mixture
            # IPD: (B, T, F)
            ipd_mix = torch.fmod(phase_mix_stereo[..., 1] - phase_mix_stereo[..., 0], np.pi) # Access channels at last dim
            # ILD: (B, T, F)
            ild_mix = 20 * torch.log10(
                torch.abs(magnitude_mix_stereo[..., 0]) /
                (torch.abs(magnitude_mix_stereo[..., 1]) + 1e-8) + 1e-8
            ) # Access channels at last dim
            # VOL: (B, T, F) - Sum of magnitudes from all channels (or first two if more than 2)
            vol_mix = torch.sum(magnitude_mix_stereo[..., :2], dim=3) # Sum first two channels (at last dim) for consistency
        else: # Should not happen if stft_data_mix has a channel dimension
            raise ValueError(f"Unexpected number of channels {num_channels_mix} in stft_data_mix")

        # For mask calculation: reference mono magnitude from the mix. Shape: (B, T, F)
        magnitude_mix_mono_ref = magnitude_mix_stereo.mean(dim=3) # Mean across channel dim (last dim)

        actual_sources_count = estimated_sources.shape[1]
        if actual_sources_count != self.num_sources:
            logging.warning(f"DQN.forward: Separator produced {actual_sources_count} sources, but DQN expects {self.num_sources}. Features will be based on min(actual, expected).")

        features = []
        
        # Log shapes of base features from mixture (once per forward pass)
        # logging.info(f"DQN.forward PRE-LOOP: ipd_mix.shape: {ipd_mix.shape if hasattr(ipd_mix, 'shape') else 'N/A'}")
        # logging.info(f"DQN.forward PRE-LOOP: ild_mix.shape: {ild_mix.shape if hasattr(ild_mix, 'shape') else 'N/A'}")
        # logging.info(f"DQN.forward PRE-LOOP: magnitude_mix_stereo.shape: {magnitude_mix_stereo.shape if hasattr(magnitude_mix_stereo, 'shape') else 'N/A'}")

        # Loop over sources to calculate masks and apply them
        for i in range(min(actual_sources_count, self.num_sources)):
            current_source_audio = estimated_sources[:, i, :]  # (B, N_samples_src)
            current_source_audio_ch = current_source_audio.unsqueeze(1)  # (B, 1, N_samples_src)
            
            # STFT of the current (mono) separated source (output is complex)
            stft_mono_source = self.stft_diff(current_source_audio_ch, direction='transform') # (B, 1, T, F)
            
            # Magnitude of the mono source STFT
            magnitude_mono_source = torch.abs(stft_mono_source.squeeze(3)) # Squeeze channel dim (C_in=1 for source)  # (B, T, F)
            # logging.info(f"DQN.forward LOOP source {i}: magnitude_mono_source.shape: {magnitude_mono_source.shape if hasattr(magnitude_mono_source, 'shape') else 'N/A'}")
            # logging.info(f"DQN.forward LOOP source {i}: magnitude_mix_mono_ref.shape: {magnitude_mix_mono_ref.shape if hasattr(magnitude_mix_mono_ref, 'shape') else 'N/A'}")
            # Estimate Time-Frequency mask for this source
            estimated_mask = magnitude_mono_source / (magnitude_mix_mono_ref + 1e-8)  # (B, T, F)
            # estimated_mask = torch.clamp(estimated_mask, 0.0, 1.0) # Optional: clamp mask values

            # Apply estimated mask to the (B,T,F) features. estimated_mask is (B,T,F,1)
            masked_ipd_source = estimated_mask * ipd_mix # Both are (B,T,F)
            masked_ild_source = estimated_mask * ild_mix # Both are (B,T,F)
            masked_vol_source = estimated_mask * vol_mix # Both are (B,T,F)

            # logging.info(f"DQN.forward LOOP source {i}: estimated_mask.shape: {estimated_mask.shape}")
            # logging.info(f"DQN.forward LOOP source {i}: ipd_mix.shape for masking: {ipd_mix.shape}")
            # logging.info(f"DQN.forward LOOP source {i}: masked_ipd_source.shape: {masked_ipd_source.shape}")

            source_features = torch.stack([masked_ipd_source, masked_ild_source, masked_vol_source], dim=-1) # Stack along new last dim -> (B,T,F,3)

            # Aggregate masked features to scalar per source
            # source_features is (B,T,F,3)
            # Take mean over Time and Frequency (dims 1 and 2) for each feature type
            per_type_means = source_features.mean(dim=(1, 2)) # Shape: (B,3)
            
            # logging.info(f"DQN.forward LOOP source {i}: per_type_means.shape: {per_type_means.shape}")

            # Append each feature type's mean for the current source
            features.append(per_type_means[:, 0].unsqueeze(1)) # IPD mean for current source (B,1)
            features.append(per_type_means[:, 1].unsqueeze(1)) # ILD mean for current source (B,1)
            features.append(per_type_means[:, 2].unsqueeze(1)) # VOL mean for current source (B,1)
        
        # 5. Padding if separator produced fewer sources than DQN processes
        processed_sources_count = min(actual_sources_count, self.num_sources)
        missing_features_for_padding = self.num_sources - processed_sources_count
        if missing_features_for_padding > 0:
            logging.debug(f"DQN.forward: Padding features for {missing_features_for_padding} source slots.")
            zero_feature_val = torch.zeros(batch_size, 1, 1, device=mix_audio.device)
            for _ in range(missing_features_for_padding):
                features.append(zero_feature_val.clone()) # IPD
                features.append(zero_feature_val.clone()) # ILD
                features.append(zero_feature_val.clone()) # VOL
        
        # 6. Concatenate all features and process through MLP
        # Each feature is (B, 1, 1), total 3*self.num_sources such features
        # logging.info(f"DQN.forward: features: {len(features)}")
        X = torch.cat(features, dim=1)  # Shape: (B, 3 * self.num_sources)
        # logging.info(f"DQN.forward: X (after cat features).shape: {X.shape}")
        # logging.info(f"DQN.forward: X.shape: {X.shape}")
        
        # Add agent info (e.g., position, orientation)
        # agent_info is (N_agent_features), e.g. (3). We need to expand it to (B, N_agent_features)
        # logging.info(f"DQN.forward: agent_info.shape: {agent_info.shape}")
        batch_size_X = X.shape[0]
        if agent_info.ndim == 1:
            # Case: agent_info is 1D (e.g., from choose_action)
            # Expand to (batch_size_X, N_agent_features)
            agent_info_to_cat = agent_info.unsqueeze(0).expand(batch_size_X, -1)
        elif agent_info.ndim == 2:
            # Case: agent_info is 2D (e.g., from update, already batched)
            if agent_info.shape[0] != batch_size_X:
                # This case should ideally not happen if data loading is correct
                # For now, let's try to accommodate if agent_info has B=1 and X has B>1
                if agent_info.shape[0] == 1 and batch_size_X > 1:
                    logging.warning(f"DQN.forward: agent_info batch size 1, X batch size {batch_size_X}. Expanding agent_info.")
                    agent_info_to_cat = agent_info.expand(batch_size_X, -1)
                else:
                    raise ValueError(f"Batch size mismatch: X has {batch_size_X} but agent_info has {agent_info.shape[0]}")
            else:
                agent_info_to_cat = agent_info
        else:
            raise ValueError(f"Unexpected agent_info ndim: {agent_info.ndim}. Shape: {agent_info.shape}")
        
        # logging.info(f"DQN.forward: agent_info_to_cat.shape: {agent_info_to_cat.shape}")
        X = torch.cat((X, agent_info_to_cat), dim=1)  # (B, 3 * self.num_sources + N_agent_features)
        # logging.info(f"DQN.forward: after agent info X.shape: {X.shape}")
        
        # Pass through fully connected layers
        X = self.prelu(self.fc(X))
        q_values = F.softmax(X, dim=1) # Softmax might be for policy gradient; for Q-values usually not applied here
        
        return q_values

    def flatten_features(self, x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features








