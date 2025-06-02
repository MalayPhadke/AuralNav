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
from nussl_separation_model import SeparationModel


class RnnAgent(agent.AgentBase):
    def __init__(self, env_config, dataset_config, rnn_config=None, stft_config=None, 
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

            rnn_config (dict):  Dictionary containing the parameters for the model
            stft_config (dict):  Dictionary containing the parameters for STFT
        """

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('DEVICE:', self.device)
        # Use default config if configs are not provided by user
        # Get number of sources from environment if not specified
        if num_sources is None and hasattr(env_config, 'get'):
            self.num_sources = env_config.get('num_sources', 2)
        else:
            self.num_sources = num_sources or 2
            
        if rnn_config is None:
            # Build default config with specified number of sources
            self.rnn_config = builders.build_recurrent_end_to_end(
                num_filters=512,
                filter_length=1024,
                hop_length=256,
                window_type='hann',
                hidden_size=300,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,
                num_sources=self.num_sources,  # Use dynamic source count
                mask_activation=['sigmoid'],
                num_audio_channels=1
            )
        else:
            # If rnn_config provided, update num_sources if needed
            if 'num_sources' not in rnn_config:
                rnn_config['num_sources'] = self.num_sources
            else:
                self.num_sources = rnn_config['num_sources']
                
            self.rnn_config = builders.build_recurrent_end_to_end(**rnn_config)

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

        # Initialize the rnn model with num_sources
        self.rnn_model = RnnSeparator(self.rnn_config, num_sources=self.num_sources).to(self.device)
        self.rnn_model_stable = RnnSeparator(self.rnn_config, num_sources=self.num_sources).to(self.device)

        # Load pretrained model
        if pretrained:
            model_dict = torch.load(constants.PRETRAIN_PATH)
            self.rnn_model.rnn_model.load_state_dict(model_dict)

        # Initialize dataset related parameters
        self.bs = dataset_config['batch_size']
        self.num_updates = dataset_config['num_updates']
        self.dynamic_dataset = RLDataset(buffer=self.dataset, sample_size=self.bs, sample_rate=self.sample_rate)
        self.dataloader = torch.utils.data.DataLoader(self.dynamic_dataset, batch_size=self.bs)

        # Initialize network layers for DQN network
        filter_length = (stft_config['num_filters'] // 2 + 1) * 2 if stft_config is not None else 514
        total_actions = self.env.action_space.n
        network_params = {
            'filter_length': filter_length, 
            'total_actions': total_actions, 
            'stft_diff': self.stft_diff,
            'num_sources': self.num_sources  # Pass num_sources to DQN
        }
        self.q_net = DQN(network_params).to(self.device)
        self.q_net_stable = DQN(network_params).to(self.device)  # Fixed Q net

        # Tell optimizer which parameters to learn
        if pretrained:
            # Freezing the rnn model weights, only optimizing Q-net
            params = self.q_net.parameters()
        else:
            params = list(self.rnn_model.parameters()) + list(self.q_net.parameters())
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
            logging.info(f"Not enough samples in the dataset. Current size: {len(self.dataset.items)}, batch size: {self.bs}")
            return

        for index, data in enumerate(self.dataloader):
            if index > self.num_updates:
                break
            logging.info(f"RnnAgent.update (loop index {index}): data['mix_audio_prev_state'].shape: {data['mix_audio_prev_state'].shape if 'mix_audio_prev_state' in data else 'N/A'}")
            logging.info(f"RnnAgent.update (loop index {index}): data['mix_audio_new_state'].shape: {data['mix_audio_new_state'].shape if 'mix_audio_new_state' in data else 'N/A'}")

            # Get the total number of time steps
            total_time_steps = data['mix_audio_prev_state'].shape[-1]

            # Reshape the mixture to pass through the separation model (Convert dual channels into one)
            # Also, rename the state to work on to mix_audio so that it can pass through remaining nussl architecture
            # Move to GPU
            data['mix_audio'] = data['mix_audio_prev_state'].float().view(
                -1, 1, total_time_steps).to(self.device)
            logging.info(f"RnnAgent.update: current_state data['mix_audio'].shape: {data['mix_audio'].shape}")
            data['action'] = data['action'].to(self.device)
            data['reward'] = data['reward'].to(self.device)
            agent_info = data['agent_info'].to(self.device)
            logging.info(f"RnnAgent.update: agent_info.shape: {agent_info.shape}")

            # Get the separated sources by running through RNN separation model
            output = self.rnn_model(data)
            logging.info(f"RnnAgent.update: output from rnn_model: keys: {output.keys() if isinstance(output, dict) else 'N/A'}, estimated_sources shape: {output.get('estimated_sources').shape if isinstance(output, dict) and output.get('estimated_sources') is not None else 'N/A'}")
            output['mix_audio'] = data['mix_audio']

            # Pass then through the DQN model to get q-values
            q_values = self.q_net(output, agent_info, total_time_steps)
            logging.info(f"RnnAgent.update: q_values from self.q_net.shape: {q_values.shape}")
            logging.info(f"RnnAgent.update: q_values before gather.shape: {q_values.shape}")
            q_values = q_values.gather(1, data['action'])
            logging.info(f"RnnAgent.update: q_values after gather.shape: {q_values.shape}")

            with torch.no_grad():
                # Now, get q-values for the next-state
                # Get the total number of time steps
                total_time_steps = data['mix_audio_new_state'].shape[-1]

                # Reshape the mixture to pass through the separation model (Convert dual channels into one)
                data['mix_audio'] = data['mix_audio_new_state'].float().view(
                    -1, 1, total_time_steps).to(self.device)
                logging.info(f"RnnAgent.update: next_state data['mix_audio'].shape: {data['mix_audio'].shape}")
                stable_output = self.rnn_model_stable(data)
                logging.info(f"RnnAgent.update: stable_output from rnn_model_stable: keys: {stable_output.keys() if isinstance(stable_output, dict) else 'N/A'}, estimated_sources shape: {stable_output.get('estimated_sources').shape if isinstance(stable_output, dict) and stable_output.get('estimated_sources') is not None else 'N/A'}")
                stable_output['mix_audio'] = data['mix_audio']
                next_q_values = self.q_net_stable(stable_output, agent_info, total_time_steps)
                logging.info(f"RnnAgent.update: next_q_values.shape: {next_q_values.shape}")
                next_q_values_max = next_q_values.max(1)[0].unsqueeze(-1)

            expected_q_values = data['reward'] + self.gamma * next_q_values_max

            logging.info(f"RnnAgent.update: expected_q_values.shape: {expected_q_values.shape}")

            # Calculate loss
            loss = F.l1_loss(q_values, expected_q_values)
            self.losses.append(loss)
            self.writer.add_scalar('Loss/train', loss, len(self.losses))

            # Optimize the model with backprop
            self.optimizer.zero_grad()
            loss.backward()

            # Applying AutoClip
            self.grad_norms = utils.autoclip(self.rnn_model, self.percentile, self.grad_norms)

            # Stepping optimizer
            self.optimizer.step()

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
            logging.info(f"RnnAgent.choose_action: data['mix_audio_new_state'].shape: {data['mix_audio_new_state'].shape if 'mix_audio_new_state' in data else 'N/A'}")
            logging.info(f"RnnAgent.choose_action: data.keys(): {data.keys()}")
            # Perform the forward pass (RNN separator => DQN)
            total_time_steps = data['mix_audio_new_state'].shape[-1]
            data['mix_audio'] = data['mix_audio_new_state'].float().view(
                -1, 1, total_time_steps).to(self.device)
            logging.info(f"RnnAgent.choose_action: next_state data['mix_audio'].shape: {data['mix_audio'].shape}")
            output = self.rnn_model(data)
            logging.info(f"RnnAgent.choose_action: output from rnn_model: keys: {output.keys() if isinstance(output, dict) else 'N/A'}, estimated_sources shape: {output.get('estimated_sources').shape if isinstance(output, dict) and output.get('estimated_sources') is not None else 'N/A'}")
            output['mix_audio'] = data['mix_audio']
            agent_info = data['agent_info'].to(self.device)
            logging.info(f"RnnAgent.choose_action: agent_info.shape: {agent_info.shape}")

            # action = argmax(q-values)
            q_values = self.q_net(output, agent_info, total_time_steps)
            logging.info(f"RnnAgent.choose_action: q_values.shape: {q_values.shape}")
            action = q_values.max(1)[1].unsqueeze(-1)
            action = action[0].item()

            action = int(action)
            logging.info(f"RnnAgent.choose_action: chosen action: {action}")
            return action

    def update_stable_networks(self):
        self.rnn_model_stable.load_state_dict(self.rnn_model.state_dict())
        self.q_net_stable.load_state_dict(self.q_net.state_dict())

    def save_model(self, name):
        """
        Args:
            name (str): Name contains the episode information (To give saved models unique names)
        """
        # Save the parameters for rnn model and q net separately
        metadata = {
            'sample_rate': 8000
        }
        self.rnn_model.rnn_model.save(os.path.join(self.SAVE_PATH, 'sp_' + name), metadata)
        torch.save(self.rnn_model.state_dict(), os.path.join(self.SAVE_PATH, 'rnn_' + name))
        torch.save(self.q_net.state_dict(), os.path.join(self.SAVE_PATH, 'qnet_' + name))


class RnnSeparator(nn.Module):
    def __init__(self, rnn_config, verbose=False, num_sources=None):
        super(RnnSeparator, self).__init__()
        # Store num_sources for reference
        self.num_sources = num_sources
        # Initialize SeparationModel with num_sources
        self.rnn_model = SeparationModel(rnn_config, verbose=verbose, num_sources=num_sources)

    def forward(self, x):
        logging.info(f"RnnSeparator.forward: input data keys: {x.keys() if isinstance(x, dict) else 'N/A'}, input mix_audio shape: {x.get('mix_audio').shape if isinstance(x, dict) and x.get('mix_audio') is not None else 'N/A'}")
        result = self.rnn_model(x)
        logging.info(f"RnnSeparator.forward: output result keys: {result.keys() if isinstance(result, dict) else 'N/A'}, estimated_sources shape: {result.get('estimated_sources').shape if isinstance(result, dict) and result.get('estimated_sources') is not None else 'N/A'}")
        logging.info(f"RnnSeparator.forward: output result shape: {result['audio'].shape}, mask shape: {result['mask'].shape}")
        return result

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

    def forward(self, output, agent_info, total_time_steps):
        logging.info(f"DQN.forward: input output['mix_audio'].shape: {output.get('mix_audio').shape if output.get('mix_audio') is not None else 'N/A'}")
        logging.info(f"DQN.forward: input output['mask'].shape: {output.get('mask').shape if output.get('mask') is not None else 'N/A'}")
        logging.info(f"DQN.forward: input agent_info.shape: {agent_info.shape}")
        # Reshape the output again to get dual channels
        # Perform short time fourier transform of this output
        _, _, nt = output['mix_audio'].shape
        output['mix_audio'] = output['mix_audio'].reshape(-1, 2, nt)
        stft_data = self.stft_diff(output['mix_audio'], direction='transform')
        logging.info(f"DQN.forward: stft_data.shape: {stft_data.shape}")

        # Get the IPD and ILD features from the stft data
        _, nt, nf, _, ns = output['mask'].shape
        
        # Check if the number of sources in output matches expected number
        actual_sources = ns
        if actual_sources != self.num_sources:
            print(f"Warning: Mask has {actual_sources} sources but DQN expects {self.num_sources}")
        
        # Process audio features for each source
        output['mask'] = output['mask'].view(-1, 2, nt, nf, ns)
        # Take max across channels (batch_size, nt, nf, ns)
        output['mask'] = output['mask'].max(dim=1)[0]
        
        # Extract audio features
        ipd, ild, vol = audio_processing.ipd_ild_features(stft_data)
        logging.info(f"DQN.forward: ipd.shape: {ipd.shape}, ild.shape: {ild.shape}, vol.shape: {vol.shape}")
        ipd = ipd.unsqueeze(-1)
        ild = ild.unsqueeze(-1)
        vol = vol.unsqueeze(-1)
        
        # Initialize feature list
        features = []
        
        # Process each source separately
        for source_idx in range(min(ns, self.num_sources)):
            # Get source-specific mask
            source_mask = output['mask'][..., source_idx:source_idx+1]
            logging.info(f"DQN.forward loop (source {source_idx}): source_mask.shape: {source_mask.shape}")
            
            # Calculate features for this source
            ipd_mean = (source_mask * ipd).mean(dim=[1, 2]).unsqueeze(1)
            ild_mean = (source_mask * ild).mean(dim=[1, 2]).unsqueeze(1)
            vol_mean = (source_mask * vol).mean(dim=[1, 2]).unsqueeze(1)
            logging.info(f"DQN.forward loop (source {source_idx}): ipd_mean.shape: {ipd_mean.shape}, ild_mean.shape: {ild_mean.shape}, vol_mean.shape: {vol_mean.shape}")
            
            # Add features to list
            features.append(ipd_mean)
            features.append(ild_mean)
            features.append(vol_mean)
        
        # If we have fewer sources than expected, pad with zeros
        missing_sources = self.num_sources - ns
        if missing_sources > 0:
            batch_size = output['mask'].shape[0]
            zero_feature = torch.zeros(batch_size, 1, 1).to(ipd.device)
            for _ in range(missing_sources):
                features.append(zero_feature.clone())  # IPD
                features.append(zero_feature.clone())  # ILD
                features.append(zero_feature.clone())  # VOL
        
        logging.info(f"DQN.forward: features: {len(features)}")
        # Concatenate all features
        X = torch.cat(features, dim=1)
        logging.info(f"DQN.forward: X.shape {X.shape}")
        X = X.reshape(X.shape[0], -1)
        logging.info(f"DQN.forward: X.shape: {X.shape}")
        
        # Add agent info
        logging.info(f"DQN.forward: agent_info.shape: {agent_info.shape}")
        agent_info = agent_info.view(-1, 3)
        logging.info(f"DQN.forward: agent_info.shape: {agent_info.shape}")
        X = torch.cat((X, agent_info), dim=1)
        logging.info(f"DQN.forward: X.shape: {X.shape}")
        # Process through network
        X = self.prelu(self.fc(X))
        q_values = F.softmax(X, dim=1)
        logging.debug(f"DQN.forward: final q_values.shape: {q_values.shape}")
        
        return q_values

    def flatten_features(self, x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features








