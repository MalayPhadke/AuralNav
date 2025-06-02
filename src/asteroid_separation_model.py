import torch
import torch.nn as nn
from asteroid.models import DPRNNTasNet
import logging
class AsteroidSeparationModel(nn.Module):
    """
    A wrapper for Asteroid's source separation models, e.g., DPRNNTasNet.
    This model takes a time-domain mixture and returns time-domain separated sources.
    """
    def __init__(self, num_sources, sample_rate=8000, **asteroid_model_kwargs):
        """
        Initializes the AsteroidSeparationModel.

        Args:
            num_sources (int): The number of sources to separate.
            sample_rate (int): The sample rate of the audio expected by the Asteroid model.
                                 DPRNNTasNet in Asteroid is often pretrained on 8kHz or 16kHz.
                                 Ensure this matches your data or the pretrained model's requirements.
            **asteroid_model_kwargs: Additional keyword arguments to pass to the Asteroid model constructor.
                                     For DPRNNTasNet, this could include 'n_filters', 'n_kernel', 'n_repeat',
                                     'n_layers', 'n_units', 'chunk_size', 'hop_size', etc.
                                     Example: AsteroidSeparationModel(num_sources=2, sample_rate=16000, n_filters=64)
        """
        super().__init__()
        self.num_sources = num_sources
        self.sample_rate = sample_rate

        # Default DPRNNTasNet parameters (can be overridden by asteroid_model_kwargs)
        # These are just illustrative; consult Asteroid documentation for optimal/default values.
        default_kwargs = {
            'n_in': 1, # Assuming mono input
            'n_src': num_sources,
            # Common DPRNNTasNet parameters - you might need to adjust these or use Asteroid's defaults
            'n_filters': 64, 
            'n_kernel': 16,  
            'n_repeat': 3,   
            'n_layers': 8,   
            'n_units': 512,  
            'chunk_size': 100, 
            'hop_size': 50,    
            'causal': False,
            # 'sample_rate': sample_rate # sample_rate is not always a direct arg for DPRNNTasNet constructor
        }
        # Merge default_kwargs with user-provided asteroid_model_kwargs
        # User-provided kwargs take precedence
        final_kwargs = {**default_kwargs, **asteroid_model_kwargs}

        # fb_name is often 'free' or 'stft' for filterbank type. Default to 'free'.
        if 'fb_name' not in final_kwargs: 
            final_kwargs['fb_name'] = 'free'

        # DPRNNTasNet with 'free' filterbank doesn't typically use sample_rate in its constructor.
        # If 'stft' filterbank is used, sample_rate might be needed for STFT params.
        # Remove 'sample_rate' from final_kwargs if it came from default_kwargs (i.e., not user-provided)
        # and fb_name is 'free', to avoid potential TypeError if the model doesn't accept it.
        if final_kwargs.get('fb_name') == 'free' and 'sample_rate' not in asteroid_model_kwargs and 'sample_rate' in final_kwargs:
            del final_kwargs['sample_rate']
        elif 'sample_rate' in default_kwargs and 'sample_rate' not in asteroid_model_kwargs and final_kwargs.get('fb_name') != 'free':
            # If sample_rate was in default_kwargs (which it is not anymore after this edit) 
            # and not explicitly by user, but fb_name is NOT 'free', we might still need it.
            # However, current default_kwargs above has sample_rate commented out, so this branch is less likely.
            # If user provides sample_rate via asteroid_model_kwargs, it will be in final_kwargs.
            pass 

        self.asteroid_model = DPRNNTasNet(**final_kwargs)

    def forward(self, mixture_audio_tensor):
        """
        Performs source separation on the input mixture.

        Args:
            mixture_audio_tensor (torch.Tensor): A tensor representing the mixture audio.
                                                Expected shape: (batch_size, num_channels, num_samples)
                                                where num_channels is typically 2 for stereo.

        Returns:
            dict: A dictionary containing:
                'estimated_sources' (torch.Tensor): Separated audio sources (mono).
                                                    Shape: (batch_size, num_sources, num_samples).
                'mix_audio' (torch.Tensor): The original input mixture_audio_tensor.
                                            Shape: (batch_size, num_channels, num_samples).
        """
        original_mix_audio = mixture_audio_tensor

        # Prepare mono input for the Asteroid model
        # DPRNNTasNet in Asteroid usually expects (batch, samples)
        if mixture_audio_tensor.ndim == 3:
            if mixture_audio_tensor.shape[1] == 1:
                # Input is mono but 3D (B, 1, N), squeeze channel dim
                mono_mixture_for_asteroid = mixture_audio_tensor.squeeze(1)
            elif mixture_audio_tensor.shape[1] > 1:
                # Input is stereo or multi-channel (B, C, N), mix down to mono
                mono_mixture_for_asteroid = torch.mean(mixture_audio_tensor, dim=1)
            else: # mixture_audio_tensor.shape[1] == 0, should not happen
                raise ValueError(f"Invalid number of channels: {mixture_audio_tensor.shape[1]}")
        elif mixture_audio_tensor.ndim == 2:
            # Input is already mono (B, N)
            mono_mixture_for_asteroid = mixture_audio_tensor
        else:
            raise ValueError(f"Unsupported mixture_audio_tensor ndim: {mixture_audio_tensor.ndim}")

        # Asteroid models typically return (batch_size, num_sources, num_samples)
        estimated_sources = self.asteroid_model(mono_mixture_for_asteroid)
        
        return {
            'estimated_sources': estimated_sources, 
            'mix_audio': original_mix_audio
        }

    def save(self, location):
        """
        Saves the model's state_dict.
        Args:
            location (str): Path to save the model.
        """
        torch.save(self.state_dict(), location)
        print(f"AsteroidSeparationModel saved to {location}")

    @classmethod
    def load(cls, location, num_sources, sample_rate=16000, **asteroid_model_kwargs):
        """
        Loads a model's state_dict.
        Args:
            location (str): Path to the saved model state_dict.
            num_sources (int): Number of sources the model was trained for.
            sample_rate (int): Sample rate for the model.
            **asteroid_model_kwargs: Additional arguments for model instantiation if needed.
        Returns:
            AsteroidSeparationModel: An instance of the model with loaded weights.
        """
        model = cls(num_sources=num_sources, sample_rate=sample_rate, **asteroid_model_kwargs)
        model.load_state_dict(torch.load(location, map_location=lambda storage, loc: storage))
        print(f"AsteroidSeparationModel loaded from {location}")
        return model

# if __name__ == '__main__':
#     # Example Usage (for testing this file directly)
#     batch_size = 2
#     num_samples = 16000 * 2 # 2 seconds of audio at 16kHz
#     n_src = 2
#     sr = 16000
# 
#     # Create a dummy mixture (mono)
#     dummy_mixture_mono = torch.randn(batch_size, num_samples)
#     
#     # Create a dummy mixture (stereo)
#     dummy_mixture_stereo = torch.randn(batch_size, 2, num_samples)
# 
#     # Instantiate the model
#     # You might need to install asteroid: pip install asteroid asteroid-filterbanks
#     try:
#         separator = AsteroidSeparationModel(num_sources=n_src, sample_rate=sr)
#         separator.eval() # Set to evaluation mode
# 
#         print(f"Testing with mono input ({dummy_mixture_mono.shape}):")
#         with torch.no_grad():
#             estimated_sources_mono = separator(dummy_mixture_mono)
#         print(f"Output shape (mono input): {estimated_sources_mono.shape}") # Expected: (batch, n_src, samples)
#         assert estimated_sources_mono.shape == (batch_size, n_src, num_samples)
# 
#         print(f"\nTesting with stereo input ({dummy_mixture_stereo.shape}):")
#         with torch.no_grad():
#             estimated_sources_stereo = separator(dummy_mixture_stereo)
#         print(f"Output shape (stereo input): {estimated_sources_stereo.shape}") # Expected: (batch, n_src, samples)
#         assert estimated_sources_stereo.shape == (batch_size, n_src, num_samples)
#         
#         print("\nAsteroidSeparationModel basic test passed.")
# 
#         # Test save and load
#         save_path = './temp_asteroid_model.pth'
#         separator.save(save_path)
#         loaded_separator = AsteroidSeparationModel.load(save_path, num_sources=n_src, sample_rate=sr)
#         loaded_separator.eval()
#         with torch.no_grad():
#             re_estimated_sources = loaded_separator(dummy_mixture_mono)
#         assert torch.allclose(estimated_sources_mono, re_estimated_sources, atol=1e-6)
#         print("Model save and load test passed.")
#         import os
#         os.remove(save_path)
# 
#     except ImportError:
#         print("Asteroid library not found. Please install it: pip install asteroid asteroid-filterbanks")
#     except Exception as e:
#         print(f"An error occurred during testing: {e}")
#         import traceback
#         traceback.print_exc()
