import os

# source data
DIR_MALE = "./sounds/dry_recordings/dev/051/"
DIR_FEMALE = "./sounds/dry_recordings/dev/050/"
DIR_CAR = './sounds/car/'
DIR_PHONE = './sounds/siren/'
AUDIO_EXTENSION = ".wav"

# saved data during experiment
DATA_PATH = "./data"
DIR_PREV_STATES = os.path.join(DATA_PATH, 'prev_states/')
DIR_NEW_STATES = os.path.join(DATA_PATH, 'new_states/')
DIR_DATASET_ITEMS = os.path.join(DATA_PATH, 'dataset_items/')
MODEL_SAVE_PATH = './models/'
DIST_URL = "init_dist_to_target.p"
STEPS_URL = "steps_to_completion.p"
REWARD_URL = "rewards_per_episode.p"
PRETRAIN_PATH = './models/pretrained.pth'

# audio stuff
RESAMPLE_RATE = 8000

# env stuff
DIST_BTWN_EARS = 0.15

# max and min values of exploration rate
MAX_EPSILON = 0.9
MIN_EPSILON = 0.1

# reward structure (keep as floats)
STEP_PENALTY = -0.5
TURN_OFF_REWARD = 100.0
ORIENT_PENALTY = -0.1
COMPLETED_PROGRESS_REWARD = 100.0  # Bonus for making progress with multiple sources

# dataset
MAX_BUFFER_ITEMS = 10000

"""
A repository containing all of the constants frequently used in
this wacky, mixed up source separation stuff.
"""
import os
from collections import OrderedDict
from six.moves.urllib_parse import urljoin

import scipy.signal
from scipy.signal.windows import hamming, hann, blackman

__all__ = ['DEFAULT_SAMPLE_RATE', 'DEFAULT_WIN_LEN_PARAM', 'DEFAULT_BIT_DEPTH',
           'DEFAULT_MAX_VAL', 'EPSILON', 'MAX_FREQUENCY',
           'WINDOW_HAMMING', 'WINDOW_RECTANGULAR', 'WINDOW_HANN',
           'WINDOW_BLACKMAN', 'WINDOW_TRIANGULAR', 'WINDOW_DEFAULT',
           'ALL_WINDOWS', 'NUMPY_JSON_KEY', 'LEN_INDEX', 'CHAN_INDEX',
           'STFT_VERT_INDEX', 'STFT_LEN_INDEX', 'STFT_CHAN_INDEX',
           'LEVEL_MAX', 'LEVEL_MIN']

DEFAULT_SAMPLE_RATE = 44100  #: (int): Default sample rate. 44.1 kHz, CD-quality
DEFAULT_WIN_LEN_PARAM = 0.032  #: (float): Default window length. 32ms
DEFAULT_WIN_LENGTH = 2048  #: (int): Default window length, 2048 samples.
DEFAULT_BIT_DEPTH = 16  #: (int): Default bit depth. 16-bits, CD-quality
DEFAULT_MAX_VAL = 2 ** 16  #: (int): Max value of 16-bit audio file (unsigned)
EPSILON = 1e-16  #: (float): epsilon for determining small values
MAX_FREQUENCY = DEFAULT_SAMPLE_RATE // 2  #: (int): Maximum frequency representable. 22050 Hz

WINDOW_HAMMING = hamming.__name__  #: (str): Name for calling Hamming window. 'hamming'
WINDOW_RECTANGULAR = 'rectangular'  #: (str): Name for calling Rectangular window. 'rectangular'
WINDOW_HANN = hann.__name__  #: (str): Name for calling Hann window. 'hann'
WINDOW_BLACKMAN = blackman.__name__  #: (str): Name for calling Blackman window. 'blackman'
WINDOW_TRIANGULAR = 'triang'  #: (str): Name for calling Triangular window. 'triangular'
WINDOW_SQRT_HANN = 'sqrt_hann'  #: (str): Name for calling square root of hann window. 'sqrt_hann'.

WINDOW_DEFAULT = WINDOW_SQRT_HANN  #: (str): Default window, Hamming.
ALL_WINDOWS = [
    WINDOW_HAMMING, WINDOW_RECTANGULAR, WINDOW_HANN, WINDOW_BLACKMAN, 
    WINDOW_TRIANGULAR, WINDOW_SQRT_HANN]
"""list(str): list of all available windows in *nussl*
"""

NUMPY_JSON_KEY = "py/numpy.ndarray"  #: (str): key used when turning numpy arrays into json

BINARY_MASK = 'binary'
""" String alias for setting this object to return :class:`separation.masks.binary_mask.BinaryMask` objects
"""

SOFT_MASK = 'soft'
""" String alias for setting this object to return :class:`separation.masks.soft_mask.SoftMask` objects
"""

# ############# Array Indices ############# #

# audio_data
LEN_INDEX = 1  #: (int): Index of the number of samples in an audio signal. Used in :ref:`audio_signal`
CHAN_INDEX = 0  #: (int): Index of the number of channels in an audio signal. Used in :ref:`audio_signal`

# stft_data
STFT_VERT_INDEX = 0
"""
(int) Index of the number of frequency (vertical) values in a time-frequency representation. 
Used in :ref:`audio_signal` and in :ref:`mask_base`.
"""
STFT_LEN_INDEX = 1
"""
(int) Index of the number of time (horizontal) hops in a time-frequency representation. 
Used in :ref:`audio_signal` and in :ref:`mask_base`.
"""
STFT_CHAN_INDEX = 2
"""
(int) Index of the number of channels in a time-frequency representation. 
Used in :ref:`audio_signal` and in :ref:`mask_base`.
"""
USE_LIBROSA_STFT = False  #: (bool): Whether *nussl* will use librosa's stft function by default


# ############# MUSDB interface ############### #
STEM_TARGET_DICT = OrderedDict([
    ('drums', 'drums'), 
    ('bass', 'bass'), 
    ('other', 'other'),
    ('vocals', 'vocals'), 
    ('accompaniment', 'bass+drums+other'), 
    ('linear_mixture', 'vocals+bass+drums+other')
])

# ################## Effects ################## #
# These values are found in the ffmpeg documentation for filters
# that use the level_in argument:
LEVEL_MIN = .015625
LEVEL_MAX = 64
