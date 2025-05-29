import gymnasium as gym #using gymnasium instead of gym
from pyroomacoustics import MicrophoneArray, ShoeBox, Room, linear_2D_array, Constants
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from copy import deepcopy
from audio_signal import AudioSignal ## original AudioSignal class from nussl
import sys
sys.path.append("../../")
import constants
from utils import choose_random_files
import os

# Suppress 'c' argument errors caused by room.plot()
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

import pygame

# Pygame constants
PYGAME_SCREEN_WIDTH = 800
PYGAME_SCREEN_HEIGHT = 600
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
AGENT_COLOR = BLUE
SOURCE_COLOR = RED


class AudioEnv(gym.Env):
    def __init__(
        self,
        room_config,
        source_folders_dict,
        resample_rate=8000,
        num_channels=2,
        bytes_per_sample=2,
        corners=False,
        absorption=0.0,
        max_order=2,
        step_size=1,
        acceptable_radius=.5,
        num_sources=2,
        max_detectable_sources=None,
        degrees=np.deg2rad(30),
        reset_sources=True,
        same_config=False,
        visualize_pygame=False,
        play_audio_on_step=True
    ):
        """
        This class inherits from OpenAI Gymnasium Env and is used to simulate the agent moving in PyRoom.

        Args:
            room_config (List or np.array): dimensions of the room. For Shoebox, in the form of [10,10]. Otherwise,
                in the form of [[1,1], [1, 4], [4, 4], [4, 1]] specifying the corners of the room
            resample_rate (int): sample rate in Hz
            num_channels (int): number of channels (used in playing what the mic hears)
            bytes_per_sample (int): used in playing what the mic hears
            corners (bool): False if using Shoebox config, otherwise True
            absorption (float): Absorption param of the room (how walls absorb sound)
            max_order (int): another room parameter
            step_size (float): specified step size else we programmatically assign it
            acceptable_radius (float): source is considered found/turned off if agent is within this distance of src
            num_sources (int): the number of audio sources the agent will listen to
            max_detectable_sources (int): the maximum number of sources the agent can detect
            degrees (float): value of degrees to rotate in radians (.2618 radians = 15 degrees)
            reset_sources (bool): True if you want to choose different sources when resetting env
            source_folders_dict (Dict[str, int]): specify how many source files to choose from each folder
                e.g.
                    {
                        'car_horn_source_folder': 1,
                        'phone_ringing_source_folder': 1
                    }
            same_config (bool): If set to true, difficulty of env becomes easier - agent initial loc, source placement,
                                and audio files don't change over episodes
        """
        self.resample_rate = resample_rate
        self.absorption = absorption
        self.max_order = max_order
        self.audio = []
        self.num_channels = num_channels
        self.bytes_per_sample = bytes_per_sample
        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)
        self.action_to_string = {
            0: "Forward",
            1: "Backward",
            2: "Rotate Right",
            3: "Rotate Left",
        }
        self.corners = corners
        self.room_config = room_config
        self.acceptable_radius = acceptable_radius
        self.step_size = step_size
        self.num_sources = num_sources
        self.max_detectable_sources = max_detectable_sources
        self.source_locs = None
        self.min_size_audio = np.inf
        self.degrees = degrees
        self.cur_angle = 0
        self.cur_step = 0
        self.dim = len(self.room_config)
        # Define observation space
        # Using large dimensions for audio that will be updated after loading audio
        self.observation_space = spaces.Dict({
            'audio': spaces.Box(low=-np.inf, high=np.inf, shape=(100000, self.num_channels), dtype=np.float32),
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        self.reset_sources = reset_sources
        self.source_folders_dict = source_folders_dict
        self.same_config = same_config
        self.visualize_pygame = visualize_pygame
        self.play_audio_on_step = play_audio_on_step
        print(f"DEBUG: AudioEnv __init__: play_audio_on_step={self.play_audio_on_step}, visualize_pygame={self.visualize_pygame}")
        self.pygame_screen = None
        self.pygame_clock = None
        self.world_to_pixel_scale = 1.0
        # For coordinate transformation
        self.world_min_x = 0
        self.world_min_y = 0
        self.room_width_world = 0
        self.room_height_world = 0
        self.screen_map_offset_x = 0
        self.screen_map_offset_y = 0

        if self.visualize_pygame or self.play_audio_on_step:
            fs = self.resample_rate
            num_channels = self.num_channels
            print(f"DEBUG: AudioEnv __init__: Target mixer settings: fs={fs}, channels={num_channels}")
            os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'
            try:
                # Set parameters for the mixer
                pygame.mixer.pre_init(frequency=fs, channels=num_channels, buffer=512)
                print("DEBUG: AudioEnv __init__: pygame.mixer.pre_init called.")

                # Initialize all Pygame modules (including attempting to init mixer with pre_init settings)
                pygame.init()
                if pygame.get_init():
                    print("DEBUG: AudioEnv __init__: pygame.init() successful.")
                else:
                    print("CRITICAL WARN: AudioEnv __init__: pygame.init() FAILED. Audio/Visuals will not work.")

                # Check if mixer was initialized by pygame.init()
                if pygame.mixer.get_init():
                    print("DEBUG: AudioEnv __init__: pygame.mixer initialized successfully (likely by pygame.init() using pre_init settings).")
                else:
                    print("WARN: AudioEnv __init__: pygame.mixer was NOT initialized by pygame.init(). Attempting explicit pygame.mixer.init().")
                    try:
                        # Attempt to initialize the mixer explicitly.
                        # It should use the parameters from pre_init.
                        pygame.mixer.init()
                        if pygame.mixer.get_init():
                            print("DEBUG: AudioEnv __init__: pygame.mixer initialized successfully after explicit call to pygame.mixer.init().")
                        else:
                            # This case should ideally not be reached if pygame.mixer.init() didn't raise an error but still failed.
                            print("CRITICAL WARN: AudioEnv __init__: pygame.mixer FAILED to initialize even after explicit call to pygame.mixer.init(), though no error was raised.")
                            self.play_audio_on_step = False # Disable audio
                    except pygame.error as e_mixer_init:
                        print(f"CRITICAL WARN: AudioEnv __init__: Explicit pygame.mixer.init() raised an error: {e_mixer_init}")
                        if "No such audio device" in str(e_mixer_init) or "dsp" in str(e_mixer_init):
                            print("INFO: This 'No such audio device' error usually means Pygame cannot find a working audio output on your system.")
                            print("INFO: Please check your system's audio configuration, ensure speakers/headphones are connected and selected as output, and consider installing audio libraries like libasound2-dev or libpulse-dev.")
                            print("INFO: You can also try setting SDL_AUDIODRIVER (e.g., 'export SDL_AUDIODRIVER=alsa') before running the script.")
                            print("INFO: Audio playback will be disabled for this session.")
                        self.play_audio_on_step = False # Disable audio
            
            except pygame.error as e_outer:
                # This catches errors from pre_init or the main pygame.init()
                print(f"CRITICAL WARN: audio_env.py: Error during Pygame/Mixer initialization (pre_init or pygame.init stage): {e_outer}. Audio playback will likely be disabled.")
                self.play_audio_on_step = False # Disable audio if outer setup fails
                # Fallback if pygame.init() in the try block failed
                if not pygame.get_init():
                    print("DEBUG: AudioEnv __init__: Attempting pygame.init() again in except block as a last resort (for visualization perhaps).")
                    pygame.init()
                    if pygame.get_init():
                         print("DEBUG: AudioEnv __init__: pygame.init() in except block was successful.")
                    else:
                         print("CRITICAL WARN: AudioEnv __init__: pygame.init() in except block FAILED.")

            if self.visualize_pygame:
                self.pygame_screen = pygame.display.set_mode((PYGAME_SCREEN_WIDTH, PYGAME_SCREEN_HEIGHT))
                pygame.display.set_caption("Audio Environment")
                self.pygame_clock = pygame.time.Clock()

            if not self.corners:  # Shoebox room_config = [width, height]
                self.room_width_world = self.room_config[0]
                self.room_height_world = self.room_config[1]
                self.world_min_x, self.world_min_y = 0, 0
            else:  # Polygonal room_config = [[x1,y1], [x2,y2]...]
                corners_array = np.array(self.room_config)
                self.world_min_x = np.min(corners_array[:, 0])
                world_max_x = np.max(corners_array[:, 0])
                self.world_min_y = np.min(corners_array[:, 1])
                world_max_y = np.max(corners_array[:, 1])
                self.room_width_world = world_max_x - self.world_min_x
                self.room_height_world = world_max_y - self.world_min_y
            
            if self.room_width_world <= 0 or self.room_height_world <= 0: # Check for non-positive dimensions
                self.world_to_pixel_scale = 50.0 # Default scale e.g. 50 pixels per meter if dimensions are problematic
            else:
                scale_x = PYGAME_SCREEN_WIDTH / self.room_width_world
                scale_y = PYGAME_SCREEN_HEIGHT / self.room_height_world
                self.world_to_pixel_scale = min(scale_x, scale_y) * 0.9 # Use 90% of min scale to have some padding

            self.screen_map_offset_x = (PYGAME_SCREEN_WIDTH - self.room_width_world * self.world_to_pixel_scale) / 2
            self.screen_map_offset_y = (PYGAME_SCREEN_HEIGHT - self.room_height_world * self.world_to_pixel_scale) / 2

        # choose audio files as sources
        self.direct_sources = choose_random_files(self.source_folders_dict)
        self.direct_sources_copy = deepcopy(self.direct_sources)

        # create the room and add sources
        self._create_room()
        self._add_sources()
        self._move_agent(new_agent_loc=None, initial_placing=True)

        # If same_config = True, keep track of the generated locations
        self.fixed_source_locs = self.source_locs.copy()
        self.fixed_agent_locs = self.agent_loc.copy()

        # The step size must be smaller than radius in order to make sure we don't
        # overstep a audio source
        if self.acceptable_radius < self.step_size / 2:
            raise ValueError(
                """The threshold radius (acceptable_radius) must be at least step_size / 2. Else, the agent may overstep 
                an audio source."""
            )

    def _world_to_pixel(self, world_x, world_y):
        """Converts world coordinates to Pygame pixel coordinates.
        Assumes world Y increases upwards, Pygame Y increases downwards.
        (self.world_min_x, self.world_min_y) is the reference bottom-left of the world area.
        """
        if not self.visualize_pygame:
            return 0, 0
        
        # Transform x: self.world_min_x maps to self.screen_map_offset_x on screen
        pixel_x = self.screen_map_offset_x + (world_x - self.world_min_x) * self.world_to_pixel_scale
        
        # Transform y: 
        # self.world_min_y (bottom of world) maps to self.screen_map_offset_y + self.room_height_world * self.world_to_pixel_scale (bottom on screen)
        # self.world_min_y + self.room_height_world (top of world) maps to self.screen_map_offset_y (top on screen)
        pixel_y = self.screen_map_offset_y + ( (self.world_min_y + self.room_height_world) - world_y) * self.world_to_pixel_scale
        
        return int(pixel_x), int(pixel_y)

    def _create_room(self):
        """
        This function creates the Pyroomacoustics room with our environment class variables.
        """
        # non-Shoebox config (corners of room are given)
        if self.corners:
            self.room = Room.from_corners(
                self.room_config, fs=self.resample_rate,
                absorption=self.absorption, max_order=self.max_order
            )

            # The x_max and y_max in this case would be used to generate agent's location randomly
            self.x_min = min(self.room_config[0])
            self.y_min = min(self.room_config[1])
            self.x_max = max(self.room_config[0])
            self.y_max = max(self.room_config[1])

        # ShoeBox config
        else:
            self.room = ShoeBox(
                self.room_config, fs=self.resample_rate,
                absorption=self.absorption, max_order=self.max_order
            )
            self.x_max = self.room_config[0]
            self.y_max = self.room_config[1]
            self.x_min, self.y_min = 0, 0
            
        
    def _validate_room_config(self):
        """
        Validates the current room configuration to ensure it will work with PyRoomAcoustics.
        Checks for:
        1. Valid source locations (within room boundaries)
        2. Valid agent location (within room boundaries)
        3. Non-zero audio data for sources
        4. Sufficient distance between sources and agent
        
        Returns:
            bool: True if the configuration is valid, False otherwise
        """
        # Check if source locations are within room boundaries
        if hasattr(self, 'source_locs') and self.source_locs:
            for i, loc in enumerate(self.source_locs):
                if not (self.x_min <= loc[0] <= self.x_max and self.y_min <= loc[1] <= self.y_max):
                    print(f"Source {i} at {loc} is outside room boundaries: [{self.x_min}, {self.x_max}] x [{self.y_min}, {self.y_max}]")
                    return False
        else:
            # No sources defined yet
            return True
            
        # Check if agent location is within room boundaries
        if hasattr(self, 'agent_loc') and self.agent_loc is not None:
            if not (self.x_min <= self.agent_loc[0] <= self.x_max and self.y_min <= self.agent_loc[1] <= self.y_max):
                print(f"Agent at {self.agent_loc} is outside room boundaries: [{self.x_min}, {self.x_max}] x [{self.y_min}, {self.y_max}]")
                return False
                
        # Check if audio data is non-zero
        if hasattr(self, 'audio') and self.audio:
            for i, audio in enumerate(self.audio):
                if len(audio) == 0:
                    print(f"Source {i} has zero-length audio data")
                    return False
                    
        # Check minimum distance between sources and agent (to avoid numerical issues)
        if hasattr(self, 'agent_loc') and self.agent_loc is not None and hasattr(self, 'source_locs') and self.source_locs:
            min_distance = 0.1  # Minimum distance in meters
            for i, loc in enumerate(self.source_locs):
                distance = np.linalg.norm(np.array(loc) - np.array(self.agent_loc))
                if distance < min_distance:
                    print(f"Source {i} at {loc} is too close to agent at {self.agent_loc} (distance: {distance}m)")
                    return False
                    
        return True

    def _move_agent(self, new_agent_loc, initial_placing=False):
        """
        This function moves the agent to a new location (given by new_agent_loc). It effectively removes the
        agent (mic array) from the room and then adds it back in the new location.

        If initial_placing == True, the agent is placed in the room for the first time.

        Args:
            new_agent_loc (List[int] or np.array or None): [x,y] coordinates of the agent's new location.
            initial_placing (bool): True if initially placing the agent in the room at the beginning of the episode
        """
        # Placing agent in room for the first time (likely at the beginning of a new episode, after a reset)
        if initial_placing:
            if new_agent_loc is None:
                loc = self._sample_points(1, sources=False, agent=True)
                print("Placing agent at {}".format(loc))
                self.agent_loc = loc
                self.cur_angle = 0  # Reset the orientation of agent back to zero at start of an ep 
            else:
                self.agent_loc = new_agent_loc.copy()
                print("Placing agent at {}".format(self.agent_loc))
                self.cur_angle = 0
        else:
            # Set the new agent location (where to move)
            self.agent_loc = new_agent_loc
        
        # Setup microphone in agent location, delete the array at previous time step
        self.room.mic_array = None

        if self.num_channels == 2:
            # Create the array at current time step (2 mics, angle IN RADIANS, 0.2m apart)
            mic = MicrophoneArray(
                linear_2D_array(self.agent_loc, 2, self.cur_angle,
                                constants.DIST_BTWN_EARS), self.room.fs
            )
            self.room.add_microphone_array(mic)
        else:
            mic = MicrophoneArray(self.agent_loc.reshape(-1, 1), self.room.fs)
            self.room.add_microphone_array(mic)

    def _sample_points(self, num_points, sources=True, agent=False):
        """
        This function generates randomly sampled points for the sources (or agent) to be placed

        Args:
            num_points (int): Number of [x, y] random points to generate
            sources (bool): True if generating points for sources (agent must be False)
            agent(bool): True if generating points for agent (sources must be False)

        Returns:
            sample_points (List[List[int]]): A list of [x,y] points for source location
            or
            random_point (List[int]): An [x, y] point for agent location
        """
        assert(sources != agent)
        sampled_points = []

        if sources:
            angles = np.arange(0, 2 * np.pi, self.degrees).tolist()
            while len(sampled_points) < num_points:
                chosen_angles = np.random.choice(angles, num_points)
                for angle in chosen_angles:
                    direction = np.random.choice([-1, 1])
                    distance = np.random.uniform(2 * self.step_size, 5 * self.step_size)
                    x = (self.x_min + self.x_max) / 2
                    y = (self.y_min + self.y_max) / 2
                    x = x + direction * np.cos(angle) * distance
                    y = y + direction + np.sin(angle) * distance
                    point = [x, y]
                    if self.room.is_inside(point):
                        accepted = True
                        if len(sampled_points) > 0:
                            dist_to_existing = euclidean_distances(
                                np.array(point).reshape(1, -1), sampled_points)
                            accepted = dist_to_existing.min() > 2 * self.step_size
                        if accepted and len(sampled_points) < num_points:
                            sampled_points.append(point)
            return sampled_points
        elif agent:
            accepted = False
            while not accepted:
                accepted = True
                point = [
                    np.random.uniform(self.x_min, self.x_max),
                    np.random.uniform(self.y_min, self.y_max),
                ]

                # ensure agent doesn't spawn too close to sources
                for source_loc in self.source_locs:
                    if(euclidean(point, source_loc) < 2.0 * self.acceptable_radius):
                        accepted = False 
            
            return point

    def _add_sources(self, new_source_locs=None, reset_env=False, removing_source=None):
        """
        This function adds the sources to the environment.

        Args:
            new_source_locs (List[List[int]]): A list consisting of [x, y] coordinates if the programmer wants
                to manually set the new source locations
            reset_env (bool): Bool indicating whether we reset_env the agents position to be the mean
                of all the sources
            removing_source (None or int): Value that will tell us if we are removing a source
                from sources
        """
        # Can reset with NEW, randomly sampled sources (typically at the start of a new episode)
        if self.reset_sources:
            self.direct_sources = choose_random_files(self.source_folders_dict)
        else:
            self.direct_sources = deepcopy(self.direct_sources_copy)

        if new_source_locs is None:
            self.source_locs = self._sample_points(num_points=self.num_sources)
        else:
            self.source_locs = new_source_locs.copy()

        print("Source locs {}".format(self.source_locs))

        self.audio = []
        self.min_size_audio = np.inf
        for idx, audio_file in enumerate(self.direct_sources):
            # Audio will be automatically re-sampled to the given rate (default sr=8000).
            a = AudioSignal(audio_file, sample_rate=self.resample_rate)
            a.to_mono()

            # normalize audio so both sources have similar volume at beginning before mixing
            loudness = a.loudness()

            # # mix to reference db
            ref_db = -40
            db_diff = ref_db - loudness
            gain = 10 ** (db_diff / 20)
            a = a * gain

            # Find min sized source to ensure something is playing at all times
            if len(a) < self.min_size_audio:
                self.min_size_audio = len(a)
            self.audio.append(a.audio_data.squeeze())

        # Update observation space with correct audio dimensions
        self.observation_space = spaces.Dict({
            'audio': spaces.Box(low=-np.inf, high=np.inf, shape=(self.min_size_audio, self.num_channels), dtype=np.float32),
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        
        # add sources using audio data
        for idx, audio in enumerate(self.audio):
            self.room.add_source(
                self.source_locs[idx], signal=audio[: self.min_size_audio])

    def _remove_source(self, index):
        """
        This function removes a source from the environment

        Args:
            index (int): index of the source to remove
        """
        if index < len(self.source_locs):
            src = self.source_locs.pop(index)
            src2 = self.direct_sources.pop(index)

            # actually remove source from the room
            room_src = self.room.sources.pop(index)

    def step(self, action):
        """
        This function simulates the agent taking one step in the environment (and room) given an action:
            0 = Move forward
            1 = Move backward
            2 = Turn right x degrees
            3 = Turn left x degrees

        It calls _move_agent, checks to see if the agent has reached a source, and if not, computes the RIR.

        Args:
            action (int): direction agent is to move - 0 (Forward), 1 (Backward), 2 (Rotate Right), 3 (Rotate Left)

        Returns:
            tuple: Contains:
                - observation (dict): Dictionary containing 'audio' and 'state' observations
                - reward (float): Total reward for this step
                - terminated (bool): Whether the episode has ended due to task completion
                - truncated (bool): Whether the episode was truncated (e.g., time limit)
                - info (dict): Additional information including detailed rewards and positions
        """
        # Initialize reward components
        reward = {
            'step_penalty': constants.STEP_PENALTY,
            'turn_off_reward': 0,
            'closest_reward': 0,
            'orient_penalty': 0,
            'completion_reward': 0
        }
        #moving into the source of audio distance reward
        terminated = False
        truncated = False
        info = {
            'agent_loc': self.agent_loc.copy(),
            'source_locs': [loc.copy() for loc in self.source_locs],
            'cur_angle': self.cur_angle,
            'rewards': reward.copy()
        }

        # Movement logic
        x, y = self.agent_loc[0], self.agent_loc[1]

        if action in [0, 1]:  # Forward or Backward
            sign = 1 if action == 0 else -1
            x = x + sign * np.cos(self.cur_angle) * self.step_size
            y = y + sign * np.sin(self.cur_angle) * self.step_size
        elif action == 2:  # Rotate Right
            self.cur_angle = round((self.cur_angle + self.degrees) % (2 * np.pi), 4)
            reward['orient_penalty'] = constants.ORIENT_PENALTY
        elif action == 3:  # Rotate Left
            self.cur_angle = round((self.cur_angle - self.degrees) % (2 * np.pi), 4)
            reward['orient_penalty'] = constants.ORIENT_PENALTY

        # Check if new position is within room boundaries
        try:
            if self.room.is_inside([x, y]):
                points = np.array([x, y])
            else:
                points = self.agent_loc
        except:
            points = self.agent_loc

        # Update agent position
        self._move_agent(new_agent_loc=points)
        info['agent_loc'] = self.agent_loc.copy()

        # Check if any source is reached
        for index, source in enumerate(self.source_locs):
            if euclidean(self.agent_loc, source) <= self.acceptable_radius:
                print(f"source locs: {self.source_locs}, direct sources: {self.direct_sources}")
                found_source = self.direct_sources[index]
                
                print(f'Agent has found source {found_source}. \nAgent loc: {self.agent_loc}, Source loc: {source}')
                reward['turn_off_reward'] = constants.TURN_OFF_REWARD
                progress_factor = (self.num_sources - len(self.source_locs)) + 1/ self.num_sources
                reward['completion_reward'] = progress_factor * constants.COMPLETED_PROGRESS_REWARD
                # Add source info to the info dictionary for visualization
                info['source_info'] = {
                    'source_file': found_source,
                    'source_loc': source,
                    'source_index': index
                }
                
                if len(self.source_locs) > 1:
                    # Remove the source for next step
                    self._remove_source(index=index)
                    self.room.compute_rir()
                    self.room.simulate()
                    data = self.room.mic_array.signals
                    audio_obs = AudioSignal(audio_data_array=data, sample_rate=self.resample_rate)
                    
                    # Create observation
                    state_obs = np.array([*self.agent_loc, self.cur_angle], dtype=np.float32)
                    observation = {
                        'audio': audio_obs.audio_data.squeeze(),
                        'state': state_obs
                    }
                    info['rewards'] = reward
                    if self.visualize_pygame:
                        self._render_pygame()
                    return observation, sum(reward.values()), terminated, truncated, info
                else:
                    # Last source found
                    terminated = True
                    self.reset()
                    # Return empty observation for terminated state
                    if self.visualize_pygame:
                        self._render_pygame()
                    return {}, sum(reward.values()), terminated, truncated, info
        # If no source found this step
        self.room.compute_rir()
        self.room.simulate()
        data = self.room.mic_array.signals
        
        # Create audio observation
        audio_obs = AudioSignal(audio_data_array=data, sample_rate=self.resample_rate)
        
        # Calculate distance-based reward
        min_dist = euclidean_distances(
            np.array(self.agent_loc).reshape(1, -1), self.source_locs).min()
        reward['closest_reward'] = (1 / (min_dist + 1e-4))
        
        # Create observation
        state_obs = np.array([*self.agent_loc, self.cur_angle], dtype=np.float32)
        observation = {
            'audio': audio_obs.audio_data.squeeze(),
            'state': state_obs
        }
        
        info['rewards'] = reward

        if self.play_audio_on_step:
            self._play_current_audio()
            
        if self.visualize_pygame:
            self._render_pygame()
        return observation, sum(reward.values()), terminated, truncated, info

    def _play_current_audio(self):
        if not self.play_audio_on_step:
            return
        if not pygame.mixer.get_init():
            print("WARN: _play_current_audio: Pygame mixer not initialized. Skipping playback.")
            return
        if self.room.mic_array is None or self.room.mic_array.signals is None:
            print("WARN: _play_current_audio: No microphone signals available. Skipping playback.")
            return

        raw_audio_data_float = self.room.mic_array.signals
        # print(f"DEBUG: _play_current_audio: Initial self.room.mic_array.signals shape: {raw_audio_data_float.shape}, self.num_channels (agent mics): {self.num_channels}")

        # --- Step 1: Ensure audio_data_float is (num_samples, actual_agent_mic_channels) ---
        audio_data_float = None
        if raw_audio_data_float.ndim == 1:  # Mono mic array from pyroomacoustics
            if self.num_channels != 1:
                print(f"WARN: _play_current_audio: Data is 1D (mono) but self.num_channels (agent mics) is {self.num_channels}. Assuming mono data from pyroom.")
            audio_data_float = raw_audio_data_float[:, np.newaxis]  # Shape: (num_samples, 1)
        elif raw_audio_data_float.ndim == 2:
            if raw_audio_data_float.shape[0] == self.num_channels: # Expected (num_agent_mics, num_samples)
                # print(f"INFO: _play_current_audio: Transposing audio data from {raw_audio_data_float.shape} to (num_samples, num_agent_mics).")
                audio_data_float = raw_audio_data_float.T
            elif raw_audio_data_float.shape[1] == self.num_channels: # Already (num_samples, num_agent_mics)
                # print(f"INFO: _play_current_audio: Audio data already (num_samples, num_agent_mics) {raw_audio_data_float.shape}.")
                audio_data_float = raw_audio_data_float
            else:
                print(f"WARN: _play_current_audio: Ambiguous 2D audio data shape {raw_audio_data_float.shape} for self.num_channels={self.num_channels}. Cannot determine sample/channel axes. Skipping playback.")
                return
        else:
            print(f"WARN: _play_current_audio: Unexpected audio data ndim {raw_audio_data_float.ndim}. Skipping playback.")
            return

        if audio_data_float.shape[0] == 0:
            print("WARN: _play_current_audio: audio_data_float has 0 samples after processing. Skipping playback.")
            return
        
        actual_agent_mic_channels = audio_data_float.shape[1]
        # print(f"DEBUG: _play_current_audio: Actual agent mic channels after reshape: {actual_agent_mic_channels}")

        audio_data_int16 = np.int16(np.ascontiguousarray(audio_data_float) * 32767)

        # --- Step 2: Adapt agent's audio channels to what the Pygame mixer is actually using ---
        mixer_actual_num_channels = pygame.mixer.get_num_channels()
        # print(f"DEBUG: _play_current_audio: Mixer actual channels: {mixer_actual_num_channels}, Agent mic channels: {actual_agent_mic_channels}")

        final_audio_data_for_mixer = audio_data_int16

        if mixer_actual_num_channels != actual_agent_mic_channels:
            print(f"INFO: _play_current_audio: Channel count mismatch. Mixer wants {mixer_actual_num_channels}, agent data has {actual_agent_mic_channels}. Adapting...")
            adapted_sound_data = np.zeros((audio_data_int16.shape[0], mixer_actual_num_channels), dtype=np.int16)

            if actual_agent_mic_channels == 1:  # Mono agent data
                if mixer_actual_num_channels >= 2:
                    print("INFO: _play_current_audio: Playing mono agent data on mixer's front-left & front-right channels.")
                    adapted_sound_data[:, 0] = audio_data_int16[:, 0]  # FL
                    adapted_sound_data[:, 1] = audio_data_int16[:, 0]  # FR
                elif mixer_actual_num_channels == 1:
                    print("INFO: _play_current_audio: Playing mono agent data on mono mixer.")
                    adapted_sound_data[:, 0] = audio_data_int16[:, 0]
                final_audio_data_for_mixer = adapted_sound_data
            elif actual_agent_mic_channels == 2:  # Stereo agent data
                if mixer_actual_num_channels >= 2:
                    print("INFO: _play_current_audio: Playing stereo agent data on mixer's front-left & front-right channels.")
                    adapted_sound_data[:, 0] = audio_data_int16[:, 0]  # Agent Left to Mixer FL
                    adapted_sound_data[:, 1] = audio_data_int16[:, 1]  # Agent Right to Mixer FR
                elif mixer_actual_num_channels == 1:
                    print("INFO: _play_current_audio: Downmixing stereo agent data to mono for mixer (averaging L/R).")
                    adapted_sound_data[:, 0] = (audio_data_int16[:, 0] // 2 + audio_data_int16[:, 1] // 2)
                final_audio_data_for_mixer = adapted_sound_data
            elif actual_agent_mic_channels > 2:
                print(f"WARN: _play_current_audio: Agent data has {actual_agent_mic_channels} channels. Attempting to map to {mixer_actual_num_channels} mixer channels by copying first available.")
                channels_to_copy = min(actual_agent_mic_channels, mixer_actual_num_channels)
                if channels_to_copy > 0:
                    adapted_sound_data[:, :channels_to_copy] = audio_data_int16[:, :channels_to_copy]
                    final_audio_data_for_mixer = adapted_sound_data
                else:
                    print(f"WARN: _play_current_audio: Cannot map {actual_agent_mic_channels}-channel agent data to {mixer_actual_num_channels}-channel mixer. Skipping playback.")
                    return
            else:
                print(f"WARN: _play_current_audio: Unhandled case for adapting {actual_agent_mic_channels}-channel agent data to {mixer_actual_num_channels}-channel mixer. Skipping playback.")
                return

        if final_audio_data_for_mixer.shape[1] != mixer_actual_num_channels:
            print(f"CRITICAL WARN: _play_current_audio: Mismatch persists after adaptation! Mixer wants {mixer_actual_num_channels}, final data has {final_audio_data_for_mixer.shape[1]}. Playback will likely fail.")
            return
        
        # print(f"DEBUG: _play_current_audio: Final audio data for mixer - shape: {final_audio_data_for_mixer.shape}, dtype: {final_audio_data_for_mixer.dtype}")

        try:
            pygame.mixer.stop()  # Stop any currently playing sound
            sound = pygame.sndarray.make_sound(np.ascontiguousarray(final_audio_data_for_mixer))
            sound.play()
            # print("DEBUG: _play_current_audio: Sound playing.")
        except Exception as e:
            print(f"WARN: audio_env.py: Error during pygame.sndarray.make_sound or sound.play: {e}") 

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and return the initial observation.

        Args:
            seed (int, optional): Seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for environment reset. Can include:
                - removing_source (int): Index of source to remove
                - same_config (bool): Whether to use the same configuration as previous episodes

        Returns:
            tuple: Contains:
                - observation (dict): Initial observation containing 'audio' and 'state'
                - info (dict): Additional information about the environment state
        """
        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Handle options
        removing_source = options.get('removing_source') if options else None
        same_config = options.get('same_config', self.same_config) if options else self.same_config
        
        # Re-create room
        self._create_room()

        if not same_config:
            # Randomly add sources to the room
            self._add_sources()
            # Randomly place agent in room at beginning of next episode
            self._move_agent(new_agent_loc=None, initial_placing=True)
        else:
            # Place sources and agent at same locations to start every episode
            self._add_sources(new_source_locs=self.fixed_source_locs)
            self._move_agent(new_agent_loc=self.fixed_agent_locs, initial_placing=True)
        
        # Validate room configuration before computing RIR
        valid_config = self._validate_room_config()
        if not valid_config:
            # If invalid, recreate the room and try again
            print("Invalid room configuration detected, recreating room...")
            self._create_room()
            self._add_sources()
            self._move_agent(new_agent_loc=None, initial_placing=True)
    
        # Get initial observation with error handling
        try:
            self.room.compute_rir()
            self.room.simulate()
            data = self.room.mic_array.signals
        except ValueError as e:
            print(f"Error in PyRoomAcoustics: {e}\nRecreating room with different configuration...")
            # Try again with different configuration
            self._create_room()
            self._add_sources()
            self._move_agent(new_agent_loc=None, initial_placing=True)
            # Attempt computation again
            self.room.compute_rir()
            self.room.simulate()
            data = self.room.mic_array.signals
        
        # Create audio observation
        audio_obs = AudioSignal(audio_data_array=data, sample_rate=self.resample_rate)
        
        # Create state observation
        state_obs = np.array([*self.agent_loc, self.cur_angle], dtype=np.float32)
        
        # Create observation dictionary
        observation = {
            'audio': audio_obs.audio_data.squeeze(),
            'state': state_obs
        }
        
        # Create info dictionary
        info = {
            'agent_loc': self.agent_loc.copy(),
            'source_locs': [loc.copy() for loc in self.source_locs],
            'cur_angle': self.cur_angle,
            'sample_rate': self.resample_rate
        }
        
        if self.play_audio_on_step:
            self._play_current_audio()
            
        if self.visualize_pygame:
            self._render_pygame()
        return observation, info

    def _render_pygame(self):
        if not self.visualize_pygame or self.pygame_screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close() # This will call pygame.quit() and set self.pygame_screen to None
                self.visualize_pygame = False # Stop further rendering attempts in this session
                return

        self.pygame_screen.fill(BLACK)

        # Draw room boundaries
        if self.room_config is not None:
            if not self.corners:  # Shoebox
                tl_px, tl_py = self._world_to_pixel(self.world_min_x, self.world_min_y + self.room_height_world)
                br_px, br_py = self._world_to_pixel(self.world_min_x + self.room_width_world, self.world_min_y)
                pixel_room_width = abs(br_px - tl_px)
                pixel_room_height = abs(br_py - tl_py)
                rect_tl_px = min(tl_px, br_px)
                rect_tl_py = min(tl_py, br_py)
                pygame.draw.rect(self.pygame_screen, WHITE, (rect_tl_px, rect_tl_py, pixel_room_width, pixel_room_height), 2)
            else:  # Polygonal
                pixel_corners = []
                for corner_x, corner_y in self.room_config:
                    px, py = self._world_to_pixel(corner_x, corner_y)
                    pixel_corners.append((px, py))
                if len(pixel_corners) > 2:
                    pygame.draw.polygon(self.pygame_screen, WHITE, pixel_corners, 2)

        # Draw sources (self.source_locs) with dynamic radius based on RIR energy.
        # The radius is calculated as follows:
        # 1. RIR energy is computed for the source at the agent's current microphone position.
        # 2. Energy is transformed (e.g., log1p) for better perceptual scaling.
        # 3. Transformed energy is normalized to a [0, 1] range using APPROX_MAX_LOG_ENERGY.
        #    This value represents the expected maximum log_energy; tune it based on observations.
        # 4. The normalized intensity is optionally passed through a non-linear scaling (e.g., power function)
        #    to enhance visual distinction, especially at lower intensities.
        # 5. This scaled intensity is then mapped to a pixel radius between MIN_SOURCE_RADIUS and MAX_SOURCE_RADIUS.

        if self.source_locs is not None and self.room.sources is not None and self.room.rir is not None and len(self.room.rir) > 0:
            # MIN_SOURCE_RADIUS: Smallest pixel radius for a source. Ensures visibility.
            MIN_SOURCE_RADIUS = 3
            # MAX_SOURCE_RADIUS: Largest pixel radius. Defines the upper bound of visual feedback.
            MAX_SOURCE_RADIUS = 30 # Increased for more visual range
            # APPROX_MAX_LOG_ENERGY: Expected maximum for log1p(energy). Used for normalization.
            # Tune this based on typical log_energy values observed when agent is very close to a source.
            # Lowering it makes the radius more sensitive to energy changes if max observed log_energy is low.
            APPROX_MAX_LOG_ENERGY = 1.0 # Reduced for more sensitivity
            # INTENSITY_SCALING_EXPONENT: Exponent for non-linear scaling of normalized_intensity.
            # Values < 1 (e.g., 0.5-0.7) make radius grow faster at lower intensities.
            # Value = 1 means linear scaling.
            INTENSITY_SCALING_EXPONENT = 0.7

            active_source_pra_indices = {}
            for pra_idx, pra_source in enumerate(self.room.sources):
                pos_key = tuple(pra_source.position[:self.dim]) 
                active_source_pra_indices[pos_key] = pra_idx

            for env_src_idx, src_loc in enumerate(self.source_locs):
                if src_loc is not None:
                    src_px, src_py = self._world_to_pixel(src_loc[0], src_loc[1])
                    radius = 7 # Default radius if RIR processing fails or source not mapped
                    log_msg_reason = "default (no RIR processing triggered)"
                    
                    current_pos_key = tuple(src_loc[:self.dim])
                    pra_idx_for_rir = active_source_pra_indices.get(current_pos_key)

                    if pra_idx_for_rir is not None:
                        try:
                            if pra_idx_for_rir < len(self.room.rir[0]):
                                rir_data = self.room.rir[0][pra_idx_for_rir]
                                energy = np.sum(np.square(rir_data))
                                log_energy = np.log1p(energy) # log(1+energy) for stability
                                
                                normalized_intensity = np.clip(log_energy / APPROX_MAX_LOG_ENERGY, 0, 1)
                                scaled_intensity = pow(normalized_intensity, INTENSITY_SCALING_EXPONENT)
                                
                                new_radius = MIN_SOURCE_RADIUS + scaled_intensity * (MAX_SOURCE_RADIUS - MIN_SOURCE_RADIUS)
                                radius = new_radius
                                log_msg_reason = f"from RIR (E={energy:.2e}, logE={log_energy:.2f}, normI={normalized_intensity:.2f}, scaledI={scaled_intensity:.2f})"
                                
                                if not (MIN_SOURCE_RADIUS <= radius <= MAX_SOURCE_RADIUS + 1e-3): # Add tolerance for float comparisons
                                     log_msg_reason += f" - WARN: radius {radius:.2f} out of range, clipped"
                                     radius = np.clip(radius, MIN_SOURCE_RADIUS, MAX_SOURCE_RADIUS)
                            else:
                                log_msg_reason = f"default (pra_idx {pra_idx_for_rir} OOB for RIR len {len(self.room.rir[0])})"
                                radius = 7
                        except Exception as e:
                            log_msg_reason = f"default (RIR error: {str(e)[:30]}...)"
                            radius = 7
                    else:
                        log_msg_reason = "default (source not in RIR map)"
                        radius = 7
                    
                    print(f"Src {env_src_idx}@({src_loc[0]:.1f},{src_loc[1]:.1f}): r={int(radius)} ({log_msg_reason})")
                    pygame.draw.circle(self.pygame_screen, SOURCE_COLOR, (src_px, src_py), int(radius))

        # Draw agent (self.agent_loc, self.cur_angle)
        if hasattr(self, 'agent_loc') and self.agent_loc is not None:
            agent_px, agent_py = self._world_to_pixel(self.agent_loc[0], self.agent_loc[1])
            pygame.draw.circle(self.pygame_screen, AGENT_COLOR, (agent_px, agent_py), 10)
            
            line_len = 20 
            end_x = agent_px + line_len * np.cos(self.cur_angle)
            # Pygame's y-axis is inverted, so for standard math angle, sin component is subtracted.
            end_y = agent_py - line_len * np.sin(self.cur_angle) 
            pygame.draw.line(self.pygame_screen, WHITE, (agent_px, agent_py), (int(end_x), int(end_y)), 3)

        pygame.display.flip()
        if self.pygame_clock:
            self.pygame_clock.tick(30)  # Limit to 30 FPS
    def close(self):
        if self.pygame_screen is not None:
            pygame.quit()
            self.pygame_screen = None # Mark as closed to prevent further use
            self.pygame_clock = None