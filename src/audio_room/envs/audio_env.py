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

# Suppress 'c' argument errors caused by room.plot()
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


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
        degrees=np.deg2rad(30),
        reset_sources=True,
        same_config=False,
        play_audio_on_step=False,
        show_room_on_step=False
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
            play_audio_on_step (bool): If true, audio will play after every step
            show_room_on_step (bool): If true, room will be displayed after every step
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
        self.source_locs = None
        self.min_size_audio = np.inf
        self.degrees = degrees
        self.cur_angle = 0
        
        # Define observation space
        # Using large dimensions for audio that will be updated after loading audio
        self.observation_space = spaces.Dict({
            'audio': spaces.Box(low=-np.inf, high=np.inf, shape=(100000, self.num_channels), dtype=np.float32),
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        self.reset_sources = reset_sources
        self.source_folders_dict = source_folders_dict
        self.same_config = same_config
        self.play_audio_on_step = play_audio_on_step
        self.show_room_on_step = show_room_on_step

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
            'orient_penalty': 0
        }
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
                found_source = self.direct_sources[index]
                print(f'Agent has found source {found_source}. \nAgent loc: {self.agent_loc}, Source loc: {source}')
                reward['turn_off_reward'] = constants.TURN_OFF_REWARD
                
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
                    
                    if self.play_audio_on_step or self.show_room_on_step:
                        self.render(audio_obs, self.play_audio_on_step, self.show_room_on_step)
                    
                    # Create observation
                    state_obs = np.array([*self.agent_loc, self.cur_angle], dtype=np.float32)
                    observation = {
                        'audio': audio_obs.audio_data.squeeze(),
                        'state': state_obs
                    }
                    info['rewards'] = reward
                    return observation, sum(reward.values()), terminated, truncated, info
                else:
                    # Last source found
                    terminated = True
                    self.reset()
                    # Return empty observation for terminated state
                    return {}, sum(reward.values()), terminated, truncated, info

        # If no source found this step
        self.room.compute_rir()
        self.room.simulate()
        data = self.room.mic_array.signals
        
        # Create audio observation
        audio_obs = AudioSignal(audio_data_array=data, sample_rate=self.resample_rate)
        
        if self.play_audio_on_step or self.show_room_on_step:
            self.render(audio_obs, self.play_audio_on_step, self.show_room_on_step)
        
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
        return observation, sum(reward.values()), terminated, truncated, info

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
        
        return observation, info

    def render(self, data, play_audio, show_room):
        """
        Play the convolved sound using SimpleAudio.

        Args:
            data (AudioSignal): if 2 mics, should be of shape (x, 2)
            play_audio (bool): If true, audio will play
            show_room (bool): If true, room will be displayed to user
        """
        if play_audio:
            data.embed_audio(display=True)

            # Show the room while the audio is playing
            if show_room:
                fig, ax = self.room.plot(img_order=0)
                plt.pause(1)

            plt.close()

        elif show_room:
            fig, ax = self.room.plot(img_order=0)
            plt.pause(1)
            plt.close()
