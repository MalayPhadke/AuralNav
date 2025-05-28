import mesa
import numpy as np
import gymnasium as gym

from ..audio_room.envs.audio_env import AudioEnv
from .agents import NavigatingAgent, SourceAgent

class AudioMesaModel(mesa.Model):
    """
    A Mesa model that integrates with the AudioEnv Gymnasium environment.
    """

    def __init__(self, room_config, source_folders_dict, env_config_overrides={}):
        """
        Create a new AudioMesaModel.

        Args:
            room_config: Configuration for the room geometry.
            source_folders_dict: Dictionary mapping source types to their sound folders.
            env_config_overrides: Dictionary of overrides for the AudioEnv configuration.
        """
        super().__init__()
        self.running = True

        base_env_config = {
            "resample_rate": 8000,
            "num_channels": 2,
            "corners": False,  # Assuming ShoeBox for default
            "absorption": 0.0,
            "max_order": 2,
            "step_size": 0.5,
            "acceptable_radius": 0.5,
            "num_sources": len(source_folders_dict),  # Derived from source_folders_dict
            "degrees": np.deg2rad(30),
            "reset_sources": True,
            "same_config": False,
            "play_audio_on_step": False,
            "show_room_on_step": False
        }
        base_env_config.update(env_config_overrides)

        self.audio_env = AudioEnv(
            room_config=room_config,
            source_folders_dict=source_folders_dict,
            **base_env_config
        )
        self.audio_env.reset()  # Initialize and set source_locs, agent_loc

        self.schedule = mesa.time.RandomActivationByType(self)

        # Determine room boundaries for ContinuousSpace
        x_min = self.audio_env.x_min
        x_max = self.audio_env.x_max
        y_min = self.audio_env.y_min
        y_max = self.audio_env.y_max
        self.space = mesa.space.ContinuousSpace(x_max, y_max, False, x_min, y_min)

        # Populate Agents
        # Create and add SourceAgent instances
        # Assuming self.audio_env.direct_sources contains Path objects or strings
        # and the folder name before the sound file is the source_type.
        # Example: .../sounds/phone/sound.wav -> source_type = 'phone'
        source_types = list(source_folders_dict.keys()) # Get available source types

        for i, source_loc in enumerate(self.audio_env.source_locs):
            pos = tuple(source_loc)
            source_type = "unknown" # Default
            if i < len(self.audio_env.direct_sources) and i < len(source_types):
                # Attempt to derive source_type from the folder name or use keys from source_folders_dict
                # This is a simplified assumption; AudioEnv might need to expose types more directly.
                try:
                    # If direct_sources are paths, extract parent folder name
                    # This logic might need adjustment based on actual structure of audio_env.direct_sources
                    path_obj = self.audio_env.direct_sources[i] # Assuming it's a Path or string
                    if isinstance(path_obj, str): # If it's a string path
                        type_candidate = path_obj.split('/')[-2]
                    else: # Assuming it's a Path object
                        type_candidate = path_obj.parent.name

                    if type_candidate in source_types:
                        source_type = type_candidate
                    elif source_types: # Fallback if specific type not found but types exist
                        source_type = source_types[i % len(source_types)]

                except (AttributeError, IndexError, TypeError):
                    # Fallback if direct_sources doesn't provide clear type info
                    if source_types:
                         source_type = source_types[i % len(source_types)]


            source_agent = SourceAgent(self.next_id(), self, pos, source_type=source_type)
            self.space.place_agent(source_agent, pos)
            self.schedule.add(source_agent)

        # Create and add NavigatingAgent
        agent_pos = tuple(self.audio_env.agent_loc)
        agent_angle = self.audio_env.cur_angle
        nav_agent = NavigatingAgent(self.next_id(), self, agent_pos, agent_angle)
        self.space.place_agent(nav_agent, agent_pos)
        self.schedule.add(nav_agent)

        # Initialize a datacollector
        self.datacollector = mesa.DataCollector(
            model_reporters={"AgentCount": lambda m: m.schedule.get_agent_count()}
        )
        self.datacollector.collect(self) # Initial collection

    def step(self):
        """
        Advance the model by one step.
        """
        if not self.running:
            return

        # Choose an action
        action = self.audio_env.action_space.sample()  # Requires gymnasium import

        # Step the underlying Gym environment
        observation, reward, terminated, truncated, info = self.audio_env.step(action)

        # Update Agent States in Mesa
        # Find the NavigatingAgent
        # Assuming there's only one NavigatingAgent
        navigating_agent_instance = None
        if NavigatingAgent in self.schedule.agents_by_type:
            for agent in self.schedule.agents_by_type[NavigatingAgent].values():
                navigating_agent_instance = agent
                break # Should be only one

        if navigating_agent_instance:
            new_pos = tuple(self.audio_env.agent_loc)
            self.space.move_agent(navigating_agent_instance, new_pos)
            navigating_agent_instance.pos = new_pos
            navigating_agent_instance.angle = self.audio_env.cur_angle

        # Handle source removal (simplified: assume sources are static for now)
        # If sources could be removed, one would need to:
        # 1. Get current source locations from self.audio_env.source_locs
        # 2. Compare with self.schedule.agents_by_type[SourceAgent]
        # 3. Identify and remove agents from self.space and self.schedule
        # For example, if a source is removed and audio_env.source_locs is updated:
        # current_env_source_positions = [tuple(loc) for loc in self.audio_env.source_locs]
        # agents_to_remove = []
        # for agent in self.schedule.agents_by_type.get(SourceAgent, {}).values():
        #     if agent.pos not in current_env_source_positions:
        #         agents_to_remove.append(agent)
        # for agent in agents_to_remove:
        #     self.space.remove_agent(agent)
        #     self.schedule.remove(agent)


        self.schedule.step()
        self.datacollector.collect(self)

        if terminated or truncated:
            self.running = False
