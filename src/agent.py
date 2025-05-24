import time
import logging
import warnings
from collections import deque

import numpy as np
import gymnasium as gym
from scipy.spatial.distance import euclidean

import utils
import constants
from datasets import BufferData
from visualization import EnvironmentVisualizer

# setup logging
logging.basicConfig(level=logging.INFO)
logging_str = (
        f"\n\n"
        f"------------------------------ \n"
        f"- Starting to Fit Agent\n"
        f"------------------------------- \n\n"
)
logging.info(logging_str)


class AgentBase:
    def __init__(
        self,
        env,
        dataset,
        episodes=1,
        max_steps=100,
        gamma=0.98,
        alpha=0.001,
        decay_rate=0.0005,
        stable_update_freq=-1,
        save_freq=1,
        play_audio=False,
        show_room=False,
        writer=None,
        dense=True,
        decay_per_ep=False,
        validation_freq=None
    ):
        """
        This class is a base agent class which will be inherited when creating various agents.

        Args:
            self.env (gym object): The gym self.environment object which the agent is going to explore
            dataset (nussl dataset): Nussl dataset object for experience replay
            episodes (int): # of episodes to simulate
            max_steps (int): # of steps the agent can take before stopping an episode
            gamma (float): Discount factor
            alpha (float): Learning rate alpha
            decay_rate (float): decay rate for exploration rate (we want to decrease exploration as time proceeds)
            stable_update_freq (int): Update frequency value for stable networks (Target networks)
            play_audio (bool): choose to play audio at each iteration
            show_room (bool): choose to display the configurations and movements within a room
            writer (torch.utils.tensorboard.SummaryWriter): for logging to tensorboard
            dense (bool): makes the rewards more dense, less sparse
                gives reward for distance to closest source every step
            decay_per_ep (bool): If set to true, epsilon is decayed per episode, else decayed per step
            validation_freq (None or int): if not None, then do a validation run every validation_freq # of episodes
                validation run means epsilon=0 (no random actions) and no model update/training
        """
        self.env = env
        self.dataset = dataset
        self.episodes = episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = constants.MAX_EPSILON
        self.decay_rate = decay_rate
        self.stable_update_freq = stable_update_freq
        self.save_freq = save_freq
        self.play_audio = play_audio
        self.show_room = show_room
        self.writer = writer
        self.dense = dense
        self.decay_per_ep = decay_per_ep
        self.validation_freq = validation_freq
        self.losses = []
        self.cumulative_reward = 0
        self.total_experiment_steps = 0
        self.mean_episode_reward = []
        self.action_memory = deque(maxlen=4)

        # for saving model at best validation episode (as determined by mean reward in the episode)
        if self.validation_freq is not None:
            self.max_validation_reward = -np.inf

    def fit(self, visualize=False):
        # Initialize visualizer if visualization is enabled
        # if visualize:
        #     self.visualizer = EnvironmentVisualizer()
        #     self.source_detection_tracker = []  # Track source detections for visualization
        
        for episode in range(1, self.episodes + 1):
            # Reset the self.environment and any other variables at beginning of each episode
            prev_state = None

            episode_rewards = []
            found_sources = []

            # # Setup visualization for this episode
            # if visualize:
            #     # Get room boundaries from environment
            #     if hasattr(self.env.unwrapped, 'corners') and self.env.unwrapped.corners:
            #         room_bounds = [self.env.unwrapped.room_config[0], self.env.unwrapped.room_config[1]]
            #     else:
            #         room_bounds = [self.env.unwrapped.x_min, self.env.unwrapped.y_min, 
            #                        self.env.unwrapped.x_max, self.env.unwrapped.y_max]
                
            #     # Get source locations - convert np.float64 to regular floats if needed
            #     source_locs = []
            #     for loc in self.env.unwrapped.source_locs:
            #         # Convert numpy values to standard float if needed
            #         if hasattr(loc[0], 'item'):
            #             source_locs.append([loc[0].item(), loc[1].item()])
            #         else:
            #             source_locs.append([float(loc[0]), float(loc[1])])
            
            #     # Store source locations for later reference
            #     self.original_source_locs = source_locs.copy()
                
            #     # Reset visualizer with new room configuration
            #     self.visualizer.reset(
            #         room_boundaries=room_bounds, 
            #         source_locs=source_locs,
            #         acceptable_radius=self.env.unwrapped.acceptable_radius
            #     )
                
            #     # Add initial agent position
            #     if hasattr(self.env.unwrapped, 'agent_loc'):
            #         self.visualizer.update(self.env.unwrapped.agent_loc, self.env.unwrapped.cur_angle)
                    
            #     # Reset source detection tracker for this episode
            #     self.source_detection_tracker = []
                    
            # validation episode?
            validation_episode = False
            if self.validation_freq is not None:
                validation_episode = True if (episode % self.validation_freq == 0) else False

            # Measure time to complete the episode
            start = time.time()
            for step in range(self.max_steps):
                self.total_experiment_steps += 1

                # Perform random actions with prob < epsilon
                model_action = False
                if (np.random.uniform(0, 1) < self.epsilon):
                    action = self.env.action_space.sample()
                else:
                    model_action = True

                # validation run: no random actions and no model update/training
                if validation_episode:
                    model_action = True

                if model_action:
                    # For the first two steps (We don't have prev_state, new_state pair), then perform a random action
                    if step < 2:
                        action = self.env.action_space.sample()
                    else:
                        # This is where agent will actually do something
                        action = self.choose_action()

                # if same action 4x in a row (to avoid model infinite loop), choose an action randomly
                self.action_memory.append(action)
                if all(self.action_memory[0] == x for x in self.action_memory):
                    action = self.env.action_space.sample()
                    self.action_memory.append(action)

                # Move/rotate agent based on action
                observation, scalar_reward, terminated, truncated, info_dict = self.env.step(action)

                # Map new return values to old variable names for compatibility
                new_state = observation
                agent_info = info_dict # info_dict contains the 'rewards' dictionary
                total_step_reward = scalar_reward # AudioEnv.step already returns sum of rewards
                won = terminated
                
                # Update visualization with new agent position
                # if visualize and hasattr(self.env.unwrapped, 'agent_loc'):
                #     self.visualizer.update(self.env.unwrapped.agent_loc, self.env.unwrapped.cur_angle)
                    
                #     # Check if a source was found by looking at the reward
                #     if agent_info.get('rewards', {}).get('turn_off_reward') == constants.TURN_OFF_REWARD:
                #         # Get exact agent and source locations from agent_info
                #         agent_loc = agent_info['agent_loc']
                        
                #         # If source_info is available directly in agent_info (from environment)
                #         if 'source_info' in agent_info:
                #             source_info = agent_info['source_info']
                #             detected_source = source_info['source_loc']
                #             detected_idx = source_info['source_index']
                            
                #             # Create detection record with exact positions
                #             self.visualizer.record_detection(
                #                 agent_position=np.array(agent_loc),
                #                 source_position=np.array(detected_source),
                #                 step_number=step,
                #                 source_index=detected_idx
                #             )
                            
                #             # Add to our tracking list
                #             self.source_detection_tracker.append((agent_loc, step, detected_idx))
                            
                #             print(f"Visualization: Recorded detection of source {detected_idx+1} at step {step}")
                #         else:
                #             # Fallback to the old method of finding closest source
                #             detected_source = None
                #             min_dist = float('inf')
                #             detected_idx = -1
                            
                #             # Check each source that we haven't already detected
                #             detected_indices = [idx for _, _, idx in self.source_detection_tracker]
                            
                #             for idx, src_loc in enumerate(self.original_source_locs):
                #                 # Skip sources we've already detected
                #                 if idx in detected_indices:
                #                     continue
                                    
                #                 # Calculate distance to this source
                #                 dist = np.linalg.norm(np.array(agent_loc) - np.array(src_loc))
                                
                #                 # If this is the closest source, record it
                #                 if dist < min_dist:
                #                     min_dist = dist
                #                     detected_idx = idx
                #                     detected_source = src_loc
                            
                #             # Record the detection if we found a source
                #             if detected_source is not None and detected_idx >= 0:
                #                 # Create detection record with exact positions
                #                 self.visualizer.record_detection(
                #                     agent_position=np.array(agent_loc),
                #                     source_position=np.array(detected_source),
                #                     step_number=step,
                #                     source_index=detected_idx
                #                 )
                                
                #                 # Add to our tracking list
                #                 self.source_detection_tracker.append((agent_loc, step, detected_idx))
                                
                #                 print(f"Visualization: Recorded detection of source {detected_idx+1} at step {step}")
                        
                #         # Check if all sources have been found
                #         if len(self.source_detection_tracker) == len(self.original_source_locs):
                #             print(f"All sources found at step {step}")
                
                # record reward stats
                self.cumulative_reward += total_step_reward
                episode_rewards.append(total_step_reward)

                # Access detailed rewards through agent_info (which is info_dict)
                if agent_info.get('rewards', {}).get('turn_off_reward') == constants.TURN_OFF_REWARD:
                    # Extract the source that was just found from agent_info
                    if 'source_info' in agent_info and 'source_file' in agent_info['source_info']:
                        found_source_file = agent_info['source_info']['source_file']
                        found_source_loc = agent_info['source_info'].get('source_loc')
                        print(f"Agent has found source {found_source_file}. \nAgent loc: {agent_info['agent_loc']}, Source loc: {found_source_loc}")
                    
                    print('In FIT. Received reward: {} at step {}\n'.format(total_step_reward, step))
                    logging.info(f"In FIT. Received reward {total_step_reward} at step: {step}\n")
                        
                # Perform Update
                if not validation_episode:
                    self.update()

                # store SARS in buffer
                if prev_state is not None and new_state is not None and not won:
                    self.dataset.write_buffer_data(
                        prev_state, action, total_step_reward, new_state, 
                        agent_info, episode, step, sample_rate=self.env.unwrapped.resample_rate
                    )

                # Decay epsilon based on total steps (across all episodes, not within an episode)
                if not self.decay_per_ep:
                    self.epsilon = constants.MIN_EPSILON + (
                        constants.MAX_EPSILON - constants.MIN_EPSILON
                    ) * np.exp(-self.decay_rate * self.total_experiment_steps)
                    if self.total_experiment_steps % 200 == 0:
                        print("Epsilon decayed to {} at step {} ".format(self.epsilon, self.total_experiment_steps))

                # Update stable networks based on number of steps
                if step % self.stable_update_freq == 0:
                    self.update_stable_networks()

                # Terminate the episode if episode is won, truncated, or at max steps
                if won or truncated or (step == self.max_steps - 1):
                    # terminal state is silence
                    # prev_state is a dict like {'audio': np_array, 'state': np_array}
                    if prev_state and 'audio' in prev_state and 'state' in prev_state:
                        silence_array = np.zeros_like(prev_state['audio'])
                        terminal_silent_state = {'audio': silence_array, 'state': prev_state['state']}
                    else:
                        # Fallback or error handling if prev_state is not as expected
                        # This might happen on the very first step if not handled carefully
                        # For now, creating a dummy structure if prev_state is problematic
                        # A more robust solution might be needed depending on execution flow
                        silence_array = np.array([]) # Or some default shape
                        terminal_silent_state = {'audio': silence_array, 'state': np.array([])}

                    self.dataset.write_buffer_data(
                        prev_state, action, total_step_reward, terminal_silent_state, 
                        agent_info, episode, step, sample_rate=self.env.unwrapped.resample_rate
                    )

                    # record mean reward for this episode
                    self.mean_episode_reward.append(np.mean(episode_rewards))

                    if validation_episode:
                        # new best validation reward
                        if np.mean(episode_rewards) > self.max_validation_reward:
                            self.max_validation_reward = np.mean(episode_rewards)

                            # save best validation model
                            self.save_model('best_valid_reward.pt')

                        if self.writer is not None:
                            self.writer.add_scalar('Reward/validation_mean_per_episode', np.mean(episode_rewards), episode)
                    elif self.writer is not None:
                        self.writer.add_scalar('Reward/mean_per_episode', np.mean(episode_rewards), episode)
                        self.writer.add_scalar('Reward/cumulative', self.cumulative_reward, self.total_experiment_steps)

                    end = time.time()
                    total_time = end - start

                    # log episode summary
                    logging_str = (
                        f"\n\n"
                        f"Episode Summary \n"
                        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n"
                        f"- Episode: {episode}\n"
                        f"- Won?: {won}\n"
                        f"- Finished at step: {step+1}\n"
                        f"- Time taken:   {total_time:04f} \n"
                        f"- Steps/Second: {float(step+1)/total_time:04f} \n"
                        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n"
                    )
                    print(logging_str)
                    logging.info(logging_str)

                    # End episode in visualizer and generate visualization
                    # if visualize:
                    #     # Add information about why the episode ended
                    #     end_reason = "All sources found" if won else "Max steps reached" if step == self.max_steps - 1 else "Truncated"
                        
                    #     # Create a record of episode summary in the visualization
                    #     episode_summary = {
                    #         "episode": episode,
                    #         "won": won,
                    #         "step": step+1,
                    #         "sources_found": len(self.source_detection_tracker),
                    #         "total_sources": len(self.original_source_locs),
                    #         "end_reason": end_reason
                    #     }
                        
                    #     # End the episode with summary data
                    #     self.visualizer.end_episode(episode_summary=episode_summary)
                        
                    # break and go to new episode
                    break

                prev_state = new_state

            if episode % self.save_freq == 0:
                name = 'ep{}.pt'.format(episode)
                self.save_model(name)

            # Decay epsilon per episode
            if self.decay_per_ep:
                self.epsilon = constants.MIN_EPSILON + (
                    constants.MAX_EPSILON - constants.MIN_EPSILON
                ) * np.exp(-self.decay_rate * episode)
                print("Decayed epsilon value: {}".format(self.epsilon))

            # Reset the environment
            self.env.reset()

    def choose_action(self):
        """
        This function must be implemented by subclass.
        It will choose the action to take at any time step.

        Returns:
            action (int): the action to take
        """
        raise NotImplementedError()

    def update(self):
        """
        This function must be implemented by subclass.
        It will perform an update (e.g. updating Q table or Q network)
        """
        raise NotImplementedError()

    def update_stable_networks(self):
        """
        This function must be implemented by subclass.
        It will perform an update to the stable networks. I.E Copy values from current network to target network
        """
        raise NotImplementedError()

    def save_model(self, name):
        raise NotImplementedError()


class RandomAgent(AgentBase):
    def choose_action(self):
        """
        Since this is a random agent, we just randomly sample our action 
        every time.
        """
        return self.env.action_space.sample()

    def update(self):
        """
        No update for a random agent
        """
        pass

    def update_stable_networks(self):
        pass

    def save_model(self, name):
        pass


# Create a perfect agent that steps to each of the closest sources one at a time.
class OracleAgent(AgentBase):
    """
    This agent is a perfect agent. It knows where all of the sources are and
    will iteratively go to the closest source at each time step.
    """

    def choose_action(self):
        """
        Since we know all information about the environment, the agent moves the following way.
        1. Find the closest audio source using euclidean distance
        2. Rotate to face the audio source if necessary
        3. Step in the direction of the audio source
        """
        agent = self.env.agent_loc
        radius = self.env.acceptable_radius
        action = None

        # find the closest audio source
        source, minimum_distance = self.env.source_locs[0], np.linalg.norm(
            agent - self.env.source_locs[0])
        for s in self.env.source_locs[1:]:
            dist = np.linalg.norm(agent - s)
            if dist < minimum_distance:
                minimum_distance = dist
                source = s

        # Determine our current angle
        angle = abs(int(np.degrees(self.env.cur_angle))) % 360
        # The agent is too far left or right of the source
        if np.abs(source[0] - agent[0]) >= radius:
            # First check if we need to turn
            if angle not in [0, 1, 179, 180, 181]:
                action = 2
            # We are facing correct way need to move forward or backward
            else:
                if source[0] < agent[0]:
                    if angle == 0:
                        action = 1
                    else:
                        action = 0
                else:
                    if angle == 0:
                        action = 0
                    else:
                        action = 1
        # Agent is to the right of the source
        elif np.abs(source[1] - agent[1]) >= radius:
            # First check if we need to turn
            if angle not in [89, 90, 91, 269, 270, 271]:
                action = 2
            # We are facing correct way need to move forward or backward
            else:
                if source[1] < agent[1]:
                    if angle == 90:
                        action = 1
                    else:
                        action = 0
                else:
                    if angle == 90:
                        action = 0
                    else:
                        action = 1
        else:
            action = np.random.randint(0, 4)
        return action

    def update(self):
        """
        Since this is a perfect agent, we do not update any network nor save any models
        """
        pass

    def update_stable_networks(self):
        """
        Since this is a perfect agent, we do not update any network nor save any models
        """
        pass

    def save_model(self, name):
        """
        Since this is a perfect agent, we do not update any network nor save any models
        """
        pass
