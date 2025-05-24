import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.collections import LineCollection
import os

class EnvironmentVisualizer:
    """
    Visualizes the audio environment, including room boundaries, source locations,
    and agent trajectories during episodes.
    """
    def __init__(self, save_dir='../visualizations', acceptable_radius=1.0):
        """
        Initialize the visualizer.
        
        Args:
            save_dir (str): Directory to save visualization images
            acceptable_radius (float): Radius around sources where agent can detect them
        """
        self.save_dir = save_dir
        self.acceptable_radius = acceptable_radius
        self.trajectories = []  # List to store agent trajectories
        self.current_trajectory = []  # Current episode trajectory
        self.source_locations = []  # List of source locations
        self.room_boundaries = None  # Room boundaries
        self.episode_counter = 0  # To track episode number
        self.detection_events = []  # List of (position, step, source_index) tuples for detection events
        
        # Create the save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def reset(self, room_boundaries=None, source_locs=None, acceptable_radius=None):
        """
        Reset the visualizer for a new episode.
        
        Args:
            room_boundaries: The boundaries of the room (x_min, y_min, x_max, y_max) or corners
            source_locs: List of source locations [[x1, y1], [x2, y2], ...]
            acceptable_radius: Radius around sources where agent can detect them
        """
        self.current_trajectory = []
        self.detection_events = []
        
        if room_boundaries is not None:
            self.room_boundaries = room_boundaries
            
        if source_locs is not None:
            self.source_locations = source_locs
            
        if acceptable_radius is not None:
            self.acceptable_radius = acceptable_radius
            
        self.episode_counter += 1
        
    def update(self, agent_location, agent_angle=None):
        """
        Update the visualizer with the agent's current position.
        
        Args:
            agent_location: [x, y] coordinates of the agent
            agent_angle: Orientation angle of the agent in radians (optional)
        """
        step = len(self.current_trajectory)
        
        if agent_angle is not None:
            # Store position, orientation and step number
            self.current_trajectory.append((agent_location.copy(), agent_angle, step))
        else:
            # Store position and step number
            self.current_trajectory.append((agent_location.copy(), step))
    
    def record_detection(self, agent_position, step_number, source_index, source_position=None):
        """
        Record when the agent detects a source.
        
        Args:
            agent_position: [x, y] coordinates of the agent when detection occurred
            step_number: The step number when detection occurred
            source_index: Index of the source that was detected
            source_position: [x, y] coordinates of the source that was detected (optional)
        """
        if source_position is not None:
            # Store agent position, step number, source index, and source position
            self.detection_events.append((agent_position.copy(), step_number, source_index, source_position.copy()))
        else:
            # Backward compatibility - store without source position
            self.detection_events.append((agent_position.copy(), step_number, source_index))
            
    def end_episode(self, episode_summary=None):
        """
        End the current episode and save the trajectory.
        
        Args:
            episode_summary (dict, optional): Information about why the episode ended,
                                              including episode number, won status, final step,
                                              sources found, and end reason.
        """
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            self.episode_summary = episode_summary
            self.visualize_episode(save=True)
            
    def visualize_episode(self, save=False):
        """
        Visualize the current episode, showing room, sources, and agent trajectory.
        
        Args:
            save (bool): Whether to save the visualization to a file
        """
        plt.figure(figsize=(12, 10))
        
        # Set up figure title with episode summary if available
        title = "Agent Trajectory and Source Detections"
        if hasattr(self, 'episode_summary') and self.episode_summary is not None:
            summary = self.episode_summary
            title = f"Episode {summary['episode']} - {summary['end_reason']}\n"
            title += f"Found {summary['sources_found']}/{summary['total_sources']} sources in {summary['step']} steps"
        
        plt.suptitle(title, fontsize=14, y=0.98)
        
        # Plot room boundaries
        if self.room_boundaries is not None:
            if len(self.room_boundaries) == 4:  # (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = self.room_boundaries
                plt.plot([x_min, x_max, x_max, x_min, x_min], 
                         [y_min, y_min, y_max, y_max, y_min], 'k-', linewidth=2)
            else:  # Room corners as arrays of x and y coordinates
                plt.plot(np.append(self.room_boundaries[0], self.room_boundaries[0][0]),
                         np.append(self.room_boundaries[1], self.room_boundaries[1][0]), 'k-', linewidth=2)
        
        # Plot source locations with detection radius circles
        for i, source in enumerate(self.source_locations):
            # Draw detection radius circle
            detection_circle = plt.Circle((source[0], source[1]), self.acceptable_radius, 
                                         color='r', alpha=0.2, label='Detection Zone' if i == 0 else "")
            plt.gca().add_patch(detection_circle)
            
            # Draw source marker
            plt.plot(source[0], source[1], 'ro', markersize=10, label=f'Source {i+1}' if i == 0 else "")
            plt.text(source[0] + 0.1, source[1] + 0.1, f'S{i+1}', fontsize=12)
        
        # Plot agent trajectory
        if self.current_trajectory:
            # Check trajectory data format
            has_orientation = len(self.current_trajectory[0]) >= 2
            has_step_info = len(self.current_trajectory[0]) >= 3
            
            # Extract positions from trajectory data
            if has_orientation:
                positions = np.array([pos for pos, *_ in self.current_trajectory])
            else:
                positions = np.array(self.current_trajectory)
            
            # Create colormap for trajectory to show progression over time
            cmap = plt.cm.Blues
            norm = plt.Normalize(0, len(positions))
            
            # Plot trajectory segments with color gradient
            for i in range(len(positions) - 1):
                plt.plot(positions[i:i+2, 0], positions[i:i+2, 1], '-', 
                        color=cmap(norm(i)), alpha=0.8, linewidth=1.5)
            
            # Plot starting position
            plt.plot(positions[0, 0], positions[0, 1], 'gs', markersize=8, label='Start')
            
            # Plot ending position with label that includes step number
            end_label = 'End'
            if hasattr(self, 'episode_summary') and self.episode_summary is not None:
                end_label = f"End (Step {self.episode_summary['step']})"
            plt.plot(positions[-1, 0], positions[-1, 1], 'r*', markersize=12, label=end_label)
            
            # Plot step markers at intervals
            if has_step_info:
                interval = max(1, len(positions) // 10)  # Show up to 10 step markers
                for i in range(0, len(positions), interval):
                    if i < len(positions):
                        pos = positions[i]
                        step = self.current_trajectory[i][-1]  # Get step number
                        plt.text(pos[0], pos[1], f'{step}', fontsize=8, 
                                ha='center', va='center', 
                                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            
            # Plot detection events (if any)
            for i, event in enumerate(self.detection_events):
                if len(event) == 4:  # New format with source position
                    agent_pos, step, src_idx, src_pos = event
                    
                    # Draw agent position at detection
                    plt.plot(agent_pos[0], agent_pos[1], 'yo', markersize=10, alpha=0.8)
                    
                    # Draw source position that was detected
                    plt.plot(src_pos[0], src_pos[1], 'ro', markersize=10, alpha=0.8)
                    
                    # Draw line connecting agent to source at detection
                    plt.plot([agent_pos[0], src_pos[0]], [agent_pos[1], src_pos[1]], 'y--', alpha=0.6)
                    
                    # Determine if this is the last detection before episode end
                    is_last_detection = (i == len(self.detection_events) - 1)
                    if hasattr(self, 'episode_summary') and self.episode_summary is not None:
                        if is_last_detection and self.episode_summary['end_reason'] == "All sources found":
                            bbox_color = 'lime'
                            extra_text = "\n(Episode End)"
                        else:
                            bbox_color = 'yellow'
                            extra_text = ""
                    else:
                        bbox_color = 'yellow'
                        extra_text = ""
                    
                    # Add text annotation
                    plt.text(agent_pos[0], agent_pos[1] + 0.3, 
                            f'Found S{src_idx+1}\nStep {step}{extra_text}', 
                            fontsize=10, ha='center', va='center',
                            bbox=dict(facecolor=bbox_color, alpha=0.7, boxstyle='round'))
                else:  # Old format without source position
                    pos, step, src_idx = event
                    plt.plot(pos[0], pos[1], 'yo', markersize=12, alpha=0.8)
                    plt.text(pos[0], pos[1] + 0.2, f'Found S{src_idx+1}\nStep {step}', 
                            fontsize=10, ha='center', va='center',
                            bbox=dict(facecolor='yellow', alpha=0.7, boxstyle='round'))
                
            # Indicate direction with arrows at intervals if orientation data available
            if has_orientation and len(self.current_trajectory[0]) >= 2:
                interval = max(1, len(positions) // 20)  # Show up to 20 direction indicators
                for i in range(0, len(positions), interval):
                    if i < len(positions):
                        pos = positions[i]
                        angle = self.current_trajectory[i][1]  # Get angle
                        # Create a small arrow indicating direction
                        dx = 0.2 * np.cos(angle)
                        dy = 0.2 * np.sin(angle)
                        plt.arrow(pos[0], pos[1], dx, dy, head_width=0.1, head_length=0.1, 
                                 fc='blue', ec='blue', alpha=0.7)
        
        # Set plot limits with some padding
        if self.room_boundaries is not None:
            if len(self.room_boundaries) == 4:
                x_min, y_min, x_max, y_max = self.room_boundaries
                plt.xlim(x_min - 0.5, x_max + 0.5)
                plt.ylim(y_min - 0.5, y_max + 0.5)
            else:
                x_coords = self.room_boundaries[0]
                y_coords = self.room_boundaries[1]
                plt.xlim(min(x_coords) - 0.5, max(x_coords) + 0.5)
                plt.ylim(min(y_coords) - 0.5, max(y_coords) + 0.5)
        
        # Add legend, grid, and labels
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        
        # Add detection event details to title if any
        if self.detection_events:
            # Handle both 3-tuple and 4-tuple formats for detection events
            detection_strings = []
            for event in self.detection_events:
                if len(event) == 4:  # New format (agent_pos, step, src_idx, src_pos)
                    _, step, src_idx, _ = event
                else:  # Old format (agent_pos, step, src_idx)
                    _, step, src_idx = event
                detection_strings.append(f"S{src_idx+1} at step {step}")
            
            detections_str = ", ".join(detection_strings)
            plt.title(f'Agent Trajectory - Episode {self.episode_counter}\nSources found: {detections_str}')
        else:
            plt.title(f'Agent Trajectory - Episode {self.episode_counter}\nNo sources found')
        
        # Add summary box with episode details
        if hasattr(self, 'episode_summary') and self.episode_summary is not None:
            summary = self.episode_summary
            summary_text = f"Episode: {summary['episode']}\n"
            summary_text += f"Result: {'Success' if summary['won'] else 'Failure'}\n"
            summary_text += f"Sources Found: {summary['sources_found']}/{summary['total_sources']}\n"
            summary_text += f"Steps: {summary['step']}\n"
            summary_text += f"End Reason: {summary['end_reason']}"
            
            # Place text box in bottom right corner
            plt.gcf().text(0.95, 0.05, summary_text, 
                         fontsize=10, ha='right', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        if save:
            filename = os.path.join(self.save_dir, f'episode_{self.episode_counter:03d}.png')
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            print(f"Saved visualization to {filename}")
        
        plt.show()
        
    def animate_episode(self, interval=50, save=False):
        """
        Create an animation of the agent's movement during the episode.
        
        Args:
            interval (int): Time in milliseconds between frames
            save (bool): Whether to save the animation to a file
        """
        from matplotlib.animation import FuncAnimation
        
        if not self.current_trajectory:
            print("No trajectory data to animate")
            return
            
        # Check if trajectory includes orientation angles
        has_orientation = isinstance(self.current_trajectory[0], tuple)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot room boundaries
        if self.room_boundaries is not None:
            if len(self.room_boundaries) == 4:
                x_min, y_min, x_max, y_max = self.room_boundaries
                ax.plot([x_min, x_max, x_max, x_min, x_min], 
                       [y_min, y_min, y_max, y_max, y_min], 'k-', linewidth=2)
                ax.set_xlim(x_min - 0.5, x_max + 0.5)
                ax.set_ylim(y_min - 0.5, y_max + 0.5)
            else:
                ax.plot(np.append(self.room_boundaries[0], self.room_boundaries[0][0]),
                       np.append(self.room_boundaries[1], self.room_boundaries[1][0]), 'k-', linewidth=2)
                x_coords = self.room_boundaries[0]
                y_coords = self.room_boundaries[1]
                ax.set_xlim(min(x_coords) - 0.5, max(x_coords) + 0.5)
                ax.set_ylim(min(y_coords) - 0.5, max(y_coords) + 0.5)
        
        # Plot source locations
        for i, source in enumerate(self.source_locations):
            ax.plot(source[0], source[1], 'ro', markersize=10)
            ax.text(source[0] + 0.1, source[1] + 0.1, f'S{i+1}', fontsize=12)
        
        # Initialize empty line for trajectory
        if has_orientation:
            positions = np.array([pos for pos, _ in self.current_trajectory])
        else:
            positions = np.array(self.current_trajectory)
            
        line, = ax.plot([], [], 'b-', alpha=0.6, linewidth=1.5)
        
        # Agent marker (will be updated in animation)
        agent_marker, = ax.plot([], [], 'bo', markersize=8)
        
        # Direction indicator (if orientation data available)
        if has_orientation:
            direction_arrow = ax.arrow(0, 0, 0, 0, head_width=0.1, head_length=0.1, 
                                      fc='blue', ec='blue', alpha=0.7)
        
        # Set up the animation
        def init():
            line.set_data([], [])
            agent_marker.set_data([], [])
            return line, agent_marker
        
        def update(frame):
            # Update trajectory line
            if has_orientation:
                x_data = [pos[0] for pos, _ in self.current_trajectory[:frame+1]]
                y_data = [pos[1] for pos, _ in self.current_trajectory[:frame+1]]
                
                # Update agent position
                pos, angle = self.current_trajectory[frame]
                agent_marker.set_data([pos[0]], [pos[1]])
                
                # Update direction arrow
                try:
                    direction_arrow.remove()
                except:
                    pass
                dx = 0.2 * np.cos(angle)
                dy = 0.2 * np.sin(angle)
                direction_arrow = ax.arrow(pos[0], pos[1], dx, dy, head_width=0.1, head_length=0.1, 
                                         fc='blue', ec='blue', alpha=0.7)
            else:
                x_data = [pos[0] for pos in self.current_trajectory[:frame+1]]
                y_data = [pos[1] for pos in self.current_trajectory[:frame+1]]
                
                # Update agent position
                pos = self.current_trajectory[frame]
                agent_marker.set_data([pos[0]], [pos[1]])
            
            line.set_data(x_data, y_data)
            
            return line, agent_marker
        
        # Create animation
        num_frames = len(self.current_trajectory)
        anim = FuncAnimation(fig, update, frames=range(num_frames),
                             init_func=init, blit=True, interval=interval)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_title(f'Agent Movement - Episode {self.episode_counter}')
        
        if save:
            filename = os.path.join(self.save_dir, f'animation_episode_{self.episode_counter:03d}.mp4')
            anim.save(filename, writer='ffmpeg', fps=30, dpi=200)
            print(f"Saved animation to {filename}")
        
        plt.show()
        
    def visualize_all_episodes(self, save=False):
        """
        Visualize all recorded episodes in a single plot.
        
        Args:
            save (bool): Whether to save the visualization to a file
        """
        plt.figure(figsize=(12, 10))
        
        # Plot room boundaries
        if self.room_boundaries is not None:
            if len(self.room_boundaries) == 4:
                x_min, y_min, x_max, y_max = self.room_boundaries
                plt.plot([x_min, x_max, x_max, x_min, x_min], 
                         [y_min, y_min, y_max, y_max, y_min], 'k-', linewidth=2)
                plt.xlim(x_min - 0.5, x_max + 0.5)
                plt.ylim(y_min - 0.5, y_max + 0.5)
            else:
                plt.plot(np.append(self.room_boundaries[0], self.room_boundaries[0][0]),
                         np.append(self.room_boundaries[1], self.room_boundaries[1][0]), 'k-', linewidth=2)
                x_coords = self.room_boundaries[0]
                y_coords = self.room_boundaries[1]
                plt.xlim(min(x_coords) - 0.5, max(x_coords) + 0.5)
                plt.ylim(min(y_coords) - 0.5, max(y_coords) + 0.5)
        
        # Plot source locations (using the most recent ones)
        for i, source in enumerate(self.source_locations):
            plt.plot(source[0], source[1], 'ro', markersize=10, label=f'Source {i+1}' if i == 0 else "")
            plt.text(source[0] + 0.1, source[1] + 0.1, f'S{i+1}', fontsize=12)
        
        # Plot trajectories for all episodes with different colors
        colors = plt.cm.jet(np.linspace(0, 1, len(self.trajectories)))
        
        for i, trajectory in enumerate(self.trajectories):
            # Check if trajectory includes orientation angles
            if isinstance(trajectory[0], tuple):
                positions = np.array([pos for pos, _ in trajectory])
            else:
                positions = np.array(trajectory)
            
            plt.plot(positions[:, 0], positions[:, 1], '-', color=colors[i], alpha=0.7, 
                    linewidth=1.5, label=f'Episode {i+1}')
            
            # Mark start and end points
            plt.plot(positions[0, 0], positions[0, 1], 'o', color=colors[i], markersize=6)
            plt.plot(positions[-1, 0], positions[-1, 1], '*', color=colors[i], markersize=10)
        
        # Add legend, grid, and labels
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title(f'Agent Trajectories - All Episodes')
        
        if save:
            filename = os.path.join(self.save_dir, 'all_episodes.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {filename}")
        
        plt.show()
