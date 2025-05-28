import mesa

class NavigatingAgent(mesa.Agent):
    """
    An agent that can navigate in the environment.
    """
    def __init__(self, unique_id, model, pos, angle):
        """
        Create a new navigating agent.

        Args:
            unique_id: The agent's unique ID.
            model: The model instance.
            pos: The agent's position (x, y) tuple.
            angle: The agent's orientation angle in radians.
        """
        super().__init__(unique_id, model)
        self.pos = pos
        self.angle = angle

class SourceAgent(mesa.Agent):
    """
    An agent representing a sound source.
    """
    def __init__(self, unique_id, model, pos, source_type='unknown'):
        """
        Create a new source agent.

        Args:
            unique_id: The agent's unique ID.
            model: The model instance.
            pos: The agent's position (x, y) tuple.
            source_type: The type of the sound source (e.g., 'phone', 'siren').
                         Defaults to 'unknown'.
        """
        super().__init__(unique_id, model)
        self.pos = pos
        self.source_type = source_type
