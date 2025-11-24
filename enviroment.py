"""
enviroment.py

Defines the simulation enviroment and abstracts classes required for demonstrating EKF-SLAM. This
includes the visuals, enviroment layouts, predefined agent-movement patterns, and landmark handling.
"""

class EKF_SLAM:
    """Base class EKF-SLAM based on observations."""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def observation(self, observation: dict) -> str:
        """Process a given obersvation and updates precepts.

        Args:
            observation (dict): A dictionary containing information about
                the environment, such as the robot position and leaf distance.

        Returns:
            str: The chosen action, typically one of {"left", "right", "stay"}.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError
    
