from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    Base class for all environments.
    """

    def __init__(self, server_path, platform, **kwargs):
        self.server_path = server_path
        self.platform = platform
        self.action_history = []

    @abstractmethod
    def reset(self, **kwargs):
        pass

    @abstractmethod
    def get_screen_size(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def get_screenshot(self):
        pass

    @abstractmethod
    def get_a11ytree(self):
        pass

    @abstractmethod
    def execute_single_action(self, action: dict):
        pass
