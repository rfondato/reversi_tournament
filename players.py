import os
import uuid
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from boardgame2 import BoardGameEnv
from stable_baselines3 import PPO


class BasePlayer(ABC):
    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False,
                 name: str = None
                 ):
        self.id = uuid.uuid4()
        self.name = name if name is not None else self.__class__.__name__
        self.env = env
        self.player = player  # player number. 1 o -1
        self.flatten_action = flatten_action
        self.board_shape = self.env.board.shape[0]

    @abstractmethod
    def predict(self, board: np.ndarray) -> Union[int, np.ndarray]:
        """
        Returns the action to play given a board.
        :param board: Numpy array of board_shape x board_shape with current board
        :return: Numpy array of dimension 2 with row and column to play if flatten_action is False.
                If flatten_action is True, it returns an int with the slot number.
        """

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.id == other.id


class GreedyPlayer(BasePlayer):
    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False
                 ):
        super().__init__(player, env, flatten_action)

    def predict(self, board: np.ndarray) -> Union[int, np.ndarray]:
        valid_actions = np.argwhere(self.env.get_valid((board, self.player)) == 1)
        if len(valid_actions) == 0:
            action = self.env.PASS
        else:
            moves_score = []
            for a in valid_actions:
                next_state, _, _, _ = self.env.next_step((board, self.player), a)
                moves_score.append(next_state[0].sum() * self.player)
            best_score = max(moves_score)
            best_actions = valid_actions[np.array(moves_score) == best_score]
            action = best_actions[np.random.randint(len(best_actions))]
        if self.flatten_action:
            return action[0] * self.board_shape + action[1]
        else:
            return action


class RandomPlayer(BasePlayer):
    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False
                 ):
        super().__init__(player, env, flatten_action)

    def predict(self, board: np.ndarray) -> Union[int, np.ndarray]:
        valid_actions = np.argwhere(self.env.get_valid((board, self.player)) == 1)
        if len(valid_actions) == 0:
            action = self.env.PASS
        else:
            action = valid_actions[np.random.randint(len(valid_actions))]
        if self.flatten_action:
            return action[0] * self.board_shape + action[1]
        else:
            return action


class BaseTorchPlayer(BasePlayer, ABC):

    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False,
                 model_path: str = None,
                 deterministic: bool = True,
                 only_valid: bool = True,
                 device: str = 'auto'
                 ):

        if model_path is None:
            raise Exception("model_path cannot be None")

        super().__init__(player, env, flatten_action, os.path.splitext(os.path.basename(model_path))[0])

        self.model = PPO.load(model_path, device=device)
        self.model_path = model_path
        self.deterministic = deterministic
        self.device = device
        self.only_valid = only_valid
