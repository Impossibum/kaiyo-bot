from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState
from rlgym.utils.common_values import BALL_RADIUS


class BallTouchGroundCondition(TerminalCondition):
    """
    A condition that will terminate an episode after ball touches ground
    """

    def __init__(self, min_steps=50):
        super().__init__()
        self.min_steps = min_steps
        self.steps = 0

    def reset(self, initial_state: GameState):
        self.steps = 0
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        """
        return True if ball is touching the ground and it has been minimum number of steps
        """
        self.steps += 1
        if self.steps > self.min_steps:
            return current_state.ball.position[2] < (2 * BALL_RADIUS)
        else:
            return False
