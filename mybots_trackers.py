import numpy as np

from rocket_learn.utils.gamestate_encoding import StateConstants
from rocket_learn.utils.stat_trackers.stat_tracker import StatTracker


class AirTouch(StatTracker):
    def __init__(self):
        super().__init__("air_touch_rate")
        self.count = 0
        self.total_touches = 0

    def reset(self):
        self.count = 0
        self.total_touches = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        is_touch = np.asarray([a * (not b) for a, b in
                              zip(players[:, StateConstants.BALL_TOUCHED], players[:, StateConstants.ON_GROUND])])

        self.total_touches += np.sum(is_touch)
        self.count += is_touch.size

    def get_stat(self):
        return self.total_touches / (self.count or 1)


class AirTouchHeight(StatTracker):
    def __init__(self):
        super().__init__("air_touch_height")
        self.count = 0
        self.total_height = 0

    def reset(self):
        self.count = 0
        self.total_height = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        ball_z = gamestates[:, StateConstants.BALL_POSITION.start + 2]
        touch_heights = ball_z[players[:, StateConstants.BALL_TOUCHED].any(axis=1)]
        touch_heights = touch_heights[touch_heights >= 175]  # remove dribble touches and below

        self.total_height += np.sum(touch_heights)
        self.count += touch_heights.size

    def get_stat(self):
        return self.total_height / (self.count or 1)
