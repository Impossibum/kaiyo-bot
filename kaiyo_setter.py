from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter
from rlgym.utils import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.state_setters import DefaultState
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter


class KaiyoSetter(DynamicGMSetter):
    def __init__(self):
        self.setters = []  # [1v1, 2v2, 3v3]
        replays = ["ssl_1v1.npy", "ssl_2v2.npy", "ssl_3v3.npy"]
        for i in range(3):
            self.setters.append(
                WeightedSampleSetter(
                    (
                        DefaultState(),
                        AugmentSetter(ReplaySetter(replays[i]))
                    ),
                    (0.15, 0.85)
                )
            )

    def reset(self, state_wrapper: StateWrapper):
        self.setters[(len(state_wrapper.cars) // 2) - 1].reset(state_wrapper)
