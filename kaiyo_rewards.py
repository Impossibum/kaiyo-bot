from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rewards import JumpTouchReward, TouchVelChange, OmniBoostDiscipline


class KaiyoRewards(RewardFunction):
    def __init__(self):
        super().__init__()
        self.goal_weight = 10
        self.boost_weight = 1.5
        self.demo_weight = 3
        self.boost_disc_weight = self.boost_weight * 0.02223
        self.reward = CombinedReward(
            (
             TouchVelChange(),
             VelocityPlayerToBallReward(),
             VelocityBallToGoalReward(),
             # EventReward(goal=5.0, concede=-5.0)  #  replace this event reward with one below after basic proficiency is gained
             EventReward(team_goal=self.goal_weight, concede=-self.goal_weight, demo=self.demo_weight, boost_pickup=self.boost_weight),  # 1.0
             JumpTouchReward(min_height=120),  # 2.0  - add when introducing new event reward
             #  OmniBoostDiscipline()  self.boost_disc_weight - Don't add until solid game mechanics learned and boost abuse is observed

             ),
            (0.35, 0, 0.05, 1.0, 2.0))

    def reset(self, initial_state: GameState) -> None:
        self.reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.reward.get_reward(player, state, previous_action)


