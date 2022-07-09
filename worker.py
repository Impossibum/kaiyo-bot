import sys
from redis import Redis
from rlgym.envs import Match
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker
from N_Parser import NectoAction
from kaiyo_rewards import KaiyoRewards
from torch import set_num_threads
set_num_threads(1)


if __name__ == "__main__":
    rew = KaiyoRewards()
    frame_skip = 8
    fps = 120 / frame_skip
    ts_index = int(sys.argv[1])
    team_sizes = [1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 2, 1, 3, 1, 2, 3, 1, 2, 1, 3, 1, 2, 1, 3, 2, 1, 3, 2, 1, 1, 3, 2]

    match = Match(
        game_speed=100,
        self_play=True,
        team_size=team_size,
        state_setter=DefaultState(),
        obs_builder=AdvancedObsPadder(team_size=3, expanding=True),
        action_parser=NectoAction(),
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
        reward_function=rew
    )

    r = Redis(password="password")
    RedisRolloutWorker(r, "kaiyo", match, past_version_prob=0.2, sigma_target=2,
                       send_gamestates=False, evaluation_prob=0.01, force_paging=True, deterministic_old_prob=0.5).run()


