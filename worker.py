import sys
from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from rlgym.envs import Match
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
from N_Parser import NectoAction
from kaiyo_rewards import KaiyoRewards
from torch import set_num_threads
import os
set_num_threads(1)


if __name__ == "__main__":
    rew = KaiyoRewards()
    frame_skip = 8
    fps = 120 // frame_skip
    name = "Default"
    send_gamestate = False
    local = True
    host = "127.0.0.1"
    if len(sys.argv) > 1:
        host = sys.argv[1]
        if host != "127.0.0.1" and host != "localhost":
            local = False
    if len(sys.argv) > 2:
        name = sys.argv[2]
    if len(sys.argv) > 3:
        if sys.argv[3] == 'GAMESTATE':
            send_gamestate = True
    # ts_index = int(sys.argv[1])
    # team_sizes = [1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 2, 1, 3, 1, 2, 3, 1, 2, 1, 3, 1, 2, 1, 3, 2, 1, 3, 2, 1, 1, 3, 2]

    replay_options = ["ssl_3v3.npy"]

    match = Match(
        game_speed=100,
        spawn_opponents=True,
        team_size=3,
        state_setter=WeightedSampleSetter((
            DefaultState(),
            AugmentSetter(
                ReplaySetter(replay_options[0])
            )),
            (1, 0)),
        obs_builder=AdvancedObsPadder(team_size=3, expanding=True),
        action_parser=NectoAction(),
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
        reward_function=rew
    )

    # local Redis
    if local:
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  )

    # remote Redis
    else:
        # noinspection PyArgumentList
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  retry_on_error=[ConnectionError, TimeoutError],
                  retry=Retry(ExponentialBackoff(cap=10, base=1), 25)
                  )
    RedisRolloutWorker(r, name, match,
                       past_version_prob=0.2,
                       sigma_target=2,
                       evaluation_prob=0.01,
                       force_paging=True,
                       dynamic_gm=True,
                       send_obs=True,
                       auto_minimize=True,
                       send_gamestates=send_gamestate,
                       ).run()


