"""Capture the full traceback of the deterministic AssertionError on a deep seed."""
import sys, traceback
import gym
from autoascend.env_wrapper import EnvWrapper

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 138
env = EnvWrapper(gym.make("NetHackChallenge-v0", no_progress_timeout=1000),
                 agent_args=dict(panic_on_errors=False, verbose=False))
env.env.seed(seed, seed)
try:
    env.main()
    print("NO CRASH; end_reason:", env.end_reason)
except BaseException as e:
    print("=== EXCEPTION ===", type(e).__name__, flush=True)
    traceback.print_exc()
    s = env.get_summary()
    print("=== at score", s.get("score"), "DL", s.get("level_num"), "turns", s.get("turns"))
