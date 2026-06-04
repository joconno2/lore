"""Baseline eval for agent2 (v2)."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/trx/nethack-aall/lore")
import nle.nethack as nh
from nhc.elbereth_env import NetHackScoreEngrave
import gymnasium as gym
gym.register(id="NetHackScoreEngrave-v0", entry_point="nhc.elbereth_env:NetHackScoreEngrave",
    kwargs={"character": "val-hum-law-fem", "max_episode_steps": 1_000_000,
            "penalty_step": 0.0, "observation_keys": ("glyphs", "blstats", "message",
            "inv_strs", "inv_letters", "inv_oclasses", "misc"),
            "allow_all_yn_questions": True, "actions": tuple(nh.ACTIONS)})
from nhc.agent2 import AgentV2
N = 10
scores = []
for ep in range(N):
    seed = 42 + ep
    t0 = time.time()
    try:
        env = gym.make("NetHackScoreEngrave-v0")
        agent = AgentV2(env, seed=seed)
        score = agent.main()
        elapsed = time.time() - t0
        turn = agent.blstats.time if agent.blstats else 1
        spt = agent.step_count / max(1, turn)
        scores.append(score)
        print(f"ep={ep} seed={seed} score={score:.0f} spt={spt:.2f} time={elapsed:.1f}s")
        env.close()
    except Exception as e:
        print(f"ep={ep} seed={seed} ERROR: {e}")
        try: env.close()
        except: pass
if scores:
    arr = np.array(scores)
    print(f"\nmean={arr.mean():.1f} median={np.median(arr):.1f} max={arr.max():.0f} min={arr.min():.0f}")
