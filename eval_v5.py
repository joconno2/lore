"""Quick eval for agent2_v4 expert system."""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/trx/nethack-aall/lore")

from nhc.elbereth_env import NetHackScoreEngrave
import gymnasium as gym

# Register the env
import nle.nethack as nh

gym.register(
    id="NetHackScoreEngrave-v0",
    entry_point="nhc.elbereth_env:NetHackScoreEngrave",
    kwargs={
        "character": "val-hum-law-fem",
        "max_episode_steps": 1_000_000,
        "penalty_step": 0.0,
        "observation_keys": (
            "glyphs", "blstats", "message",
            "inv_strs", "inv_letters", "inv_oclasses",
            "misc",
        ),
        "allow_all_yn_questions": True,
        "actions": tuple(nh.ACTIONS),
    },
)

from nhc.agent2_v5 import AgentV5 as AgentV4

N_EPISODES = 10
scores = []
steps_list = []
errors = []

for ep in range(N_EPISODES):
    seed = 42 + ep
    t0 = time.time()
    try:
        env = gym.make("NetHackScoreEngrave-v0")
        agent = AgentV4(env, seed=seed, verbose=True)
        score = agent.main()
        if score is None:
            score = agent.score
        elapsed = time.time() - t0
        turn = agent.blstats.time if agent.blstats else 1
        spt = agent.step_count / max(1, turn)
        scores.append(score)
        steps_list.append(agent.step_count)
        print(f"ep={ep} seed={seed} score={score:.0f} steps={agent.step_count} "
              f"spt={spt:.2f} time={elapsed:.1f}s")
        env.close()
    except Exception as e:
        elapsed = time.time() - t0
        errors.append((ep, str(e)))
        print(f"ep={ep} seed={seed} ERROR: {e} time={elapsed:.1f}s")
        try:
            env.close()
        except:
            pass

print(f"\n--- Results ({len(scores)}/{N_EPISODES} completed) ---")
if scores:
    arr = np.array(scores)
    print(f"mean={arr.mean():.1f} median={np.median(arr):.1f} "
          f"max={arr.max():.0f} min={arr.min():.0f} std={arr.std():.1f}")
if errors:
    print(f"errors: {len(errors)}")
    for ep, e in errors:
        print(f"  ep={ep}: {e[:200]}")
