#!/usr/bin/env python3
"""Test the exact NLE prompt sequence for Elbereth engraving."""
import gymnasium as gym
import nle.nethack as nethack

env = gym.make("NetHackScore-v0",
    observation_keys=("glyphs", "blstats", "message", "misc",
                      "inv_strs", "inv_letters", "inv_oclasses"),
    actions=nethack.ACTIONS,
    character="val-hum-fem-neu",
    max_episode_steps=500,
)

act_list = list(nethack.ACTIONS)

def char_idx(ch):
    t = ord(ch)
    for i, a in enumerate(act_list):
        if int(a) == t:
            return i
    return None

def get_msg(obs):
    return bytes(obs["message"]).rstrip(b"\x00").decode("latin-1").strip()

ENGRAVE = 36   # Command.ENGRAVE
SEARCH = 75    # Command.SEARCH
LOOK = 51      # Command.LOOK
EAST = 2       # CompassDirection.E

obs, info = env.reset(seed=42)

# Move off starting stairs
for _ in range(5):
    obs, r, term, trunc, info = env.step(EAST)

# Clear messages
obs, r, term, trunc, info = env.step(SEARCH)
bl = obs["blstats"]
print(f"Position: ({int(bl[1])},{int(bl[0])})")

# Try ENGRAVE
obs, r, term, trunc, info = env.step(ENGRAVE)
msg = get_msg(obs)
misc = list(obs["misc"])
print(f"\n1. ENGRAVE: {msg!r}  misc={misc}")

if "What do you want to write with" in msg:
    # Send '-' for fingers
    obs, r, term, trunc, info = env.step(char_idx("-"))
    msg = get_msg(obs)
    misc = list(obs["misc"])
    print(f"2. Dash: {msg!r}  misc={misc}")

    # Handle --More-- and misc[0] (xwaitingforspace)
    while "--More--" in msg or misc[0]:
        obs, r, term, trunc, info = env.step(char_idx(" "))
        msg = get_msg(obs)
        misc = list(obs["misc"])
        print(f"   More: {msg!r}  misc={misc}")

    # Handle "add to current engraving"
    if "add to" in msg.lower():
        obs, r, term, trunc, info = env.step(char_idx("n"))
        msg = get_msg(obs)
        misc = list(obs["misc"])
        print(f"   Add->n: {msg!r}  misc={misc}")

    print(f"\n   in_getlin={misc[1]}  Ready to type.")

    if misc[1]:
        # Type Elbereth character by character
        for ch in "Elbereth":
            obs, r, term, trunc, info = env.step(char_idx(ch))
            misc = list(obs["misc"])

        # Send CR to finish
        obs, r, term, trunc, info = env.step(19)  # CR/MORE
        msg = get_msg(obs)
        misc = list(obs["misc"])
        print(f"3. After typing+CR: {msg!r}  misc={misc}")

        # Verify with LOOK
        obs, r, term, trunc, info = env.step(LOOK)
        msg = get_msg(obs)
        print(f"4. LOOK: {msg!r}")

        while "--More--" in msg:
            obs, r, term, trunc, info = env.step(char_idx(" "))
            msg = get_msg(obs)
            print(f"   More: {msg!r}")
    else:
        print(f"   NOT in text entry mode. msg={msg!r}")
else:
    print(f"\n   ENGRAVE rejected: {msg!r}")

env.close()
print("\nDone.")
