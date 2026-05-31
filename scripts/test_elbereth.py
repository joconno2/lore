#!/usr/bin/env python3
"""Test the exact NLE prompt sequence for Elbereth engraving."""
import gymnasium as gym
import nle.nethack as nethack

env = gym.make("NetHackChallenge-v0",
    observation_keys=("glyphs", "blstats", "message", "misc",
                      "inv_strs", "inv_letters", "inv_oclasses"),
    character="val-hum-fem-neu",
    max_episode_steps=500,
    no_progress_timeout=500,
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

ENGRAVE = 36
SEARCH = 75
LOOK = 51
EAST = 2
SPACE = char_idx(" ")
CR = char_idx("\r")

obs, info = env.reset(seed=42)

# Move off starting stairs
for _ in range(5):
    obs, r, term, trunc, info = env.step(EAST)
obs, r, term, trunc, info = env.step(SEARCH)
bl = obs["blstats"]
print(f"Position: ({int(bl[1])},{int(bl[0])})")

# Step-by-step engrave with full misc tracking
print(f"\n=== Step 1: ENGRAVE ===")
obs, r, term, trunc, info = env.step(ENGRAVE)
msg = get_msg(obs)
misc = list(obs["misc"])
print(f"  msg: {msg!r}")
print(f"  misc: {misc}  (wait={misc[0]} getlin={misc[1]} yn={misc[2]})")

print(f"\n=== Step 2: Send '-' ===")
obs, r, term, trunc, info = env.step(char_idx("-"))
msg = get_msg(obs)
misc = list(obs["misc"])
print(f"  msg: {msg!r}")
print(f"  misc: {misc}")

# Now step through ALL intermediate states
for i in range(20):
    # If in_yn, answer yes/no based on content
    if misc[2]:
        if "add to" in msg.lower():
            print(f"\n=== Step {i+3}: yn 'add to' -> 'n' ===")
            obs, r, term, trunc, info = env.step(char_idx("n"))
        else:
            print(f"\n=== Step {i+3}: yn prompt -> space ===")
            obs, r, term, trunc, info = env.step(SPACE)
        msg = get_msg(obs)
        misc = list(obs["misc"])
        print(f"  msg: {msg!r}")
        print(f"  misc: {misc}")
        continue

    # If xwaitingforspace, press space
    if misc[0]:
        print(f"\n=== Step {i+3}: waiting for space -> space ===")
        obs, r, term, trunc, info = env.step(SPACE)
        msg = get_msg(obs)
        misc = list(obs["misc"])
        print(f"  msg: {msg!r}")
        print(f"  misc: {misc}")
        continue

    # If in_getlin, we're ready to type
    if misc[1]:
        print(f"\n=== Step {i+3}: in_getlin, typing Elbereth ===")
        for ch in "Elbereth":
            obs, r, term, trunc, info = env.step(char_idx(ch))
            m2 = list(obs["misc"])
            msg2 = get_msg(obs)
            if msg2:
                print(f"  after '{ch}': msg={msg2!r} misc={m2}")
        # Send CR
        obs, r, term, trunc, info = env.step(CR)
        msg = get_msg(obs)
        misc = list(obs["misc"])
        print(f"  after CR: msg={msg!r} misc={misc}")

        # Verify
        print(f"\n=== Verify: LOOK ===")
        obs, r, term, trunc, info = env.step(LOOK)
        msg = get_msg(obs)
        misc = list(obs["misc"])
        print(f"  msg: {msg!r}")
        print(f"  misc: {misc}")
        # Handle --More--
        while "--More--" in msg or misc[0]:
            obs, r, term, trunc, info = env.step(SPACE)
            msg = get_msg(obs)
            misc = list(obs["misc"])
            print(f"  more: {msg!r} misc={misc}")
        break

    # No flags set, try pressing space or check message
    if "--More--" in msg:
        print(f"\n=== Step {i+3}: --More-- -> space ===")
        obs, r, term, trunc, info = env.step(SPACE)
        msg = get_msg(obs)
        misc = list(obs["misc"])
        print(f"  msg: {msg!r}")
        print(f"  misc: {misc}")
        continue

    print(f"\n=== Step {i+3}: no flags, no --More--. Breaking. ===")
    print(f"  msg: {msg!r}")
    print(f"  misc: {misc}")
    break

env.close()
print("\nDone.")
