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
lookup = {}
for i, a in enumerate(act_list):
    name = a.name if hasattr(a, "name") else str(a)
    if name not in lookup:
        lookup[name] = i

def char_to_idx(ch):
    target = ord(ch)
    for i, a in enumerate(act_list):
        if int(a) == target:
            return i
    return None

def get_msg(obs):
    return bytes(obs["message"]).rstrip(b"\x00").decode("latin-1").strip()

def get_misc(obs):
    m = obs.get("misc", [0, 0, 0])
    return {"xwaitingforspace": bool(m[0]), "in_getlin": bool(m[1]), "in_yn": bool(m[2])}

# Print action mappings
print("=== Action mappings ===")
print(f"ENGRAVE: idx={lookup.get('ENGRAVE', '?')}")
for ch in "Elbereth-\ry":
    idx = char_to_idx(ch)
    print(f"  {ch!r} (ord {ord(ch)}) -> idx {idx}")

obs, info = env.reset(seed=42)
print(f"\n=== Starting game ===")
print(f"Message: {get_msg(obs)!r}")
print(f"Misc: {get_misc(obs)}")

# Take a few steps first to get past initial messages
for _ in range(3):
    obs, r, term, trunc, info = env.step(lookup.get("SEARCH", 22))

print(f"\nAfter 3 searches:")
print(f"Message: {get_msg(obs)!r}")
print(f"Misc: {get_misc(obs)}")

# Now try ENGRAVE
print(f"\n=== Step 1: ENGRAVE ===")
obs, r, term, trunc, info = env.step(lookup["ENGRAVE"])
msg = get_msg(obs)
misc = get_misc(obs)
print(f"Message: {msg!r}")
print(f"Misc: {misc}")

# If we got "What do you want to write with?", send '-'
if "write with" in msg.lower() or "engrave with" in msg.lower():
    print(f"\n=== Step 2: Send '-' (fingers) ===")
    obs, r, term, trunc, info = env.step(char_to_idx("-"))
    msg = get_msg(obs)
    misc = get_misc(obs)
    print(f"Message: {msg!r}")
    print(f"Misc: {misc}")

    # Handle --More-- prompts
    step = 2
    while misc["xwaitingforspace"] or "--More--" in msg:
        step += 1
        print(f"\n=== Step {step}: Clear --More-- ===")
        obs, r, term, trunc, info = env.step(char_to_idx(" "))
        msg = get_msg(obs)
        misc = get_misc(obs)
        print(f"Message: {msg!r}")
        print(f"Misc: {misc}")

    # Handle "Do you want to add to the current engraving?"
    if "add to" in msg.lower():
        step += 1
        print(f"\n=== Step {step}: Answer 'n' to add prompt ===")
        obs, r, term, trunc, info = env.step(char_to_idx("n"))
        msg = get_msg(obs)
        misc = get_misc(obs)
        print(f"Message: {msg!r}")
        print(f"Misc: {misc}")

    # Now we should be in text-entry mode (in_getlin)
    if misc["in_getlin"] or "write in the dust" in msg.lower() or "engrave" in msg.lower():
        print(f"\n=== Typing 'Elbereth' ===")
        for ch in "Elbereth":
            step += 1
            obs, r, term, trunc, info = env.step(char_to_idx(ch))
            msg = get_msg(obs)
            misc = get_misc(obs)
            if msg or not misc["in_getlin"]:
                print(f"  After '{ch}': msg={msg!r} misc={misc}")

        # Send CR to finish
        step += 1
        print(f"\n=== Step {step}: Send CR ===")
        obs, r, term, trunc, info = env.step(char_to_idx("\r"))
        msg = get_msg(obs)
        misc = get_misc(obs)
        print(f"Message: {msg!r}")
        print(f"Misc: {misc}")
    else:
        print(f"\nNot in text-entry mode. Current state:")
        print(f"  Message: {msg!r}")
        print(f"  Misc: {misc}")

    # Verify with LOOK
    print(f"\n=== Verify: LOOK ===")
    look_idx = None
    for i, a in enumerate(act_list):
        if int(a) == ord(":"):  # LOOK command is ':'
            look_idx = i
            break
    if look_idx:
        obs, r, term, trunc, info = env.step(look_idx)
        msg = get_msg(obs)
        print(f"Message: {msg!r}")
        # Clear any --More--
        while "--More--" in msg:
            obs, r, term, trunc, info = env.step(char_to_idx(" "))
            msg = get_msg(obs)
            print(f"  More: {msg!r}")
else:
    print(f"\nENGRAVE didn't trigger prompt. Got: {msg!r}")
    print("Trying to understand why...")
    # Check if the action was interpreted as something else
    print(f"ENGRAVE action value: {int(act_list[lookup['ENGRAVE']])}")
    print(f"That's ASCII: {chr(int(act_list[lookup['ENGRAVE']]))!r}")

env.close()
print("\nDone.")
