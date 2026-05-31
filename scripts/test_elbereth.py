#!/usr/bin/env python3
"""Test the exact NLE prompt sequence for Elbereth engraving."""
import gymnasium as gym
import nle.nethack as nethack

env = gym.make("NetHackScore-v0",
    observation_keys=("glyphs", "blstats", "message", "inv_strs", "inv_letters", "inv_oclasses"),
    actions=nethack.ACTIONS,
    character="val-hum-fem-neu",
    max_episode_steps=200,
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

print("ENGRAVE idx:", lookup.get("ENGRAVE", "NOT FOUND"))

# Map Elbereth chars
for ch in "Elbereth-\r":
    idx = char_to_idx(ch)
    print(f"  {ch!r} (ord {ord(ch)}) -> idx {idx}")

obs, info = env.reset(seed=42)

# Try ENGRAVE
engrave_idx = lookup["ENGRAVE"]
obs, r, term, trunc, info = env.step(engrave_idx)
msg = bytes(obs["message"]).rstrip(b"\x00").decode("latin-1").strip()
print(f"\n1. After ENGRAVE: {msg!r}")

# Send dash (write with fingers)
obs, r, term, trunc, info = env.step(char_to_idx("-"))
msg = bytes(obs["message"]).rstrip(b"\x00").decode("latin-1").strip()
print(f"2. After '-': {msg!r}")

# Handle --More-- if present
while "--More--" in msg:
    obs, r, term, trunc, info = env.step(char_to_idx("\r"))
    msg = bytes(obs["message"]).rstrip(b"\x00").decode("latin-1").strip()
    print(f"   After MORE: {msg!r}")

# Type Elbereth one char at a time
for ch in "Elbereth":
    idx = char_to_idx(ch)
    obs, r, term, trunc, info = env.step(idx)
    msg = bytes(obs["message"]).rstrip(b"\x00").decode("latin-1").strip()
    if msg:
        print(f"3. After '{ch}': {msg!r}")

# Send Return to finish
obs, r, term, trunc, info = env.step(char_to_idx("\r"))
msg = bytes(obs["message"]).rstrip(b"\x00").decode("latin-1").strip()
print(f"4. After CR: {msg!r}")

# Check what happens next step
obs, r, term, trunc, info = env.step(lookup.get("LOOK", lookup.get("SEARCH")))
msg = bytes(obs["message"]).rstrip(b"\x00").decode("latin-1").strip()
print(f"5. After LOOK/SEARCH: {msg!r}")

env.close()
print("\nDone.")
