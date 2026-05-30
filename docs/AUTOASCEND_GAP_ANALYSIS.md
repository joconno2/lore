# AutoAscend vs LORE Expert Agent: Gap Analysis

AutoAscend median 5,336. Our expert agent mean 115. ~46x difference.

This document identifies what AutoAscend does that we don't, estimates score impact per gap, and rates implementation difficulty. Gaps are ordered by estimated score impact, largest first.

---

## 1. Equipment Management

**What AutoAscend does:**
- `get_best_armorset()` evaluates every armor piece across 7 slots (suit, cloak, helm, gloves, boots, shirt, shield) by AC value. Selects the best piece per slot from all available items, including those on the ground.
- `wear_best_stuff()` strategy continuously swaps armor: takes off inferior pieces, handles layer ordering (cloak before suit, suit before shirt), checks for cursed items blocking removal.
- `get_best_melee_weapon()` computes DPS via `calc_dps(to_hit, damage)` using weapon skill bonuses, strength/dex modifiers, and weapon enchantment. Selects the highest-DPS weapon and wields it automatically.
- `get_best_ranged_set()` evaluates all launcher+ammo combinations and thrown projectiles. Wields launchers before firing.
- Full skill parsing: reads the `#enhance` screen, upgrades available skills every combat encounter.

**What we do:**
- Nothing. The expert agent operates within NetHackScore-v0's 23-action space. WIELD, WEAR, TAKEOFF, APPLY are all mapped to SEARCH (no-op). The agent cannot equip any weapon or armor.

**Score impact:** ~2,000-3,000 points. Armor reduces incoming damage dramatically (each AC point matters exponentially in early game). Wielding a real weapon doubles or triples melee DPS vs bare hands. This is the single largest gap.

**Difficulty:** Hard. Requires switching to a full action space (NetHackChallenge-v0 or similar). The 23-action NetHackScore-v0 environment physically cannot equip items.

---

## 2. Sokoban Solving

**What AutoAscend does:**
- Full Sokoban solver in `soko_solver.py`. Matches the current level layout against known Sokoban maps, computes boulder push sequences, handles mimic detection, and uses pick-axes to clear stuck boulders.
- Sokoban completion yields: bag of holding or amulet of reflection + ~500-1000 XP + cleared path for future travel.
- Milestone-driven: the agent specifically navigates to Sokoban, solves it, then continues.

**What we do:**
- Nothing. No Sokoban awareness. The agent has no concept of dungeon branches.

**Score impact:** ~500-1,000 points. The Sokoban prizes (reflection, bag of holding) are survival multipliers in mid-game. The XP from Sokoban monsters is significant at the levels you encounter it.

**Difficulty:** Hard. Requires dungeon branch detection, level memory across levels, boulder mechanics, and a puzzle solver. This is a complex planning problem.

---

## 3. Level Descent Strategy and Dungeon Navigation

**What AutoAscend does:**
- Milestone-driven progression through a defined sequence: Dungeons of Doom level 1 (until XL 8) -> Gnomish Mines -> Minetown -> Sokoban -> Mines End -> deeper Dungeons.
- `go_to_level_strategy()` tracks connections between levels, performs BFS over the level graph, and navigates through stair connections across multiple dungeon branches.
- Stays on level 1 until experience level 8, building strength before descending.
- Explores each level thoroughly before moving on: uses a priority system where tiles adjacent to unseen stone or closed doors get exploration priority.

**What we do:**
- Descend stairs immediately whenever HP > 40%. No level lingering. No branch awareness. No experience gating.
- Descend-first policy means the agent reaches dangerous depths at low experience levels with no equipment.

**Score impact:** ~1,500-2,500 points. Premature descending is likely the second biggest killer. An XL 2 character on DL 4 faces monsters scaled to DL 4 with XL 2 stats. AutoAscend waits until XL 8 before leaving DL 1.

**Difficulty:** Medium. The core fix (don't descend until XL N) is simple. Full milestone navigation is harder but the biggest win comes from just gating descent.

---

## 4. Combat Strategy

**What AutoAscend does:**
- Priority-based heatmap system in `fight_heur.py` and `movement_priority.py`. Each monster generates positive attraction zones (approach for melee/ranged) and negative repulsion zones (avoid if dangerous).
- Monster categorization: ONLY_RANGED_SLOW (floating eye, acid blob, etc.), EXPLODING (gas spore, yellow light), WEAK (lichen, newt), WEIRD (leprechaun, nymph). Each category gets different handling.
- Floating eyes: never melee. Only engage with ranged weapons. If no ranged, avoid entirely.
- Gas spores: strong negative priority at radius 1 (explosion). Engage from range.
- DPS calculation for weapon selection considers to-hit probability, damage dice, strength/dex bonuses, and weapon skill level.
- Elbereth engraving: writes "Elbereth" when surrounded and low HP. Waits on Elbereth tile while healing.
- Wand usage: simulates wand beam paths with bouncing, evaluates expected hit count per monster, fires when priority exceeds threshold.
- Ranged combat: evaluates all 8 directions for line-of-fire, picks highest-priority target considering distance penalty.

**What we do:**
- Step toward the closest hostile monster. Melee it. The `_StubThreatReport` provides basic danger classification (instakill risk, ranged-preferred) but doesn't affect weapon selection since no weapons can be wielded.
- Floating eyes: correctly identified as ranged-preferred, agent flees. But can't actually range-attack them.
- No Elbereth. ENGRAVE action maps to SEARCH in NetHackScore-v0.
- No wand usage, no ranged attacks (THROW/ZAP unavailable).

**Score impact:** ~500-1,000 points. The agent already handles basic melee correctly. The gap is in avoiding bad fights (floating eyes, gas spores) and in damage output (no weapons). Weapon DPS is counted under Equipment above.

**Difficulty:** Medium for combat logic improvements in a full action space. Easy for better flee/avoid heuristics in current action space.

---

## 5. Food Strategy

**What AutoAscend does:**
- Corpse eating system with extensive safety checking: `_is_corpse_editable()` checks poison, acid, lycanthropy, polymorph, hallucination, stun, aggravation, petrification, cannibalism, and corpse age (50 turn limit, except lizard/lichen).
- Tracks corpse locations and ages on every level via `corpses_to_eat` dict. Walks to fresh safe corpses when hungry.
- Eats from ground first (avoids picking up corpses, saving inventory space).
- Nutritional priority: food sorted by `nutrition_per_weight()`. Wolfsbane kept for curing lycanthropy, not eaten as food.
- Emergency food: prays when fainting (if prayer is safe), drinks fruit juice potions.

**What we do:**
- Eats when hungry from inventory (basic EAT action).
- Message-based corpse detection: checks for "corpse" in tile messages, eats if `_corpse_safe_to_eat()` returns true.
- Small unsafe-corpse blacklist (12 entries vs AutoAscend's ~30+ categories of checks).
- No corpse age tracking. No nutritional priority. No ground-eating.

**Score impact:** ~300-500 points. Food management prevents starvation deaths and provides intrinsic resistances. The current agent's blacklist misses several dangerous corpses (bats cause stun, dogs cause aggravation, mimics cause paralysis).

**Difficulty:** Easy-Medium. Expanding the corpse blacklist is easy. Corpse age tracking and ground-eating require more infrastructure.

---

## 6. Excalibur

**What AutoAscend does:**
- `dip_for_excalibur()` strategy: if lawful alignment AND experience level >= 5 AND carrying a long sword AND near a fountain, dips the long sword to create Excalibur.
- Excalibur is +1d10 damage, +5 to-hit, confers automatic searching, and is an artifact (bonus damage vs many monsters). Single biggest weapon upgrade available in early game.
- Item priority keeps long swords for lawful characters specifically for this purpose.
- Checks every 10 steps after XL 7, tries fountains on the current level.

**What we do:**
- Nothing. No fountain interaction. No item dipping. No awareness of Excalibur.

**Score impact:** ~300-800 points (conditional on lawful alignment, which is ~25% of random starts). When it fires, Excalibur roughly doubles combat effectiveness for the rest of the game.

**Difficulty:** Medium. Requires DIP action, fountain detection, and carrying a long sword. Conditional on role/alignment.

---

## 7. Altar Use and Sacrifice

**What AutoAscend does:**
- `offer_corpses()` strategy: carries fresh corpses to co-aligned altars and sacrifices them. Tracks altar alignment by examining with LOOK command.
- `identify_items_on_altar()`: drops unidentified items on altars to learn their BUC (blessed/uncursed/cursed) status. This gates all equipment decisions since cursed items can't be removed.
- Sacrifice farming: continues sacrificing until receiving an artifact gift from the god ("Use my gift wisely"). Only stops after receiving one artifact.
- BUC identification enables safe equipping: AutoAscend won't wear unknown-status items (might be cursed).

**What we do:**
- Nothing. No altar detection. No BUC identification. No sacrifice.

**Score impact:** ~200-500 points. Artifact gifts are powerful weapons. BUC identification prevents the common death of putting on cursed armor and being unable to remove it.

**Difficulty:** Medium-Hard. Requires OFFER action, altar detection, corpse carrying, alignment tracking.

---

## 8. Exploration Efficiency

**What AutoAscend does:**
- BFS-based exploration with priority heatmap: tiles adjacent to unseen stone get high priority, dead-end corridors and door-adjacent tiles get extra priority for searching.
- Search budget: searches 5 times at each search target. Tracks `search_count` per tile to avoid re-searching.
- Trap detection and untrapping: searches near object piles, untraps bear traps/land mines/dart traps/arrow traps/webs to clear paths.
- Door handling: opens doors, kicks locked doors, tracks door-open attempts.
- When stuck (search_diff > 400), allows walking through traps and attacking all monsters as escape heuristics.

**What we do:**
- BFS frontier exploration (correct approach). Tracks seen/walkable masks. Opens doors by walking into them, kicks locked doors.
- 200-search cap per level, then random walk. No priority heuristic for where to search.
- No trap untrapping. No search count per tile.

**Score impact:** ~200-400 points. Better exploration means finding stairs faster, fewer wasted turns, and discovering items/features that would be missed.

**Difficulty:** Easy-Medium. Per-tile search counting and priority heuristics are easy to add.

---

## 9. Item Management

**What AutoAscend does:**
- `ItemPriority._split()` evaluates every item by category, calculates weight budget from carrying capacity, and decides what to keep vs drop.
- Priority order: best weapon > best armor set > healing potions > ranged ammo > food (by nutrition/weight) > sacrificial corpses > potions/rings/wands/scrolls > everything else.
- Weight management: tracks `carrying_capacity` based on strength/constitution, won't pick up items that would overburden.
- Price identification: identifies items by shop prices.
- Wand identification: engraves with each unidentified wand to determine its type from the message.
- Container management: uses bags to extend effective inventory (though some of this is disabled in current code).

**What we do:**
- PICKUP action when standing on items (if not encumbered). No item evaluation. No weight management. No identification.

**Score impact:** ~200-400 points. Healing potions alone are worth hundreds of points of survival. Wand of death/sleep/fire can clear dangerous situations. Item identification enables all other item-using strategies.

**Difficulty:** Hard. Full item management requires item parsing, BUC tracking, identification methods, and weight budgeting. All require the full action space.

---

## 10. Pet Management

**What AutoAscend does:**
- Tracks pet location via `_last_pet_seen`. Waits for pet near stairs before descending (pet follows through stairs if adjacent).
- Won't sacrifice initial pet corpse types (checks for pony/kitten/little dog by name).
- Pet is implicitly used as a meat shield and item BUC-tester (pets won't step on cursed items).

**What we do:**
- Tracks pet position for BFS (swap positions by walking into pet). No waiting for pet at stairs. Pet usually dies or gets lost by DL 2.

**Score impact:** ~100-200 points. Keeping the pet alive provides a combat ally and implicit BUC testing. The pet also generates kills which grant XP.

**Difficulty:** Easy. Wait at stairs for pet proximity before descending. The biggest win is just not losing the pet.

---

## Summary Table

| Gap | Score Impact | Difficulty | Action Space Blocked? |
|-----|-------------|------------|----------------------|
| Equipment management | 2,000-3,000 | Hard | Yes |
| Level descent strategy | 1,500-2,500 | Medium | No |
| Sokoban | 500-1,000 | Hard | Yes (partially) |
| Combat strategy | 500-1,000 | Medium | Partially |
| Excalibur | 300-800 | Medium | Yes |
| Food strategy | 300-500 | Easy-Medium | No |
| Altar use/sacrifice | 200-500 | Medium-Hard | Yes |
| Exploration efficiency | 200-400 | Easy-Medium | No |
| Item management | 200-400 | Hard | Yes |
| Pet management | 100-200 | Easy | No |

## Key Finding

The fundamental constraint is the action space. NetHackScore-v0's 23 actions cannot wield weapons, wear armor, use items, engrave Elbereth, sacrifice corpses, or dip objects. Six of the ten gaps are fully or partially blocked by the action space. Switching to the full NLE action space (NetHackChallenge-v0, 113 actions) is a prerequisite for closing the majority of the gap.

## Actionable Wins Without Action Space Change

Three gaps can be addressed immediately:

1. **Level descent gating** (1,500-2,500 pts): Don't descend until experience level >= 5-8. Stay on early levels and fight. This is the single highest-value change available in the current action space.

2. **Food corpse blacklist expansion** (100-200 pts of the 300-500): Add bats (stun), dogs/cats (aggravation), mimics (paralysis), yellow mold (hallucination), purple worm (engulf), and disenchanter to the unsafe list.

3. **Pet stair waiting** (100-200 pts): When at stairs, wait up to 16 turns for pet to arrive adjacent before descending.
