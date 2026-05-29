# NetHack Knowledge Base Deep Dive

> Comprehensive reference for building the LORE expert system and LLM oracle.
> Target: NetHack 3.6.6/3.6.7 (NLE version).

## Data Sources Collected

| Source | Location | Size | Content |
|--------|----------|------|---------|
| NetHackWiki XML dump | `data/wiki/nethackwiki_current.xml` | 186MB, 38,224 pages | Every monster, item, mechanic, strategy |
| NetHack 3.6.7 source | `data/source/NetHack/` | 108 .c files | Ground truth for all mechanics |
| Steelypips spoilers (3.4.3) | `data/spoilers/spoiler_files/` | 40 structured text files | Items, monsters, food, spells, wands, etc. |
| AutoAscend source | `data/bots/autoascend/` | ~16K lines Python | NeurIPS 2021 winner, best symbolic bot |
| BotHack source | `data/bots/BotHack/` | ~24K lines Clojure+Java | Only bot to ascend (3.4.3) |
| TAEB source | `data/bots/TAEB/` | ~21K lines Perl | Framework + multiple AI backends |
| Saiph source | `data/bots/saiph/` | ~29K lines C++ | 37 analyzers, highest non-ascending score |
| HiHack source | `data/bots/hihack/` | Python | Best hierarchical neural agent |
| Wiki fetch script | `scripts/fetch_wiki.py` | - | MediaWiki API fetcher for individual pages |

### Still needed
- NetHack Discord dump (channel history, strategy discussions)
- NAO/Hardfought xlogfiles (game statistics)
- GameFAQs guides (Maniac's Ascension Guide, Object ID FAQ)
- StrategyWiki NetHack pages
- Reddit r/nethack top strategy posts

---

## Existing Bot Performance

| Bot | Approach | Score (median) | Max DL | Ascended? | Version |
|-----|----------|---------------|--------|-----------|---------|
| AutoAscend | Symbolic (Python) | 5,300 | ~10 | No | 3.6.6 |
| BotHack | Symbolic (Clojure) | - | Full game | Yes (3.4.3) | 3.4.3 |
| TAEB Behavioral | Symbolic (Perl) | - | 28 | No | 3.4.3 |
| Saiph | Symbolic (C++) | - | 29 | No | 3.4.3 |
| Sample Factory APPO | RL | ~3,245 mean | ~2 | No | 3.6.6 |
| SOL+Motif | RL (30B steps) | new SOTA | - | No | 3.6.6 |
| HiHack | RL+BC | 972 | ~2 | No | 3.6.6 |
| LORE B4+CMA-ES | RL+KB+EC | ~180 best | ~2 | No | 3.6.6 |

No bot has ascended on 3.6.6. The gap between learned and symbolic is ~50x on median score and ~10x on mean dungeon level reached.

---

## The Knowledge Wall

RL agents plateau at dungeon level 1-2. Symbolic agents reach DL 10-29. The difference is pure knowledge. Every section below documents a knowledge domain where RL fails and expert systems succeed.

### 1. Item Identification

NetHack randomizes item appearances each game. A "scroll labeled ZELGO MER" could be identify, genocide, or punishment. There are ~350 items across 10 classes, each with randomized descriptions. Identification is THE core skill separating beginners from ascenders.

**Methods (in order of safety/cost):**

1. **BUC testing (altar):** Drop items on an aligned altar. "Flash" tells BUC status. Zero risk. Requires finding an altar.
2. **Price identification (shop):** Buy/sell prices reveal base price. Combined with item class, narrows possibilities to 2-5 candidates. See price tables below.
3. **Engrave-ID (wands):** Engrave on the floor with each wand. Effects reveal identity:
   - "The engraving now reads ELBERETH" = wand of fire/lightning
   - "A few ice cubes drop from the wand" = cold
   - "The bugs on the ground stop moving!" = death/sleep
   - "The engraving in the dust is now gone" = teleportation/make invisible/cancellation
4. **Use-testing:** Read scrolls, quaff potions, wear rings. Risky (cursed items, bad effects). Last resort.
5. **Exclusion:** Track which appearances map to which items. As you identify items, remaining unknowns narrow.

**BotHack's approach (core.logic):** Constraint propagation. Each observation (price, effect, appearance) adds a constraint. The logic solver eliminates impossible candidates. This is the cleanest model for an expert system.

**AutoAscend's approach:** Price-based lookup only. No altar ID, no use-testing. Reverse-engineers base price from shop prices with charisma adjustment.

**What an LLM could add:** Reasoning about risk/reward of use-testing. "I have 3 unidentified scrolls at price 100. I know identify is price 20 and I've found it. These could be confuse monster, destroy armor, fire, food detection, gold detection, magic mapping, scare monster, or teleportation. Reading one while confused could be useful (some scrolls have different confused effects). Is it worth the risk given my current situation?" This kind of contextual reasoning is where static rules break down.

#### Scroll Price Table

| Price | Scrolls |
|-------|---------|
| 20 | identify |
| 50 | light |
| 60 | blank paper, enchant weapon, enchant armor |
| 80 | remove curse |
| 100 | confuse monster, destroy armor, fire, food detection, gold detection, magic mapping, scare monster, teleportation |
| 200 | amnesia, create monster, earth, taming, charging |
| 300 | genocide, punishment, stinking cloud |

#### Potion Price Table

| Price | Potions |
|-------|---------|
| 0 | uncursed water (holy water if blessed) |
| 50 | booze, fruit juice, see invisible, sickness |
| 100 | confusion, extra healing, hallucination, healing, restore ability, sleeping, water (non-zero cost) |
| 150 | blindness, gain energy, invisibility, monster detection, object detection |
| 200 | enlightenment, full healing, levitation, polymorph, speed |
| 250 | acid, oil |
| 300 | gain ability, gain level, paralysis |

### 2. Prayer and Religion

Prayer is the most powerful single action in the game. It can: cure starvation, restore HP to full, fix illness/stoning/lycanthropy, uncurse worn items, grant intrinsics. But praying at the wrong time angers your god, which can kill you.

**Mechanics (from pray.c):**
- Prayer timeout starts at 300 turns after last prayer
- Major trouble (stoning, starvation, illness): safe when timeout < 201
- Minor trouble (low HP, blindness): safe when timeout < 101
- No trouble: safe only when timeout reaches 0
- Alignment affects timeout: each kill/sacrifice adjusts alignment record
- Praying on wrong altar (cross-aligned) is always bad
- Sacrifice: corpses on altar grant alignment, can convert altar, can get artifact gifts

**What bots do:**
- AutoAscend: pray only when fainting, 500-turn cooldown (too conservative)
- BotHack: tracks alignment and timeout, uses prayer for HP recovery, starvation, illness

**What an LLM could add:** "I'm at 15 HP, a minotaur is adjacent, I last prayed 250 turns ago. My alignment is +12. Should I pray, fight, or use Elbereth?" This requires integrating timeout math, threat assessment, and alternative options.

### 3. Corpse Eating

Eating corpses is the primary way to gain intrinsics (resistances, telepathy, speed). But some corpses are poisonous, some cause illness, some have timed effects.

**Key rules:**
- Corpse age: fresh (< 100 turns old on current level). Older corpses may be tainted.
- Poisonous corpses: eating gives "you feel very sick" unless poison resistant. Can kill.
- Acidic corpses: deal damage unless acid resistant.
- Beneficial intrinsics: gained with probability (monster_level / 15), capped at 100%.
- Critical early intrinsics: poison resistance (from killer bee, snake), fire/cold resistance (dragons, giants)
- Floating eye corpse: grants telepathy (game-changing for combat)
- Wraith corpse: gain a level (always eat)
- Tin: safe preserved food, can be made from any corpse

**Priority list (from rules.py, needs expansion):**
- Always eat: wraith, floating eye, tengu, newt
- Eat for resistance: fire/cold/shock/poison/sleep/disintegration sources
- Never eat: green slime (turns you to slime), various instant-death corpses
- Conditional: corpses that grant teleportitis (bad unless you have teleport control)

### 4. Combat Tactics

**Elbereth:** Writing "Elbereth" on the floor causes most monsters to flee. The most important defensive mechanic.
- Burn with fire/lightning wand: permanent
- Write in dust: temporary, erased by movement
- Engrave with athame: semi-permanent
- Monsters that ignore Elbereth: Riders, quest nemeses, @-class (humans)
- Strategy: write Elbereth, rest to heal, let monsters come to you on adjacent tiles

**Ranged vs melee tradeoffs:**
- Ranged preferred when: multiple threats, dangerous melee monsters (cockatrice, floating eye), low HP
- Melee preferred when: single weak monster, need to conserve ammo/wand charges
- Wands of death: instakill most monsters, 4-8 charges, save for emergencies
- Wands of fire/cold/lightning: area damage, also useful for engraving

**Monster-specific tactics (partial list):**
| Monster | Counter |
|---------|---------|
| Cockatrice | Gloves required. Don't let corpse touch bare hands. |
| Floating eye | Never melee without blindfold/telepathy. Paralyzes on hit. |
| Medusa | Reflection (shield/amulet) or blindfold. |
| Rust monster | Remove metal armor first. |
| Gelatinous cube | Engulfment destroys inventory. Kill from range. |
| Shopkeeper | Don't fight unless endgame-ready. |
| Wizard of Yendor | Respawns. Kill quickly, steal amulet, run. |
| Mind flayer | Intelligence drain. Very dangerous. |
| Lich | Curse items, summon monsters. Priority kill. |

### 5. Inventory Management

51-slot inventory. Weight matters (encumbrance slows you, limits actions). Managing inventory is a constant optimization problem throughout the game.

**Ascension kit (ideal endgame loadout):**
- Weapon: Grayswandir (silver, double damage vs many) or Excalibur
- Armor: dragon scale mail (usually silver or gray), helm of brilliance, speed boots, gauntlets of power, cloak of magic resistance
- Amulet: life saving (backup), reflection (if no shield/armor source)
- Ring: conflict, levitation, free action
- Tools: bag of holding, unicorn horn, skeleton key, stethoscope, magic lamp
- Scrolls: teleportation (emergency escape), identify, genocide
- Potions: full healing, gain level, speed
- Wands: death, teleportation, digging, fire

**Bag of holding rules:**
- Reduces weight of contents
- NEVER put a bag of holding inside another bag of holding (explosion, destroys everything)
- NEVER put a wand of cancellation in a bag of holding (destroys contents)
- Cursed bag randomly eats items

### 6. Dungeon Navigation

**Branch order (typical ascension path):**
1. Main dungeon to Mines (DL 2-5 entrance)
2. Mines to Minetown (altar, shops, luckstone)
3. Back to main dungeon, down to Oracle (DL 5-9)
4. Sokoban (entrance near Oracle, 4 levels, guaranteed loot)
5. Continue main dungeon to Quest portal (DL 11-16)
6. Complete Quest (requires XL 14+, aligned)
7. Main dungeon to Castle (bottom of upper dungeon)
8. Gehennom (below Castle, fire-themed, mazes)
9. Retrieve Amulet of Yendor from Moloch's Sanctum
10. Ascend back through all levels
11. Elemental Planes (4 planes: Earth, Air, Fire, Water)
12. Astral Plane: sacrifice Amulet on correct altar

**Key waypoints:**
- Minetown: altar for BUC testing, shops for price ID
- Sokoban: guaranteed bag of holding OR amulet of reflection
- Castle: wand of wishing (in chest behind drawbridge)
- Vlad's Tower: Candelabrum of Invocation
- Wizard's Tower: Book of the Dead

### 7. Resistance Management

Resistances prevent or reduce damage from specific sources. Critical for survival in mid/late game.

**Priority order:**
1. Poison resistance (ASAP, many monsters poison, instakill without it)
2. Fire resistance (Gehennom is fire-themed, fire traps everywhere)
3. Cold resistance (ice-themed areas, wand of cold)
4. Sleep resistance (sleep attacks paralyze you)
5. Shock resistance (electric eels, wand of lightning)
6. Disintegration resistance (touch of death, black dragon breath)
7. Magic resistance (from cloak/artifact, blocks many spell effects)
8. Reflection (from shield/amulet, reflects beams including death ray)

**Sources:**
- Corpse eating: most resistances
- Dragon scale mail: one resistance matching dragon color
- Artifacts: some grant MR, reflection
- Rings: fire, cold, shock, poison resistance
- Cloak of magic resistance: MR (critical item)

---

## Architecture Patterns from Existing Bots

### BotHack: Constraint-Based Item ID (core.logic)
```
;; Simplified BotHack item ID logic
;; Each observation adds a constraint
;; core.logic eliminates impossible candidates
(defn price-constrainto [item price]
  (membero price (possible-prices item)))

(defn appearance-constrainto [item appearance]
  (== (item-appearance item) appearance))

;; Solve: given observations, what could this item be?
(run* [q]
  (price-constrainto q observed-price)
  (appearance-constrainto q observed-appearance)
  (membero q all-scrolls))
```

### AutoAscend: Strategy Preemption
```python
# Simplified AutoAscend strategy pattern
@Strategy.wrap
def global_strategy(self):
    yield True  # always active
    # Strategies preempt each other by priority
    fight = self.fight_strategy()
    explore = self.explore_strategy()
    heal = self.heal_strategy()

    # heal preempts fight preempts explore
    strategy = explore.preempt(fight).preempt(heal)
    strategy.run()
```

### Saiph: Analyzer Decomposition
```
// 37 analyzers, each handles one concern
// Maps cleanly to KB rule categories
analyzers = [
    Elbereth,   // defensive writing
    Fight,      // combat decisions
    Food,       // eating management
    Health,     // HP monitoring
    Explore,    // navigation
    Shop,       // trading
    Armor,      // equipment
    Weapon,     // weapon selection
    // ... 29 more
]
// Each proposes actions with priorities
// Highest priority wins
```

---

## Where Static Rules Fail (LLM Oracle Opportunities)

These are decision points where the correct action depends on too many contextual variables for a static rule system. This is where LORE's LLM oracle should focus.

### 1. Contextual Risk Assessment
"I'm a level 8 Valkyrie with 45/80 HP, poison resistant, AC -4. There's an unidentified potion (price 200). I'm on DL 7, no shops nearby, no altar. Should I quaff it?"

Price 200 potions: enlightenment, full healing, levitation, polymorph, speed. Enlightenment and full healing are great. Speed is good. Polymorph could kill if HP is low after transform. Levitation could strand you. The risk/reward depends on current HP, nearby threats, available escape routes.

### 2. Inventory Triage Under Pressure
"My bag of holding is full. I just found a wand of wishing (3 charges). I need to drop something. Current contents: 5 potions of healing, scroll of genocide, 2 scrolls of identify, wand of fire (2 charges), unicorn horn, 12 food rations."

This requires understanding the relative value of each item given game phase, current loadout, and upcoming challenges.

### 3. Adaptive Combat Sequencing
"Room with 4 monsters: a mind flayer, 2 gnome lords, and a floating eye. I have a wand of death (1 charge), Excalibur, and a blindfold. What sequence of actions maximizes survival?"

Kill mind flayer with wand of death (most dangerous, one charge is worth it). Melee gnome lords. Don't attack floating eye without blindfold. But what if there's a corridor nearby for chokepoint? What if I'm low on HP and should Elbereth first?

### 4. Multi-Step Planning
"I need fire resistance before entering Gehennom. I don't have it. Options: (a) eat a fire giant corpse (need to find one), (b) wish for red dragon scale mail (uses a wish), (c) find and eat a red dragon corpse (dangerous), (d) wear a ring of fire resistance (uses a ring slot). Which path is most efficient given my current resources?"

### 5. Unknown Item Disambiguation
"I have 3 unidentified rings. I've price-checked them: all are 200 zorkmids. Price 200 rings include: adornment, hunger, protection, stealth, sustain ability, protection from shape changers, conflict. I need to identify which is which. What's the safest testing order?"

---

## Bugs in Current KB (rules.py)

The decision complexity agent found price table errors in our existing `nhc/rules.py`. These need fixing before any expert system work.

**SCROLL_PRICES (wrong):** light listed at 20zm (correct: 50zm), enchant weapon/armor at 50zm (correct: 60zm), remove curse grouping wrong (correct: 80zm solo).

**POTION_PRICES (wrong):** gain energy listed at 300zm (correct: 150zm), holy/unholy water at wrong price (they're blessed/cursed water potions at base 100zm), paralysis grouping wrong (correct: 300zm).

Fix these against the steelypips spoilers (ground truth for 3.4.3) and verify against NetHack 3.6.7 objects.c for any version differences.

---

## Complete Bot Catalog

52 bots found across all eras. Key repos cloned locally in `data/bots/`.

**Symbolic bots (pre-NLE):** TAEB, TAEB-AI-Behavioral, TAEB-AI-Planar, TAEB-AI-Magus, Saiph, BotHack, Interhack, Megaman, nethack-el, ohno, trapper, Bottie, nHackBot, nhbot, rascal, EliteBot, RubyTAEB, moomaster/april, roomba, bridey, anna, Eeek, demonia, SWAGGINZZZ

**NLE-era RL agents:** NLE baselines, MiniHack baselines, AutoAscend, RAPH, nle-sample-factory-baseline, Kakao Brain brain-agent, HiHack, Katakomba, Dungeons and Data baselines, Omega (MuZero), nle-agents (DQN+MCTS), RL_NetHack_2020, gym_nethack

**LLM agents:** NetPlay (GPT-4), nethack-llm/BLINDER, BALROG, GlyphBox, NetHackPlayer (Claude Agent SDK), nle-language-wrapper

**Hybrid:** LuckyMera, iv4xr-nethack, Prolog-NetHack-Agent

**No NEAT/neuroevolution bot for NetHack exists.** LORE would be the first EC-based approach.

---

## Community Knowledge Sources

### Discord
- **Roguelikes Discord** (primary): `https://discord.gg/vg9WWf2` -- #nethack channel bridged to IRC. This is the one to dump.
- NetHack Discord (small, ~19 members): `https://discord.com/invite/uFZydVRj`

### IRC Logs (scrapeable)
- alt.org IRC logs: `https://alt.org/nethack/irc.php?l=400`
- Hardfought IRC logs: `https://www.hardfought.org/nethack/irclogs/`
- YANI archive (ideas from IRC): `https://nethack-yanis.github.io/`

### Game Data (structured, downloadable)
- NAO xlogfile: `https://alt.org/nethack/xlogfile` (tab-delimited, every game ever played)
- Hardfought xlogfiles: `https://www.hardfought.org/nethack/xlogfiles/`
- NetHack Scoreboard: `https://nethackscoreboard.org/` (aggregates all servers)
- NLD dataset: 10B transitions, 229GB, `https://dl.fbaipublicfiles.com/nld/`
- Katakomba (HuggingFace): D4RL-style offline RL datasets from AutoAscend

### Forums / Archives
- RGRN (Usenet): `https://groups.google.com/g/rec.games.roguelike.nethack` (decades of expert discussion)
- Reddit r/nethack: strategy posts, ascension reports
- Codehappy expert play statistics: `https://codehappy.net/nethack/data.htm` (35K games analyzed)

### Academic Papers
Key papers with code: NLE (2020), MiniHack (2021), NeurIPS Challenge Insights (2022), Dungeons & Data (2022), NetHack is Hard to Hack (2023), Katakomba (2023), LuckyMera (2024), NetPlay (2024), BALROG (2024), SOL+Motif (2025), "Revisiting NLE" (ICLR 2026 blog).

---

## Expert Play Statistics (codehappy.net)

From 35,131 games by expert players (15.9% overall ascension rate):

- **Survival threshold:** Reaching the Castle virtually guarantees ascension. Almost no expert deaths after Castle.
- **Sokoban prize choice:** No significant difference (amulet 61.7% vs BoH 60.6% ascension rate).
- **Early reflection > early MR:** 61.7% vs 55.4% ascension rate (99% confidence).
- **Protection racket underperforms:** 49.9% ascension vs higher for standard play.
- **After 3 consecutive wins:** ascension rate jumps to 70.7%. Knowledge compounds.

---

## Next Steps

1. **Parse wiki XML** into structured per-page text files (run fetch_wiki.py or write XML parser)
2. **Extract structured data** from NetHack source (monst.c, objects.c, pray.c, eat.c, shk.c)
3. **Build the expert system** layer by layer:
   - Monster database (from monst.c + wiki)
   - Item database with price tables (from objects.c + spoilers)
   - Corpse safety/benefit table (from eat.c + wiki)
   - Prayer timeout calculator (from pray.c)
   - Resistance tracker
   - Item ID constraint system (inspired by BotHack's core.logic)
4. **Identify LLM query points** from the "where static rules fail" analysis
5. **Design the MAP-Elites genome** for retrieval strategy optimization
6. **Benchmark**: expert system alone vs expert system + LLM oracle vs baseline PPO
