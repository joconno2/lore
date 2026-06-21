# NetHack 3.6 Ascension Strategy Corpus (planner knowledge)

Retrieved-expertise corpus for the LORE planner. Each rule is **TRIGGER → ACTION**
so it can become a planner goal/condition. Tags: `[V]` deep-research verified with
cited sources (adversarial 2/3-vote), `[C]` established NetHack canon (to be
hardened by a second research pass), `[!]` hard rule (instadeath / irreversible).
Class focus: dwarven lawful Valkyrie (the consensus first-ascension combo).

Generated Jun 21 2026 from a deep web research pass (NetHack wiki, StrategyWiki,
steelypips spoilers, Fandom, 3.6.7 source) + canon. Sources cited inline.

---

## 0. Character (pre-game / fixed facts)

- `[V]` Dwarven lawful Valkyrie is the recommended first-ascension combo: warrior
  melee survivability, **cold resistance + stealth from XL1**, **intrinsic speed
  at XL7**, high dwarf HP growth, infravision, near-peaceful Gnomish Mines.
  (nethackwiki.com/wiki/Valkyrie; fandom Valkyrie; strategywiki Choosing_Role)
- `[V]` Starting kit: **+1 long sword, +0 dagger, +3 small shield, 1 food ration**,
  1/6 chance of an oil lamp. (NO "+1 spear" — that's outdated 3.4.3.) (3.6.7
  src/u_init.c)
- `[V]` Myth corrections the planner must NOT believe: stealth is XL1 (not XL3);
  no +1 spear; post-prayer timeout is not a fixed ~1000.

## 1. Early game (DL1-5)

### Prayer (highest-leverage survival mechanic)
- `[V][!]` Initial prayer timeout = **300**, -1 per game turn. Safe to pray on
  **turn 301** (not 300) with no trouble. (nethackwiki Prayer_timeout; steelypips pray)
- `[V][!]` With a **major trouble** (e.g. HP ≤ 1/7 max, or HP ≤ 5), prayer is
  accepted when timeout < 201 → so an emergency prayer works from **turn 101**.
  Minor trouble: timeout < 101.
- `[V]` TRIGGER: HP ≤ 5 (or ≤ 1/7 max) AND prayer-safe → ACTION: **pray**. A
  successful prayer restores HP to full and adds 1d5 max HP (if maxHP < 5·XL+11).
- `[V][!]` Prayer is UNSAFE (do NOT pray) if any of: negative alignment record,
  negative Luck, god already angry, timeout above the trouble threshold, in
  Gehennom, or standing on another god's altar. Praying with too-high timeout →
  god anger +1, −3 Luck, smiting (often fatal). (nethackwiki Prayer; steelypips)
- `[C]` Prayer also fixes: starvation (Fainting/Weak), lethal illness, petrification
  in progress ("turning to stone"), lava/sinking, low HP. It is a *rare backstop*,
  not a routine tool — over-praying angers the god. **This is why pray-at-WEAK as a
  routine hunger fix backfired in LORE testing.**

### Elbereth (stand-still defense)
- `[V][!]` Only an **instantaneous** engrave protects: finger-in-dust "Elbereth"
  (8 chars, fast method) lands before monsters move. Any slower/carved engrave
  leaves you helpless and killable mid-engrave. (nethackwiki Engraving)
- `[V]` Dust Elbereth succeeds ~72.65% (each letter 1/25 corrupt); a sighted agent
  should verify with `:` before relying on it.
- `[V]` It DEGRADES from your own actions: moving off corrupts 1-5 chars, **melee
  corrupts 3**, throw/fire 2, kick 2. So Elbereth is a stand-still defense — it
  breaks the moment you attack or move. Don't expect to fight from on top of it
  (most monsters; some respect it, some ignore it).

### Threats / instadeaths
- `[V][!]` **Floating eye ('e'): NEVER melee** without reflection, free action,
  blindness, or hallucination. A non-killing melee hit triggers passive paralysis
  2/3 of the time, up to 127 turns at Wis ≤ 12 → paralyzed then nibbled to death.
  RANGED or AVOID. (nethackwiki Floating_eye)
- `[V][!]` **Cockatrice/chickatrice ('c') petrification.** Lethal contact paths,
  ALL of which stone you unless wearing gloves: bare-handed melee (hit-back/touch),
  and touching the corpse — **picking up, eating, OR offering it for sacrifice all
  count as touching**. With **gloves** the corpse is safe to handle, and a gloved
  hero can **wield a cockatrice corpse as a stoning melee weapon**. Keep a **lizard
  (or acidic) corpse/tin to cure stoning** if it begins. (nethackwiki Cockatrice,
  Petrification, Sacrifice)
- `[C]` Gas spore / other 'e'-explosion: don't melee adjacent; they explode on
  death. Engage from range or avoid.
- `[V]` Unicorns ('u') are fast (speed 24, 2× player) and hit hard — at low HP you
  can't outrun them; Elbereth or pray, don't trade losing blows. (gift the right
  gem to pacify a co-aligned unicorn.)

### Descent pacing
- `[C]` Don't dive faster than you can survive: rough guide XL ≳ 2× dungeon depth
  early, healthy HP, some armor. BUT the AutoAscend base already paces this well —
  LORE testing showed forcing *more* conservative descent hurts. Pace is not a
  LORE lever; depth-prep (gear/resources) is.

## 2. Mid game (Sokoban, Mines, Minetown) — `[V]`

### Sokoban
- `[V]` Branch entered from a **2nd up-stair on DL6-10** (one floor below the
  Oracle). No-teleport; all walls undiggable/unphasable; floor undiggable only on
  level 1. Each level seeds one random ring + one random wand + comestibles. Final
  floor: a **treasure zoo** guarding the prize in one of three closets, on a burnt
  Elbereth + (3.6) a cursed scroll of scare monster. (nethackwiki Sokoban)
- `[V]` **Prize = binary pick: bag of holding OR amulet of reflection, 50/50 in
  3.6**; grabbing one vanishes the other. RULE: take **amulet of reflection if you
  have no other reflection** (no silver DSM / shield of reflection), else bag of
  holding. (nethackwiki Sokoban_prize)
- `[V]` Prize choice is NOT ascension-critical: 35,131 expert games show no
  significant ascension-rate difference (BoH 60.6% vs AoR 61.7%). Decide by
  reflection coverage, not a "best prize" heuristic. (codehappy.net/nethack/data)

### Mines → Minetown → Mines End
- `[V]` **Minetown** (guaranteed temple except Orcish Town, usually shops): THE
  site for **price-id** (shops), **BUC-testing** (altar), and **protection**
  (#chat/donate gold to the co-aligned priest for AC). (nethackwiki Standard_strategy)
- `[V]` **Mines End** (~DL10-13): always a **luckstone** — carry it once you can
  keep Luck positive.
- `[V]` Order: clear **Sokoban early** (fixed high-value loot, no random teleport,
  no-teleport helps corner unicorns), then descend Mines to **Minetown** for
  BUC-id + protection, then Mines End for the luckstone.

### Altars (BUC + sacrifice)
- `[V]` **BUC by drop-flash:** drop item(s) on any altar — **amber = blessed,
  black = cursed, no flash = uncursed**. Defeated by blindness; hallucination
  scrambles blessed/cursed but no-flash still = uncursed. (nethackwiki Altar)
- `[V]` **Sacrifice** (co-aligned altar): corpse must be **≤50 turns old** (except
  acid blob). Timeout >0 → fresh sacrifice REDUCES prayer timeout. Timeout =0 →
  can raise Luck, improve alignment, and grant an **artifact**.
- `[V]` **Artifact gift** fires only if timeout=0 AND alignment record positive AND
  XL≥3 AND base Luck≥0 AND god not angry; prob 1/(10+2·x·y). Gifts are always
  weapons, non-cursed, erosion-proof, never one that attacks your form; **first
  gift is co-aligned**. A gift raises prayer timeout by rnz(300+50x).
- `[V][!]` Offering a cockatrice/chickatrice corpse **touches it → stones you
  unless gloved**.
- `[V][!]` **Never cross-aligned-sacrifice while alignment record is negative** —
  conversion can fail and convert YOU to the altar's alignment → unwinnable.

## 3. Resource economy across the run — `[V]`

- **Food (long-horizon supply — the real starvation fix; the base already eats at
  NOT_HUNGRY, so eat-timing is NOT the gap):**
  - `[V][!]` **Never eat kobold corpses** (poisonous, no benefit).
  - `[V][!]` **Never eat a corpse older than ~50 turns** (food poisoning), except
    lizard/lichen. Unrevived corpses disintegrate at ~250 turns.
  - `[V]` **lizard and lichen corpses never rot and never poison** — the reliable
    long-horizon food reserve; carry them. **lizard (any acidic) corpse/tin cures
    stoning** — keep one as the anti-petrification reserve.
  - `[C]` Eat safe fresh corpses up to (not past) Satiated to bank; eating while
    Satiated risks choking death.
  - `[V]` **Tins** (tinning kit) preserve any-age corpses, remove rot, no
    poison/acid penalty (cursed kit → rotten tin).
  - `[V]` **Pray for food** only when **Weak/Fainting** (a major trouble): a
    successful prayer sets nutrition to 900 (if lower). **Never works in Gehennom**
    (prayer = to Moloch). So food-prayer is a rare backstop, not the supply plan.
  - `[C]` Ring of slow digestion stretches one food source — strong anti-starvation.
- **Gold/consumables:** price-id at shops; keep healing potions + escape items.

## 4. Ascension kit + prep-before-you-dive — `[V]`

- `[V]` **PREP GATE: do NOT enter Gehennom without reflection + magic resistance +
  free action + poison resistance** — prayer fails in Gehennom, so the safety net
  is gone. This is the structural goal the base lacks.
- `[V]` **Reflection** (amulet of reflection / silver dragon scale mail / shield of
  reflection) — blocks death rays. **Cannot be prayed for — must be gear.**
- `[V]` **Magic resistance** (gray dragon scale mail / cloak of MR / Magicbane) —
  blocks many instakills. **Cannot be prayed for — must be gear.**
- `[V]` **Free action** (ring) — prevents paralysis/sleep lock.
- `[C]` **Poison resistance**, **telepathy** (eat floating eye / amulet of ESP),
  **speed** (boots/ring/potion) — intrinsics from the right corpses/gear.
- `[C]` **Escape**: teleport (scrolls/wands/intrinsic) + digging (wand/pick-axe).
  **Good AC** (target negative).

## 5. Endgame sequence (fixed ordered goals) — `[V]`

- `[V]` Retrieve the **Book of the Dead** from the Wizard of Yendor (his Tower).
  Get candelabrum candles from Vlad's Tower; assemble the Candelabrum.
- `[V]` Descend to the **Vibrating Square**; perform the **Invocation**
  (candelabrum → bell → book, in that order). (nethackwiki Invocation)
- `[V]` Take the **Amulet of Yendor** from the high priest of Moloch in the Sanctum.
- `[V]` Climb up fighting the **Mysterious Force** and the **Wizard**; cross the
  **Elemental Planes**; on the **Astral Plane** offer the Amulet on the
  **co-aligned** altar. Watch for the Riders. (nethackwiki Ascension_run, Astral_Plane)
- `[V][!]` Valkyrie quest hazard: **lava and drawbridges kill far more than nemesis
  Surtur** (~5.6× on NAO). Treat lava/drawbridge crossings as the real threat.
  (nethackwiki Valkyrie_quest)

---

## Planner takeaways (what becomes goals)
1. `prep_gate(reflection, MR, free_action, AC<target)` before deep dives — the top
   structural goal the base lacks.
2. `food_reserve(keep lizard/lichen + rations; bank to Satiated)` — supply, not timing.
3. `acquire(reflection)` via Sokoban prize / Minetown / sacrifice.
4. `branch_order(Sokoban + Mines→Minetown→Mines End)` with their prizes.
5. `instadeath_avoid(floating eye, cockatrice, gas spore)` — ranged/avoid, never melee.
6. `prayer_discipline` — emergency backstop only, never routine.
7. `endgame_sequence(...)` — the fixed ordered invocation→amulet→planes→astral.
