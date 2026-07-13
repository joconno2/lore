# Gehennom descent playbook — how skilled players + the one ascending bot do it

Mined from our KB (wiki, Maniac's/Mikko/codehappy guides, NetHack 3.6 source, and the
five bot repos) to engineer a PROPER descent (no genocide/teleport shortcuts). Jim's
call: go for the real descent. The mechanics below confirm shortcuts can't work anyway.

## 0. Why shortcuts are mechanically impossible (settles the fork)

The vibrating square is NOT a pre-baked special level. It is injected at generation
time ONLY on the level satisfying `Invocation_lev()` = `dlevel == num_dunlevs-1` of the
Gehennom branch (`mkmaze.c:1037-1082`, `dungeon.c:1578`). A wizard `^V` depth-teleport
that builds a generic maze is, by definition, not that level -> no `inv_pos`, no square.
That is exactly why our DL43-49 `^V` scans found nothing. `inv_pos` (`decl.h:128`) is only
printed in wizard `#overview`; a fair-play agent must FEEL the square by stepping on it.
=> The real invocation is reachable only by a real Gehennom descent to its penultimate
level. Proper descent is the only path.

## 1. The ascension kit — MANDATORY before the Valley (hard gates)

Prayer AND Elbereth are BOTH dead in Gehennom (prayer -> Moloch; "the power of the Valar
extends only so far"). The usual safety nets are gone, so these must be gear/intrinsic:

- **Magic resistance** (gray DSM / cloak of MR / Magicbane): blocks finger-of-death,
  destroy-armor, most instakill spells, polymorph + teleport traps. Near-essential (liches
  cast before you act).
- **Reflection** (amulet / silver DSM / shield of reflection): bounces death rays (Orcus &
  named demons carry wands of death). Only 4 sources exist.
- **Free action** (ring): no paralysis/sleep lock (frozen-then-nibbled death). Also lets you
  melee-wield potions of paralysis vs covetous foes.
- **Poison resistance**: Orcus/Baalzebub poison bites. **Fire res** (fire traps cut MAX HP;
  also BAG consumables — fire res protects body not inventory). **Cold res** (Asmodeus).
  **Drain resistance** recommended (vampires/Vlad/wraiths drain XP levels; Excalibur/
  Stormbringer grant it).
- **MC3** (magic cancellation) from a cloak: special-effect attacks succeed ~2% vs 100%.
- **Unicorn horn** (blessed): the in-Gehennom prayer substitute — cures blind/confuse/stun/
  sick/hallucination/stat-drain. Plus potions of full healing.
- **Bag of holding** with potions/scrolls/holy water/wand of cancellation bagged (fire traps
  blank/boil unbagged consumables).
- **Escape stack**: wand of teleportation (works on no-teleport levels), several wands of
  digging, scroll of teleport, **scrolls of scare monster** (the Elbereth substitute /
  panic button), cursed potion of gain level (instant jump UP a level).
- **AC -20 or better** (-25..-40 for the deep dive). **Weapon**: silver (Grayswandir best;
  demons/vampires/shades hate silver) + Frost Brand (demons resist fire, not cold).
- **Speed** (boots/ring). **Slow digestion ring** + rations/lizard/lichen corpses (never rot;
  no prayer-for-food here).

## 2. Descent tactics (fast, stair-to-stair — do NOT clear levels)

- Half of Gehennom is pure maze; goal is stair-to-stair, not a full clear.
- **Dig note (critical):** in Gehennom maze levels a wand of digging digs only ONE square per
  zap (not a full row). Dig short paths at choke points.
- **Two modes** (Maniac): (1) move fast, dig where needed, make a direct stair-to-stair path,
  descend; (2) on the Vlad's-Tower-upstair candidate levels (Geh 9-13) and the vibrating-square
  level, magic-map or search exhaustively.
- **Magic mapping** each level (scrolls via magic marker, or the spell) to find stairs fast.
- **Teleport** between stairs if teleport-controlled. A no-teleport level = you're on a
  named-demon special level (the tell).
- **Telepathy** (blindfold + ESP) / detect-monster before rounding maze corners. No dwarves
  spawn in Gehennom, so any `h` on telepathy = mind flayer.

## 3. Top killers + exact counters (the instadeaths that stop us at ~DL28-40)

- **Mind flayers `h`**: tentacle drains INT -> brainless = instadeath (life-saving does NOT
  save). NEVER melee — kill with daggers/arrows/wand of death. Worn helm blocks 7/8 per
  tentacle; ring of sustain ability / dunce cap fixes INT; unicorn horn restores it.
- **Drowning — giant eels/krakens `;`**: grab near water -> drown = instadeath (bypasses HP).
  Kill at RANGE before contact. If grabbed you get ONE turn: teleport-wand the eel, cold
  (strands it), sleep, or levitate. Prevent with oilskin cloak / greased armor. (Eels sleep
  4/5 of the time until you hold the Amulet.)
- **Demon-lord summoning** (Orcus gates nasties incl. Demogorgon): instant-kill the demon
  before it summons (wielded cockatrice corpse w/ gloves, wand of death w/ reflection). Panic
  button vs a gated horde = **scroll of scare monster** (Elbereth is dead here); ring of
  conflict scatters them.
- **Vampires/Vlad `V`**: bite drains an XP level. Drain resistance blocks it; silver weapons;
  altars scare them.
- **Green slime `P`**: touch -> sliming -> instadeath in ~9 turns. Never melee; fire kills them
  and cures in-progress sliming (carry a wand of fire — no prayer-cure here).
- **Covetous liches/demons**: warp adjacent ignoring no-teleport, hit-and-flee to upstairs to
  heal. Block/own the upstairs (teleport there first, drop scare-monster on it, boulder it),
  or paralyze, or snipe from >5 tiles (they won't flee if far).
- **Fire traps**: cut MAX HP, blank/boil unbagged consumables. Bag everything; fire res + MC3.
- **Starvation**: no prayer-for-food. Carry non-rotting food (lizard/lichen), slow digestion.

## 4. Survival reflexes (the behavior policy)

- **NO PRAYER** in Gehennom (goes to Moloch). Exception: Vlad's Tower is not Gehennom -> prayer
  works there.
- **NO ELBERETH** in Gehennom. Substitute = **scroll of scare monster** (drop/tip it). Does not
  stop covetous warping; ignored by `@`, `A`, minotaurs, Riders, mind flayers (MR 90), blinded.
- **Flee by default** — avoid fights (kills level you up -> harder monsters). Route around; use
  speed/dig/teleport-wand to bypass blockers.
- **Escape priority**: (1) teleport-wand (self or on grabber), (2) scare-monster scroll,
  (3) teleport scroll/spell, (4) dig down, (5) cursed gain-level (jump up), (6) conflict ring,
  (7) amulet of life saving (last resort; no save vs brainlessness).
- **Instant-kill priority targets** every time: Wizard of Yendor, master/arch-liches, Orcus,
  Vlad, high priest, Demogorgon.

## 5. The route + depths (budget ~20 levels below the Valley)

Dungeons -> Castle (DL25-29) -> **Valley of the Dead** (Geh L1, DL26-30; last usable altar —
do final BUC/gear here) -> maze/special levels: Asmodeus (Geh2-7), Juiblex swamp (4-7),
Baalzebub (6-9), **Vlad's Tower** (upstair Geh9-13; NOT Gehennom so prayer works; top = Vlad +
6 vampires + the **Candelabrum**; attach 7 candles), Orcus-town (10-15; altar; Book-of-the-Dead
holder Rodney nearby via Wizard's Tower), **Wizard's Tower** x3 (via magic portal on a fake-tower
level; get the **Book of the Dead** from Rodney; krakens in moats — freeze first), fake Wizard's
Towers x2 -> **vibrating-square level** (Gehennom PENULTIMATE, DL44-52; NO downstair until the
ritual; square at a RANDOM position, found by stepping on it -> "You feel a strange vibration
under your feet") -> **Moloch's Sanctum** (DL45-53; non-tele/non-map/undiggable; secret-door
temple; high priest holds the real Amulet).

## 6. The invocation ritual (already proven in our tooling)

Stand on the vibrating square (not on stairs), all three artifacts non-cursed:
1. Apply Candelabrum with 7 candles attached AND lit ("candles burn brightly!").
2. Ring the Bell of Opening within the last 5 moves ("unsettling shrill sound").
3. Read the Book of the Dead ("turn the pages...") -> `mkinvokearea`: floor shakes, a downstair
   to the Sanctum appears under you. Mis-invoke (uncharged/unlit/cursed/bell>4-turns) -> raises a
   master lich + undead. Uncurse via bagged holy water first (no prayer here).

## 7. Reference implementation: BotHack (the ONLY bot in our KB that has ascended)

`data/bots/BotHack/src/bothack/` — Clojure. Port these ideas:
- **Priority handler stack rebuilt every turn** (`mainbot.clj:2258`): survival handlers
  (illness -9, retreat -7, fight -6, drowning -13, starvation -11) PREEMPT the `progress`
  descent goal (priority 19). => descent is the LOWEST-priority goal; danger always interrupts.
  This is the correct per-step architecture (what our per-step controller was groping toward).
- **Dive = trapdoor/hole else dig-down** (`pathing.clj:695 go-down`) with pool-avoidance (dig
  where walls are adjacent so you don't fall next to water). Stairs when known.
- **No-dig-level set** (`dungeon.clj:695 diggable-walls?`): sanctum/medusa/bigroom/Vlad/quest/
  soko/astral excluded.
- **Vibrating square by MESSAGE, not computed** (`actions.clj:133`, `behaviors.clj:72 seek
  :vibrating`): walk the bottom maze until "strange vibration" tags the tile.
- **Idempotent 7-step invocation chain** (`behaviors.clj:15-78`) keyed on "downstairs appeared".
- **Chase covetous monsters** (`mainbot.clj:1357 fight-covetous`) so they can't teleport-steal.
- **Wish list = ascension-kit gate** (`mainbot.clj:1580`): scrolls of charging, ring of
  levitation (below Medusa), DSM, shield of reflection, remove curse, 7 candles, speed boots,
  helm of telepathy, wand of death — assembled BEFORE the endgame.

Other bots: saiph reaches the Castle (has a Passtune Mastermind solver worth porting), stops.
TAEB = framework + Elbereth primitives, no endgame AI. autoascend = confirmed zero Gehennom/
invocation code (`global_logic.py Milestone` ends at `GO_DOWN = auto() # TODO`). hihack wraps
autoascend. So BotHack is the sole algorithmic source for the descent + invocation.

## 8. Implications for OUR build (proper descent)

1. Our current kit is INCOMPLETE for a real descent — it lacks reflection, free action, the
   full resistance set, scare-monster scrolls, unicorn horn. Genocide is a shortcut to drop;
   replace with the real kit + scare-monster (the legit Gehennom panic button).
2. Adopt BotHack's survival-preempts-descent priority stack (per-step), with the specific
   counters in section 3 (ranged-kill flayers/eels, scare-monster hordes, block covetous
   upstairs).
3. Descent = stair-to-stair + magic-map + dig-down (single-square) with pool-avoidance, not
   explore-the-level.
4. Vibrating square found by stepping (message), then the proven ritual. This is the fair-play
   rung-4 path; no teleport shortcut can reach it.
