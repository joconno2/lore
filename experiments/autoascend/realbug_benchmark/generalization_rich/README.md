# Generalization: the diagnosis ceiling is not an AutoAscend artifact

A reviewer will ask whether the ~5-22% diagnosis rate and the exact-token mechanism are
specific to AutoAscend / NetHack. This tests the SAME co-pilot protocol on a completely
different domain: `rich` (Textualize/rich, a pure-Python terminal-rendering library,
4460 commits). Mine real logic-bug fixes (`rich_extract.py`, excludes docs/tests/typing/
typo/mypy), feed the exact buggy function + author symptom, diagnose, judge vs the diff.

## Result

| co-pilot (perfect localization) | strict | lenient |
|---|---|---|
| AutoAscend (NetHack bot) | 2/15 = 13% | 7/15 = 47% |
| **rich (terminal rendering)** | **1/20 = 5%** | **9/20 = 45%** |

The ceiling GENERALIZES: ~5-13% strict, ~45-47% lenient on both a symbolic game agent
AND a rendering library. Given the exact buggy function, the model points at the right
AREA ~45% of the time but names the specific fix only ~5-13%.

**And rich sharpens the mechanism.** Its missed fixes are overwhelmingly EXACT-TOKEN
edits: add `~` to a URL regex char-class, `==`→`is` (identity vs equality),
`int((cut/cell_length)*len(text))`→`*(len(text)-1)` (off-by-one), `assert tab_size is
not None`→default to 8. The model reliably identifies the AREA but cannot generate the
exact TOKEN — the same floor as hallucinated NetHack item names and the Sokoban `'^'`
constant. So the exact-token-generation floor is domain-general, not a NetHack quirk:
the LLM sees WHERE but not WHAT-exactly, and real fixes hinge on the exact token.

## Reading

The positive result's mechanism (grounding gates attempting; bug-simplicity/exact-token
gates succeeding) holds across two unrelated Python codebases. The debugger is a general
"narrows the search to the right area" tool (~45%), not a "names the fix" tool (~5-13%),
regardless of domain.
