import json, urllib.request
DATA = """AutoAscend (symbolic NetHack bot, competition winner) over 300 games with a
macro-strategy layer added. Outcome data:
- Depth: median DL5, 90th percentile DL8, max DL18. ~50% reach DL5+, only 5% reach DL10+.
- Death causes: combat 70%, starvation 18%, prayer 3%, paralysis 3%, crash 2%.
- Combat killers (top): dwarf, soldier ant, white unicorn, giant bat, kitten(own pet),
  werejackal, blast of frost, rothe, mumak, giant mimic, leocrotta, master mind flayer,
  "a wand" (13 -- these are @ human soldiers zapping wand of striking at the bot).
- Median XL at combat death is 6, at depth ~DL5. Most combat deaths are in the Gnomish Mines.
- Sokoban is reached by only 2% of games and truly solved by 0% (the bot abandons it)."""
PROMPT = (DATA + "\n\nList the top 3 SPECIFIC, FIXABLE weaknesses you'd hypothesize from "
"this data (ranked by expected depth impact), and for EACH give a concrete way to verify "
"it in the code/logs. Be specific to NetHack mechanics. Brief.")
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 500, "temperature": 0.3}).encode(),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"])
