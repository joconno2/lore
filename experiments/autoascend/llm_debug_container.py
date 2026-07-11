import json, urllib.request
CODE = '''# --- item_manager.py: when (re)building an inventory item, link its cached content ---
if item.is_possible_container() or item.is_container():
    if item.comment:                         # comment is a name the bot gave it, e.g. "0"
        identifier = item.comment            #   (parsed from the game text "a sack named #0")
    else:
        if position is not None:
            identifier = position
        else:
            identifier = self._get_new_container_identifier()   # returns an int: 0, 1, 2, ...
    item.container_id = identifier
    if identifier in self.container_contents:                    # container_contents: dict
        item.content = self.container_contents[identifier]       # link known content

# --- inventory.py: after physically checking a container, store its content ---
if item.content is None:
    self.item_manager.container_contents[item.container_id] = content   # store under container_id
    item.content = content
# (After the first check, the bot NAMES the container in-game so next turn its text is
#  "a sack named #0", and item.comment becomes "0".)'''
PROMPT = ("AutoAscend (NetHack bot) bug: it re-opens the SAME empty sack in its inventory "
"hundreds of times per game (message 'The sack is empty' ~269x), wasting thousands of "
"turns and contributing to starvation. It never seems to 'remember' the sack is already "
"checked. The inventory item objects are REBUILT fresh every turn from the game text. "
"Here is the relevant code:\n\n" + CODE + "\n\nWhat is the specific bug and the fix? Be brief.")
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 350, "temperature": 0.2}).encode(),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"])
