import json, urllib.request
CODE = '''class Agent:
    def can_engrave(self):
        if self.character.prop.polymorph:
            return False
        return (self.blstats.y, self.blstats.x) != self._forbidden_engrave_position

    def is_safe_to_pray(self, limit=500):
        return (
            (self.last_prayer_turn is None and self.blstats.time > 300) or
            (self.last_prayer_turn is not None and self.blstats.time - self.last_prayer_turn > limit)
        )

    def check_terrain(self, force):
        if force or self._last_terrain_check is None or self.blstats.time - self._last_terrain_check > 50:
            self._last_terrain_check = self.blstats.time
            with self.atom_operation():
                self.type_text('#te'); self.update_level()

    def emergency_strategy(self):
        # runs when in trouble (low HP):
        if self.is_safe_to_pray(500) and self.blstats.hitpoints < self.blstats.max_hitpoints // 5:
            yield True
            self.pray()   # praying takes ~4-10 turns of no action

    def get_visible_monsters(self):
        """returns (distance, y, x, permonst, glyph) for each HOSTILE monster in view"""
        ...'''
PROMPT = ("A NetHack bot (AutoAscend) dies ~3% of games with 'Killed by <monster>, while "
"praying' -- a hostile monster is adjacent and kills it during the multi-turn prayer. "
"Below are several methods of the bot. WHICH method(s) are responsible for the bug, what "
"is the bug, and what is the fix?\n\n" + CODE + "\n\nBe specific and brief.")
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 350, "temperature": 0.2}).encode(),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"])
