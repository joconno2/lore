"""NLE env that allows ENGRAVE text entry.

NetHackScore-v0 auto-ESCs text-entry prompts (in_getlin), which blocks
ENGRAVE. This subclass overrides _perform_known_steps to pass through
getlin when the last action was ENGRAVE or when we detect an engrave
prompt in the message.
"""
from nle.env.tasks import NetHackScore

ASCII_ESC = 27
ASCII_SPACE = 32
ENGRAVE_VAL = 69  # ord('E'), Command.ENGRAVE value

# Messages that indicate we're in an engrave sequence
_ENGRAVE_MSGS = [
    b"What do you want to write with",
    b"What do you want to write in the dust",
    b"What do you want to engrave",
    b"You write in the dust",
    b"You engrave in the dust",
    b"Do you want to add to the current engraving",
]


class NetHackScoreEngrave(NetHackScore):
    """NetHackScore with ENGRAVE text entry support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._in_engrave = False

    def step(self, action):
        # Track if we're sending ENGRAVE
        if action < len(self.actions):
            act_val = int(self.actions[action])
            if act_val == ENGRAVE_VAL:
                self._in_engrave = True

        result = super().step(action)

        # Check if engrave sequence ended
        obs = result[0]
        misc = obs.get("misc")
        if misc is not None and not misc[1] and not misc[2]:
            # Not in getlin or yn, engrave sequence is over
            if self._in_engrave:
                self._in_engrave = False

        return result

    def _perform_known_steps(self, observation, done, exceptions=True):
        """Override to allow getlin during engrave."""
        while not done:
            if observation[self._internal_index][3]:  # xwaitforspace
                observation, done = self.nethack.step(ASCII_SPACE)
                continue

            internal = observation[self._internal_index]
            in_yn_function = internal[1]
            in_getlin = internal[2]

            if in_getlin:
                # Check if this is an engrave prompt
                msg = bytes(observation[self._message_index])
                is_engrave = self._in_engrave or any(
                    em in msg for em in _ENGRAVE_MSGS
                )
                if is_engrave:
                    # Let the agent handle this
                    self._in_engrave = True
                    break
                # Not engrave: auto-ESC as usual
                observation, done = self.nethack.step(ASCII_ESC)
                continue

            if in_yn_function:
                if exceptions:
                    msg = bytes(observation[self._message_index])
                    # Check for engrave yn prompts
                    is_engrave = self._in_engrave or any(
                        em in msg for em in _ENGRAVE_MSGS
                    )
                    if is_engrave:
                        self._in_engrave = True
                        break
                    if self._allow_all_yn_questions or any(
                        el in msg for el in self.__class__._skip_exceptions()
                    ):
                        break
                observation, done = self.nethack.step(ASCII_ESC)

            break

        return observation, done

    @staticmethod
    def _skip_exceptions():
        from nle.env.base import SKIP_EXCEPTIONS
        return SKIP_EXCEPTIONS
