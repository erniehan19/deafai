"""Turn one shot's facts into a sentence.

Always builds a template sentence. With use_llm, Claude writes one too, and it's only used
if it states the same facts.
"""

import json
import re

import anthropic

from track import ends_in_house, is_opponent

MAX_WORDS = 25

# --- Template sentences ------------------------------------------------------------------------
IN_RING = {"button": "on the button", "four-foot": "in the four-foot", "eight-foot": "in the eight-foot",
           "twelve-foot": "in the twelve-foot", "biting": "biting the twelve-foot"}
INTO_RING = {"button": "to the button", "four-foot": "into the four-foot", "eight-foot": "into the eight-foot",
             "twelve-foot": "into the twelve-foot", "biting": "to the edge of the house, biting the twelve-foot"}


def spot_in_words(spot, going_into=False, with_detail=True):
    """A spot in words, e.g. "in the four-foot, top left". None if unknown."""
    if spot is None:
        return None
    if spot["zone"] == "out of play":
        return "out of play"
    if spot["zone"] == "guard":
        return "in front of the house"
    if spot["zone"] == "outside":
        return "outside the house"

    # In the house: ring first
    ring = spot.get("ring")
    if going_into:
        words = INTO_RING.get(ring, "into the house")
    else:
        words = IN_RING.get(ring, "in the house")

    # Then where in the ring
    depth = spot.get("depth")
    side = spot.get("side")
    if not with_detail or ring in ("button", "biting") or not (depth or side):
        return words
    if depth == "tee line":
        words += ", on the tee line"
        if side:
            words += f", {side}"
        return words
    if depth and side:
        return f"{words}, {depth} {side}"
    if depth:
        return f"{words}, at the {depth}"
    if side == "center":
        return f"{words}, in the center"
    return f"{words}, on the {side}"


def write_sentence(facts):
    """Most detailed sentence that fits in MAX_WORDS."""
    for with_detail, with_push in ((True, True), (False, True), (False, False)):
        sentence = build_sentence(facts, with_detail, with_push)
        if len(sentence.split()) <= MAX_WORDS:
            return sentence
    return sentence  # shortest one


def build_sentence(facts, with_detail, with_push):
    """Build the sentence."""
    thrower = facts["thrower_team"]
    outcome = facts["outcome"]
    thrown_stone_end = facts.get("thrown_stone_end")
    removed = facts.get("removed", [])
    moved = facts.get("moved", [])

    # Shots that touched nothing
    if outcome == "draw":
        return f"{thrower.capitalize()} draws {spot_in_words(thrown_stone_end, True, with_detail)}."
    if outcome == "guard":
        return guard_sentence(thrower, thrown_stone_end)
    if outcome == "outside":
        return f"{thrower.capitalize()}'s stone comes to rest outside the house."
    if outcome == "through":
        return f"{thrower.capitalize()}'s stone slides through and out of play."
    if outcome == "raise":
        return raise_sentence(thrower, moved, with_detail)

    # Shots that moved other stones: what happened to them, then where the thrown stone ended
    if outcome == "takeout":
        sentence, needs_comma = takeout_part(thrower, removed, with_detail)
    else:
        sentence, needs_comma = hit_part(thrower, removed, moved, with_detail, with_push)

    if thrown_stone_end is None:
        return sentence + "."
    if needs_comma:
        sentence += ","  # "..., top left, and stays"
    if thrown_stone_end["zone"] == "out of play":
        return sentence + " and rolls out."
    if thrown_stone_end["zone"] == "house":
        verb = "stays"
    else:
        verb = "rolls"
    return f"{sentence} and {verb} {spot_in_words(thrown_stone_end, False, with_detail)}."


def guard_sentence(thrower, thrown_stone_end):
    """e.g. "Red puts up a corner guard on the left." """
    side = thrown_stone_end.get("side")
    if side == "center":
        return f"{thrower.capitalize()} puts up a center guard."
    if side:
        return f"{thrower.capitalize()} puts up a corner guard on the {side}."
    return f"{thrower.capitalize()} puts up a guard."


def raise_sentence(thrower, moved, with_detail):
    """e.g. "Red raises its own stone into the four-foot." """
    for stone in moved:
        if stone.get("team") == thrower and ends_in_house(stone):
            return f"{thrower.capitalize()} raises its own stone {spot_in_words(stone['after'], True, with_detail)}."


def takeout_part(thrower, removed, with_detail):
    """e.g. "Red takes out the yellow stone in the four-foot". Returns (words, needs comma)."""
    opponents = [stone for stone in removed if is_opponent(stone, thrower)]
    opponent = opponents[0]["team"]
    needs_comma = False

    if len(opponents) == 1:
        sentence = f"{thrower.capitalize()} takes out the {opponent} stone"
        knocked_from = spot_in_words(opponents[0].get("before"), False, with_detail)
        if knocked_from:
            sentence += f" {knocked_from}"
            needs_comma = "," in knocked_from
    elif len(opponents) == 2:
        sentence = f"{thrower.capitalize()} makes a double takeout on {opponent}"
    elif len(opponents) == 3:
        sentence = f"{thrower.capitalize()} makes a triple takeout on {opponent}"
    else:
        sentence = f"{thrower.capitalize()} takes out {opponent} stones"

    removed_own_stone = any(stone.get("team") == thrower for stone in removed)
    if removed_own_stone:
        sentence += f", also removing a {thrower} stone"
        needs_comma = False
    return sentence, needs_comma


def hit_part(thrower, removed, moved, with_detail, with_push):
    """e.g. "Red hits the yellow stone, pushing it to the button". Returns (words, needs comma)."""
    touched = moved + removed

    # Name an opponent stone if one was touched, otherwise the first one
    target = touched[0]
    for stone in touched:
        if is_opponent(stone, thrower):
            target = stone
            break

    target_team = target.get("team")
    if target_team is None:
        sentence = f"{thrower.capitalize()} hits a stone"
    elif target_team != thrower:
        sentence = f"{thrower.capitalize()} hits the {target_team} stone"
    else:
        sentence = f"{thrower.capitalize()} hits a {target_team} stone"

    if not with_push:
        return sentence, False
    pushed_to = spot_in_words(target.get("after"), True, with_detail)
    if not pushed_to:
        return sentence, False
    return f"{sentence}, pushing it {pushed_to}", True


# --- Fact check --------------------------------------------------------------------------------
# Allowed words: team names, these phrases, and filler. Anything else fails.
CLAIMS = {"takeout": "takes out|removes|removing|knocks out", "double": "double takeout", "triple": "triple takeout",
          "hit": "hits|taps|bumps", "push": "pushing|pushes", "draw": "draws", "raise": "raises", "guard": "guard",
          "center guard": "center guard", "corner guard": "corner guard", "out": "rolls out|out of play",
          "rest": "stays|comes to rest|stops|sticks|settles|lands|rolls", "through": "slides through"}
PLACES = {"button": "button", "four-foot": "four-foot", "eight-foot": "eight-foot", "twelve-foot": "twelve-foot",
          "biting": "biting|edge of the house", "guard zone": "in front of the house", "outside": "outside the house",
          "house": "house", "top": "top", "back": "back", "tee line": "tee line", "left": "left", "right": "right",
          "center": "center|middle"}
FILLER = set("a an the and its it own stone stones rock rocks of to into in on at from up puts places makes with also".split())
RING_NAMES = {"button", "four-foot", "eight-foot", "twelve-foot", "biting"}

# Every phrase as (words, "claim" or "place", meaning)
ALL_PHRASES = []
for kind, table in (("claim", CLAIMS), ("place", PLACES)):
    for meaning, phrases in table.items():
        for phrase in phrases.split("|"):
            ALL_PHRASES.append((phrase.split(), kind, meaning))
ALL_PHRASES.sort(key=lambda entry: -len(entry[0]))  # longest first


def phrase_at(words, index):
    """The longest known phrase starting at words[index], or None."""
    for phrase in ALL_PHRASES:
        phrase_words = phrase[0]
        if words[index : index + len(phrase_words)] == phrase_words:
            return phrase
    return None


def sentence_to_steps(sentence, teams):
    """Sentence -> list of [team or claim, {places}]. None if it has a word we don't allow.

    "Yellow takes out the red stone on the button" -> [["yellow", {}], ["takeout", {}], ["red", {"button"}]]
    """
    text = sentence.lower().replace("’", "'")
    for number_word, digit in (("four", "4"), ("eight", "8"), ("twelve", "12")):
        text = re.sub(rf"\b(?:{number_word}|{digit})[\s-]?(?:foot|feet|ft)\b", f"{number_word}-foot", text)  # "4 ft" -> "four-foot"
    words = re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text)

    steps = []
    index = 0
    while index < len(words):
        word = words[index]
        team = word.removesuffix("'s")
        if team in teams:
            steps.append([team, set()])
            index += 1
            continue

        phrase = phrase_at(words, index)
        if phrase is None:
            if word not in FILLER:
                return None  # word we don't allow
            index += 1
            continue

        phrase_words, kind, meaning = phrase
        if kind == "place":
            if not steps:
                return None
            steps[-1][1].add(meaning)  # place belongs to the step before it
        else:
            steps.append([meaning, set()])
        index += len(phrase_words)
    return steps


def says_the_same(sentence, template_sentence, teams):
    """Same facts as the template, same order? Fewer places is OK, extra ones aren't."""
    said_steps = sentence_to_steps(sentence, teams)
    true_steps = sentence_to_steps(template_sentence, teams)
    if said_steps is None or len(said_steps) != len(true_steps):
        return False
    if len(sentence.split()) > MAX_WORDS:
        return False
    for (said_meaning, said_places), (true_meaning, true_places) in zip(said_steps, true_steps):
        if true_places & RING_NAMES:
            true_places = true_places | {"house"}  # in a ring means in the house
        if said_meaning != true_meaning:
            return False
        if not said_places <= true_places:  # every place said must be true
            return False
    return True


# --- LLM ---------------------------------------------------------------------------------------
PROMPT = """\
You write live curling commentary for blind and low-vision fans. You receive a JSON object \
of verified facts about one shot. Write one sentence describing the shot.
- Use only the facts in the JSON. Never add player names, scores, numbers, ends, strategy, \
predictions, or opinions about how good the shot was. If a field is missing, say nothing about it.
- Open with the thrower's team, e.g. "Red ..." or "Yellow's stone ...". Plain words, present \
tense, at most 20 words. Reply with the sentence only.
- Location words you may use: on the button, in the four-foot, in the eight-foot, in the \
twelve-foot, biting the twelve-foot, in the house, outside the house, in front of the house, \
top, back, on the tee line, left, right, center, out of play.
- Action words you may use: draws, takes out, hits, pushing it, raises, stays, rolls out, comes \
to rest, puts up a center guard, puts up a corner guard, slides through.

Fields: outcome is draw (stopped in the house, touched nothing), guard (stopped in front of the \
house, touched nothing), takeout (knocked an opponent stone out of play), hit (moved stones but \
removed no opponent stone), raise (pushed its own stone into the house), through (slid out of \
play, touched nothing) or outside (stopped in play, outside the house). thrown_stone_end is where \
the thrown stone finished. removed lists stones knocked out of play, with where they were before. \
moved lists stones that were pushed, with where they were before and after. Left and right are \
from the thrower's point of view; top is the thrower's side of the house, back is the far side.

Examples:
{"outcome":"draw","thrower_team":"red","thrown_stone_end":{"zone":"house","ring":"four-foot","depth":"top","side":"left"}}
Red draws into the four-foot, top left.
{"outcome":"takeout","thrower_team":"yellow","thrown_stone_end":{"zone":"house","ring":"button"},"removed":[{"team":"red","before":{"zone":"house","ring":"eight-foot","depth":"back","side":"right"}}]}
Yellow takes out the red stone in the eight-foot, back right, and stays on the button."""


class Commentator:
    """Picks the sentence for each shot."""

    def __init__(self, teams, use_llm=False, model="claude-haiku-5-5", timeout=1.5):
        self.teams = set(teams)
        self.model = model
        self.client = None
        if use_llm:
            self.client = anthropic.Anthropic(timeout=timeout, max_retries=0)  # fail fast

    def comment(self, facts):
        """Returns (sentence, source)."""
        template_sentence = write_sentence(facts)
        if self.client is None:
            return template_sentence, "template"

        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=200, system=PROMPT,
                thinking={"type": "disabled"}, output_config={"effort": "low"},
                messages=[{"role": "user", "content": json.dumps(facts)}])
            llm_sentence = "".join(block.text for block in response.content if block.type == "text").strip()
        except Exception as error:
            return template_sentence, f"template, LLM failed: {type(error).__name__}: {str(error)[:80]}"

        # Use Claude's sentence only if it passes the fact check
        if response.stop_reason == "end_turn" and says_the_same(llm_sentence, template_sentence, self.teams):
            return llm_sentence, "llm"
        return template_sentence, f"template, LLM sentence failed the fact check: {llm_sentence!r}"
