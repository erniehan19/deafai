"""Follow stones frame to frame, and sum up each shot.

Board: matches detections to known stones and tracks each one's status.
    new -> moving <-> still, then out (left play) or lost
    Out stones are still followed until they leave the picture (gone).
ShotTracker: a shot starts when a stone moves and ends when all are still for 1 s.
"""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from detect import describe_spot, is_out_of_play

MOVING_SPEED = 0.15  # m/s, faster is moving
STILL_SPEED = 0.03  # m/s, slower is still
SETTLE_TIME = 1.0  # seconds all stones must be still to end a shot
SURE_ENOUGH = 0.7  # min team confidence to name a team
TOO_FAR = 1e6  # distance meaning "can't be this stone"


class Stone:
    """One tracked stone."""

    def __init__(self, stone_id, time, detection):
        self.id = stone_id
        self.position = detection.position.copy()
        self.velocity = np.zeros(2)
        self.speed = 0.0
        self.updated_at = time
        self.first_seen = time
        self.last_seen = time
        self.status = "new"
        self.resting_position = None  # last place it sat still
        self.arrived_moving = False
        self.recent_sightings = []  # last 0.4 s of (time, x, y)
        self.color_votes = Counter()  # team color seen each frame

    def team(self):
        """Most-seen team, and how sure (0 to 1)."""
        if not self.color_votes:
            return None, 0.0
        team, votes_for_team = self.color_votes.most_common(1)[0]
        total_votes = sum(self.color_votes.values())
        sureness = votes_for_team / total_votes
        if total_votes < 5:
            sureness *= total_votes / 5  # less sure with few votes
        return team, sureness

    def is_sliding(self):
        """Moving, or out of play but not stopped yet."""
        return self.status == "moving" or (self.status == "out" and self.speed > STILL_SPEED)

    def expected_position(self, time):
        """Where it should be now."""
        if self.is_sliding():
            return self.position + self.velocity * (time - self.updated_at)
        return self.position


class Board:
    """Tracks every stone on the ice."""

    def __init__(self, camera):
        self.camera = camera
        self.stones = {}  # id -> Stone
        self.next_id = 1

    def stones_in_play(self):
        """Stones not out and not lost."""
        in_play = []
        for stone in self.stones.values():
            if stone.status in ("new", "moving", "still"):
                in_play.append(stone)
        return in_play

    def stones_to_follow(self):
        """Stones in play, plus out ones that haven't left the picture."""
        to_follow = []
        for stone in self.stones.values():
            if stone.status in ("new", "moving", "still", "out"):
                to_follow.append(stone)
        return to_follow

    def update(self, time, detections):
        """Add one frame. Returns changes: ("appeared", stone) or ("started", stone)."""
        changes = []
        stones = self.stones_to_follow()
        matches = self.match(time, stones, detections)

        # Known stones
        for stone in stones:
            if stone.id in matches:
                detection = detections[matches[stone.id]]
                change = self.seen(stone, time, detection)
            else:
                change = self.not_seen(stone, time)
            if change:
                changes.append((change, stone))

        # Leftover detections are new stones
        for index, detection in enumerate(detections):
            if index in matches.values():
                continue
            stone = self.add_stone(time, detection)
            change = self.seen(stone, time, detection)
            if change:
                changes.append((change, stone))
        return changes

    def add_stone(self, time, detection):
        """Start tracking a new stone."""
        stone = Stone(self.next_id, time, detection)
        self.stones[stone.id] = stone
        self.next_id += 1
        return stone

    def match(self, time, stones, detections):
        """Pair stones with detections, closest overall. Returns {stone id: detection index}."""
        # Table of distances: one row per stone, one column per detection
        distances = np.full((len(stones), len(detections)), TOO_FAR)
        for row, stone in enumerate(stones):
            for column, detection in enumerate(detections):
                distances[row, column] = self.match_distance(stone, detection, time)

        rows, columns = linear_sum_assignment(distances)  # pairing with the smallest total distance
        matches = {}
        for row, column in zip(rows, columns):
            if distances[row, column] < TOO_FAR:
                matches[stones[row].id] = column
        return matches

    def match_distance(self, stone, detection, time):
        """How far the detection is from the stone. TOO_FAR if it can't be that stone."""
        distance = np.linalg.norm(stone.expected_position(time) - detection.position)

        # How far it could have gone
        if stone.status == "new" or stone.is_sliding():
            max_distance = 0.6 + 0.5 * stone.speed * (time - stone.last_seen)
        else:
            max_distance = 0.35
        if distance > max_distance:
            return TOO_FAR

        team, sureness = stone.team()
        if team and sureness >= 0.8 and detection.team != team:
            distance += 0.25  # wrong color: only if nothing else fits
        return distance

    def seen(self, stone, time, detection):
        """Update a stone that was detected. Returns "appeared", "started" or None."""
        stone.last_seen = time
        stone.color_votes[detection.team] += 1
        self.add_sighting(stone, time, detection.position)

        oldest_sighting_time = stone.recent_sightings[0][0]
        if len(stone.recent_sightings) < 3 or time - oldest_sighting_time < 0.19:
            # Not enough sightings yet: use the raw position
            stone.position = detection.position.copy()
            stone.updated_at = time
            return None

        self.smooth_motion(stone, time)
        if stone.status == "out":
            return None  # out for good, just follow it
        if stone.status == "new":
            return self.update_new_stone(stone, time)
        if stone.status == "moving":
            return self.update_moving_stone(stone)
        return self.update_still_stone(stone)

    def add_sighting(self, stone, time, position):
        """Remember this sighting, keeping only the last 0.4 s."""
        recent = []
        for sighting in stone.recent_sightings:
            sighting_time = sighting[0]
            if sighting_time >= time - 0.4:
                recent.append(sighting)
        recent.append((time, position[0], position[1]))
        stone.recent_sightings = recent

    def smooth_motion(self, stone, time):
        """Position and speed from a straight-line fit through recent sightings."""
        times = np.array([sighting[0] for sighting in stone.recent_sightings]) - time  # 0 = now
        xs = np.array([sighting[1] for sighting in stone.recent_sightings])
        ys = np.array([sighting[2] for sighting in stone.recent_sightings])
        velocity_x, x_now = np.polyfit(times, xs, 1)  # slope = velocity, value at 0 = position now
        velocity_y, y_now = np.polyfit(times, ys, 1)

        stone.position = np.array([x_now, y_now])
        stone.velocity = np.array([velocity_x, velocity_y])
        stone.speed = float(np.hypot(velocity_x, velocity_y))
        stone.updated_at = time

    def update_new_stone(self, stone, time):
        """Decide if a new stone is moving or still. Returns "appeared" once decided."""
        if stone.speed > STILL_SPEED and is_out_of_play(stone.position):
            stone.status = "out"
            return None

        waited_long_enough = time - stone.first_seen > 1.0
        if stone.speed >= MOVING_SPEED or (waited_long_enough and stone.speed > STILL_SPEED):
            stone.status = "moving"
            stone.arrived_moving = True
            return "appeared"
        if stone.speed <= STILL_SPEED or waited_long_enough:
            stone.status = "still"
            stone.resting_position = stone.position.copy()
            return "appeared"
        return None

    def update_moving_stone(self, stone):
        """Did it leave play, or stop?"""
        if stone.speed > STILL_SPEED and is_out_of_play(stone.position):
            stone.status = "out"
        elif stone.speed <= STILL_SPEED:
            stone.status = "still"
            stone.resting_position = stone.position.copy()
        return None

    def update_still_stone(self, stone):
        """Did it get hit? Returns "started" if so."""
        moved_off_spot = np.linalg.norm(stone.position - stone.resting_position) >= 0.08
        if stone.speed >= MOVING_SPEED and moved_off_spot:
            stone.status = "moving"
            return "started"
        if stone.speed <= STILL_SPEED:
            stone.resting_position = stone.position.copy()
        return None

    def not_seen(self, stone, time):
        """Update a stone that wasn't detected. Still stones stay put (hidden by a player)."""
        if stone.status == "new" and time - stone.last_seen > 0.5:
            del self.stones[stone.id]  # false detection
            return None
        if stone.status == "out":
            # Keep it until it has left the picture (1 s grace for blinks at the edge)
            stone.position = stone.expected_position(time)
            stone.updated_at = time
            if time - stone.last_seen > 1.0 and not self.camera.can_see(stone.position):
                stone.status = "gone"
            return None
        if stone.status != "moving":
            return None

        # Keep it sliding
        stone.position = stone.expected_position(time)
        stone.updated_at = time
        if is_out_of_play(stone.position):
            stone.status = "out"
        elif not self.camera.can_see(stone.position):
            # Left the picture: would it slide out of play? (Ice slows it ~0.1 m/s each second.)
            stops_at = stone.position + stone.velocity * stone.speed / (2 * 0.1)
            if is_out_of_play(stops_at):
                stone.status = "out"
            else:
                stone.status = "lost"
        elif time - stone.last_seen > 1.0:
            stone.status = "lost"
        return None


class ShotTracker:
    """Finds shots and works out what happened."""

    def __init__(self):
        self.shot_number = 0  # shot 0 is already under way, never reported
        self.shot_in_progress = True
        self.shot_started_at = 0.0
        self.all_still_since = None
        self.thrown_stone = None
        self.unexplained_stones = 0  # stones that appeared moving, not the thrown one
        self.positions_before = {}  # stone id -> position before the shot

    def update(self, time, board, changes):
        """Once per frame. Returns a dict when a shot ends, else None."""
        if not self.shot_in_progress:
            if not something_started_moving(changes):
                self.remember_positions(board)
                return None
            self.start_shot(time)

        self.watch_new_stones(changes)

        if time - self.shot_started_at > 60:
            return self.end_shot(board, time, None, "the shot never settled")
        if not self.settled(time, board):
            return None

        facts, no_commentary_reason = self.what_happened(board)
        return self.end_shot(board, self.all_still_since, facts, no_commentary_reason)

    def start_shot(self, time):
        """Reset for a new shot."""
        self.shot_number += 1
        self.shot_in_progress = True
        self.shot_started_at = time
        self.all_still_since = None
        self.thrown_stone = None
        self.unexplained_stones = 0

    def watch_new_stones(self, changes):
        """Spot the thrown stone, and any stones that appear mid-shot."""
        for change, stone in changes:
            if change != "appeared":
                continue
            if not stone.arrived_moving:
                if stone.id not in self.positions_before:
                    self.positions_before[stone.id] = stone.resting_position.copy()  # was hidden before
                continue

            # Thrown stone: shows up already moving toward the house (from the edge or from under a sweeper)
            heading_to_house = stone.velocity[1] > 0
            if self.thrown_stone is None and heading_to_house:
                self.thrown_stone = stone
            else:
                self.unexplained_stones += 1

    def settled(self, time, board):
        """Have all stones been still for SETTLE_TIME?"""
        for stone in board.stones_in_play():
            if stone.status in ("new", "moving"):
                self.all_still_since = None
                return False
        if self.all_still_since is None:
            self.all_still_since = time
        return time - self.all_still_since >= SETTLE_TIME

    def end_shot(self, board, stopped_at, facts, no_commentary_reason):
        """Finish the shot."""
        self.shot_in_progress = False
        self.remember_positions(board)
        if self.shot_number == 0:
            return None
        return {"stopped_at": stopped_at, "facts": facts, "no_commentary_reason": no_commentary_reason}

    def remember_positions(self, board):
        """Save where still stones are."""
        self.positions_before = {}
        for stone in board.stones_in_play():
            if stone.status == "still":
                self.positions_before[stone.id] = stone.resting_position.copy()

    def what_happened(self, board):
        """Compare before and after. Returns (facts, None) or (None, reason)."""
        # Stay quiet if unsure
        if self.thrown_stone is None:
            return None, "never saw the thrown stone come in"
        thrower, sureness = self.thrown_stone.team()
        if thrower is None or sureness < SURE_ENOUGH:
            return None, "not sure which team threw"
        if self.unexplained_stones:
            return None, "a stone started moving that we can't explain"

        thrown_stone_end = self.where_thrown_stone_ended()
        removed, moved = self.other_stones_that_moved(board)

        outcome = name_the_shot(thrower, thrown_stone_end, removed, moved)
        if outcome is None:
            return None, "lost track of where the thrown stone stopped"

        facts = {"outcome": outcome, "thrower_team": thrower}
        if thrown_stone_end:
            facts["thrown_stone_end"] = thrown_stone_end
        if removed:
            facts["removed"] = removed
        if moved:
            facts["moved"] = moved
        return facts, None

    def where_thrown_stone_ended(self):
        """Spot, {"zone": "out of play"}, or None if we lost it."""
        if self.thrown_stone.status == "still":
            return describe_spot(self.thrown_stone.resting_position)
        if self.thrown_stone.status in ("out", "gone"):
            return {"zone": "out of play"}
        return None

    def other_stones_that_moved(self, board):
        """Stones knocked out of play (removed), and stones pushed (moved)."""
        removed = []
        moved = []
        for stone_id, position_before in self.positions_before.items():
            stone = board.stones[stone_id]  # still stones are never deleted

            # Team (if sure) and where it was
            fact = {}
            team, sureness = stone.team()
            if sureness >= SURE_ENOUGH:
                fact["team"] = team
            spot_before = describe_spot(position_before)
            if spot_before:
                fact["before"] = spot_before

            if stone.status in ("out", "gone"):
                removed.append(fact)
            elif stone.status == "lost":
                moved.append(fact)
            elif np.linalg.norm(stone.resting_position - position_before) >= 0.12:
                spot_after = describe_spot(stone.resting_position)
                if spot_after:
                    fact["after"] = spot_after
                moved.append(fact)
        return removed, moved


def something_started_moving(changes):
    """A still stone got hit, or a stone came in moving."""
    for change, stone in changes:
        if change == "started" or stone.arrived_moving:
            return True
    return False


def name_the_shot(thrower, thrown_stone_end, removed, moved):
    """takeout, raise, hit, draw, guard, outside or through. None if unknown."""
    removed_opponent = any(is_opponent(fact, thrower) for fact in removed)
    touched_opponent = any(is_opponent(fact, thrower) for fact in removed + moved)
    raised_own_stone = any(fact.get("team") == thrower and ends_in_house(fact) for fact in moved)

    if removed_opponent:
        return "takeout"
    if removed or moved:
        if raised_own_stone and not touched_opponent:
            return "raise"
        return "hit"

    # Touched nothing: named after where it stopped
    if thrown_stone_end is None:
        return None
    zone = thrown_stone_end["zone"]
    if zone == "house":
        return "draw"
    if zone == "guard":
        return "guard"
    if zone == "outside":
        return "outside"
    return "through"  # out of play


def is_opponent(fact, thrower):
    """Known to be the other team's stone?"""
    team = fact.get("team")
    return team is not None and team != thrower


def ends_in_house(fact):
    """Pushed into the house?"""
    spot_after = fact.get("after")
    return spot_after is not None and spot_after["zone"] == "house"
