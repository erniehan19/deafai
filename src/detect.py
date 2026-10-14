"""Find stones in a frame and place them on the ice (in meters).

Ice coordinates: (0, 0) is the button, +y is away from the thrower, +x is the thrower's right.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Sheet sizes, in meters
FOOT = 0.3048
STONE_RADIUS = 0.145
BUTTON_RADIUS = 0.5 * FOOT
FOUR_FOOT_RADIUS = 2 * FOOT
EIGHT_FOOT_RADIUS = 4 * FOOT
HOUSE_RADIUS = 6 * FOOT
BACK_LINE_Y = HOUSE_RADIUS
HOG_LINE_Y = -21 * FOOT
HALF_SHEET_WIDTH = 4.75 / 2

# Calibration spots on the ice (thrower's left/right)
CALIBRATION_SPOTS = {
    "front": (0.0, -HOUSE_RADIUS),
    "back": (0.0, HOUSE_RADIUS),
    "left": (-HOUSE_RADIUS, 0.0),
    "right": (HOUSE_RADIUS, 0.0),
    "button": (0.0, 0.0),  # optional
}

# Handle colors as HSV ranges: (h, s, v low, h, s, v high)
DEFAULT_TEAMS = {
    "red": [(0, 90, 70, 10, 255, 255), (170, 90, 70, 179, 255, 255)],  # red wraps around hue 0
    "yellow": [(18, 90, 90, 38, 255, 255)],
}


@dataclass
class Detection:
    """One stone seen in one frame."""

    position: np.ndarray  # meters
    team: str


class Camera:
    """Converts between pixels and meters."""

    def __init__(self, calibration: dict, frame_width: int, frame_height: int, view_area=None):
        spot_names = [name for name in CALIBRATION_SPOTS if name in calibration]
        if len(spot_names) < 4:
            raise ValueError(f"calibration needs at least 4 of {list(CALIBRATION_SPOTS)}, got {spot_names}")
        spots_in_pixels = np.float64([calibration[name] for name in spot_names])
        spots_in_meters = np.float64([CALIBRATION_SPOTS[name] for name in spot_names])
        self.pixels_to_meters, _ = cv2.findHomography(spots_in_pixels, spots_in_meters)
        self.meters_to_pixels = np.linalg.inv(self.pixels_to_meters)

        # Part of the frame we look at: (left, top, width, height)
        self.view_area = tuple(view_area) if view_area else (0, 0, frame_width, frame_height)

    def to_meters(self, pixel_points) -> np.ndarray:
        """Pixels -> meters."""
        return convert(pixel_points, self.pixels_to_meters)

    def to_pixels(self, meter_points) -> np.ndarray:
        """Meters -> pixels."""
        return convert(meter_points, self.meters_to_pixels)

    def can_see(self, position, margin: float = 8) -> bool:
        """Is this spot inside the area we look at?"""
        pixel_x, pixel_y = self.to_pixels([position])[0]
        left, top, width, height = self.view_area
        inside_x = left + margin <= pixel_x <= left + width - margin
        inside_y = top + margin <= pixel_y <= top + height - margin
        return inside_x and inside_y


def convert(points, matrix):
    """Apply a pixels <-> meters matrix to a list of (x, y) points."""
    points = np.float64(points).reshape(-1, 1, 2)  # shape OpenCV wants
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)


class StoneDetector:
    """Finds stones by their red or yellow handles."""

    def __init__(self, camera: Camera, teams: dict):
        self.camera = camera
        self.teams = teams

    def find_stones(self, image: np.ndarray) -> list[Detection]:
        """All stones on our sheet in one frame."""
        left, top, width, height = self.camera.view_area
        view = image[top : top + height, left : left + width]
        hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)

        stones = []
        for team in self.teams:
            mask = self.team_color_mask(hsv, team)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # remove specks
            blob_count, _, blob_stats, blob_centers = cv2.connectedComponentsWithStats(mask)  # group into blobs

            for blob in range(1, blob_count):  # blob 0 is the background
                blob_width = blob_stats[blob, cv2.CC_STAT_WIDTH]
                blob_height = blob_stats[blob, cv2.CC_STAT_HEIGHT]
                blob_area = blob_stats[blob, cv2.CC_STAT_AREA]
                center_x = float(blob_centers[blob][0] + left)
                center_y = float(blob_centers[blob][1] + top)

                if not self.looks_like_handle(blob_width, blob_height, blob_area, center_x, center_y):
                    continue

                position = self.camera.to_meters([(center_x, center_y)])[0]
                if not on_our_sheet(position):
                    continue

                stones.append(Detection(position, team))
        return stones

    def looks_like_handle(self, blob_width, blob_height, blob_area, center_x, center_y):
        """Right size and shape for a handle?"""
        radius = self.stone_radius_in_pixels(center_x, center_y)
        handle_area = 0.55 * np.pi * radius * radius  # a handle covers about half the stone

        too_small = blob_area < 0.5 * handle_area
        too_big = blob_area > 2.0 * handle_area
        too_patchy = blob_area < 0.5 * blob_width * blob_height
        too_long = max(blob_width, blob_height) > 2.5 * min(blob_width, blob_height)
        return not (too_small or too_big or too_patchy or too_long)

    def team_color_mask(self, hsv, team):
        """White where the pixel is this team's color."""
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for h_low, s_low, v_low, h_high, s_high, v_high in self.teams[team]:
            mask |= cv2.inRange(hsv, (h_low, s_low, v_low), (h_high, s_high, v_high))
        return mask

    def stone_radius_in_pixels(self, pixel_x, pixel_y):
        """Stone radius in pixels at this spot (smaller further away)."""
        center = self.camera.to_meters([(pixel_x, pixel_y)])[0]
        edge = center + [STONE_RADIUS, 0.0]
        center_pixels, edge_pixels = self.camera.to_pixels([center, edge])
        return float(np.linalg.norm(center_pixels - edge_pixels))


# --- Where is a stone ---------------------------------------------------------------------------
SAFE_MARGIN = 0.056  # meters a stone must be from a line before we name its side of it
HOUSE_EDGE = HOUSE_RADIUS + STONE_RADIUS  # stone touches the house inside this

# (ring, inner distance, outer distance) of a stone's center from the button
RINGS = [
    ("button", 0, BUTTON_RADIUS + STONE_RADIUS),
    ("four-foot", BUTTON_RADIUS + STONE_RADIUS, FOUR_FOOT_RADIUS + STONE_RADIUS),
    ("eight-foot", FOUR_FOOT_RADIUS + STONE_RADIUS, EIGHT_FOOT_RADIUS + STONE_RADIUS),
    ("twelve-foot", EIGHT_FOOT_RADIUS + STONE_RADIUS, HOUSE_RADIUS),
    ("biting", HOUSE_RADIUS, HOUSE_RADIUS + STONE_RADIUS),
]


def on_our_sheet(position) -> bool:
    """Between the hog line and the back boards, and not off the side."""
    x, y = position
    return HOG_LINE_Y - 1.0 <= y <= BACK_LINE_Y + 1.5 and abs(x) <= HALF_SHEET_WIDTH + 0.3


def is_out_of_play(position) -> bool:
    """Past the back line, or touching the side boards."""
    x, y = position
    return y - STONE_RADIUS > BACK_LINE_Y or abs(x) + STONE_RADIUS >= HALF_SHEET_WIDTH


def describe_spot(position) -> dict | None:
    """Spot in curling terms, e.g. {"zone": "house", "ring": "four-foot", "depth": "top", "side": "left"}.

    Leaves out anything too close to call. None if even the zone is unclear.
    """
    x, y = position
    distance_from_button = np.hypot(x, y)

    if near_zone_line(x, y):
        return None
    if is_out_of_play(position):
        return {"zone": "out of play"}

    if distance_from_button > HOUSE_EDGE:
        if abs(y) < SAFE_MARGIN:
            return None  # on the tee line: can't tell guard from outside
        if y < 0:
            spot = {"zone": "guard"}
        else:
            spot = {"zone": "outside"}
    else:
        spot = {"zone": "house"}
        ring = ring_name(distance_from_button)
        if ring:
            spot["ring"] = ring
        depth = depth_name(y)
        if depth:
            spot["depth"] = depth

    side = side_name(x)
    if side:
        spot["side"] = side
    return spot


def near_zone_line(x, y) -> bool:
    """Too close to the house edge, back line or side boards to be sure?"""
    distance_from_button = np.hypot(x, y)
    near_house_edge = abs(distance_from_button - HOUSE_EDGE) < SAFE_MARGIN
    near_back_line = abs(y - BACK_LINE_Y - STONE_RADIUS) < SAFE_MARGIN
    near_side_boards = abs(abs(x) - HALF_SHEET_WIDTH + STONE_RADIUS) < SAFE_MARGIN
    return near_house_edge or near_back_line or near_side_boards


def ring_name(distance_from_button) -> str | None:
    """Which ring. None if too close to a ring line."""
    for name, inner, outer in RINGS:
        if inner <= distance_from_button <= outer:
            if outer - distance_from_button < SAFE_MARGIN:
                return None
            if inner > 0 and distance_from_button - inner < SAFE_MARGIN:  # the button has no inner line
                return None
            return name
    return None


def depth_name(y) -> str | None:
    """Top, tee line or back. None if too close to call."""
    if abs(abs(y) - STONE_RADIUS) < SAFE_MARGIN:
        return None
    if abs(y) <= STONE_RADIUS:
        return "tee line"
    if y < 0:
        return "top"
    return "back"


def side_name(x) -> str | None:
    """Left, center or right. None if too close to call."""
    if abs(abs(x) - STONE_RADIUS) < SAFE_MARGIN:
        return None
    if abs(x) <= STONE_RADIUS:
        return "center"
    if x < 0:
        return "left"
    return "right"
