"""Run the pipeline on one clip and print the commentary.

    python src/main.py configs/game2.yaml           # print sentences
    python src/main.py configs/game2.yaml --show    # also play the video (q quits)
    python src/main.py configs/game2.yaml --track   # play it zoomed in, with the tracked stones drawn
"""

import sys
import time

import cv2
import numpy as np
import yaml

from commentary import Commentator
from detect import DEFAULT_TEAMS, Camera, StoneDetector
from track import Board, ShotTracker


def clock(seconds):
    """Seconds -> m:ss.s"""
    minutes, seconds = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes)}:{seconds:04.1f}"


# Drawing colors (blue, green, red) for each team name
TEAM_DRAW_COLORS = {"red": (0, 0, 255), "yellow": (0, 230, 255), "blue": (255, 120, 0)}


def draw_tracking(image, detector, board, stones_found):
    """Zoom into the area we look at, and mark what the tracker sees."""
    left, top, width, height = detector.camera.view_area
    view = image[top : top + height, left : left + width].copy()

    # Raw detections: small dots
    for detection in stones_found:
        pixel_x, pixel_y = detector.camera.to_pixels([detection.position])[0]
        color = TEAM_DRAW_COLORS.get(detection.team, (255, 255, 255))
        cv2.circle(view, (int(pixel_x - left), int(pixel_y - top)), 3, color, -1)

    # Tracked stones: ring + label
    for stone in board.stones_in_play():
        pixel_x, pixel_y = detector.camera.to_pixels([stone.position])[0]
        center = (int(pixel_x - left), int(pixel_y - top))
        radius = int(1.4 * detector.stone_radius_in_pixels(pixel_x, pixel_y))
        team, _ = stone.team()
        color = TEAM_DRAW_COLORS.get(team, (255, 255, 255))
        cv2.circle(view, center, radius, color, 2)
        label = f"#{stone.id} {stone.status} {stone.speed:.2f}"
        label_spot = (center[0] - radius, center[1] - radius - 4)  # just above the stone
        cv2.putText(view, label, label_spot, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA)  # dark outline
        cv2.putText(view, label, label_spot, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return view


def show(image, caption, wait_ms):
    """Show a frame with the caption below. False when q is pressed."""
    scale = min(1280 / image.shape[1], 720 / image.shape[0])  # fit on a laptop screen
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    caption_strip = np.zeros((60, image.shape[1], 3), np.uint8)
    cv2.putText(caption_strip, caption, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Curling commentary", np.vstack([image, caption_strip]))
    key = cv2.waitKey(max(1, int(wait_ms))) & 0xFF
    return key not in (ord("q"), 27)  # q or Esc


def main(config_path, show_video=False, show_tracking=False):
    # Setup
    with open(config_path) as file:
        config = yaml.safe_load(file)
    video = cv2.VideoCapture(config["video"])
    if not video.isOpened():
        sys.exit(f"Can't open the video {config['video']}")
    frames_per_second = video.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    teams = config.get("teams", DEFAULT_TEAMS)
    camera = Camera(config["calibration"], frame_width, frame_height, config.get("roi"))
    detector = StoneDetector(camera, teams)
    board = Board(camera)
    shot_tracker = ShotTracker()
    commentator = Commentator(teams, use_llm=config.get("use_llm", False))

    # One pass per frame
    print(f"Watching {config['video']} ...")
    frame_number = 0
    started = time.perf_counter()
    caption = ""
    while True:
        frame_started = time.perf_counter()
        ok, image = video.read()
        if not ok:
            break
        video_time = frame_number / frames_per_second
        frame_number += 1
        if frame_number % round(10 * frames_per_second) == 0:  # progress every 10 s
            print(f"  ... {clock(frame_number / frames_per_second)}", flush=True)

        stones = detector.find_stones(image)
        changes = board.update(video_time, stones)
        shot = shot_tracker.update(video_time, board, changes)  # None unless a shot just ended

        if shot and shot["facts"] is None:
            print(f"[{clock(shot['stopped_at'])}] (no commentary: {shot['no_commentary_reason']})")
        elif shot:
            asked_at = time.perf_counter()
            sentence, source = commentator.comment(shot["facts"])
            delay = video_time - shot["stopped_at"] + time.perf_counter() - asked_at  # time since stones stopped
            caption = f"[{clock(shot['stopped_at'])}] {sentence}"
            print(f"{caption}   ({source}, {delay:.1f}s after the stones stopped)")

        if show_tracking:
            image = draw_tracking(image, detector, board, stones)
        if show_video or show_tracking:
            time_left_ms = 1000 / frames_per_second - 1000 * (time.perf_counter() - frame_started)  # normal speed
            if not show(image, caption, time_left_ms):
                break

    took = time.perf_counter() - started
    print(f"Done: {frame_number} frames in {took:.0f}s ({frame_number / max(took, 1e-9):.0f} frames/s).")


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if arg not in ("--show", "--track")]
    if len(args) != 1:
        sys.exit("usage: python src/main.py configs/<clip>.yaml [--show | --track]")
    main(args[0], show_video="--show" in sys.argv, show_tracking="--track" in sys.argv)
