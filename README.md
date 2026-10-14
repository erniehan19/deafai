# Curling commentary

Watches a curling clip and writes one sentence about each shot.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install yt-dlp ffmpeg
```

## Get the sample videos

Their configs are already in `configs/`.

```bash
yt-dlp -f "bv*[height<=1080][ext=mp4]" --download-sections "*20:09-20:17" --force-keyframes-at-cuts -o data/clips/game1.mp4 "https://www.youtube.com/live/p_dNi-xINK4"
yt-dlp -f "bv*[height<=1080][ext=mp4]" --download-sections "*34:07-34:25" --force-keyframes-at-cuts -o data/clips/game2.mp4 "https://www.youtube.com/live/n9VodfgmrpY"
yt-dlp -f "bv*[height<=1080][ext=mp4]" --download-sections "*43:47-44:06" --force-keyframes-at-cuts -o data/clips/game3.mp4 "https://www.youtube.com/watch?v=JTIDg9aVHCc"
yt-dlp -f "bv*[height<=1080][ext=mp4]" --download-sections "*44:22-44:38" --force-keyframes-at-cuts -o data/clips/game4.mp4 "https://www.youtube.com/watch?v=JTIDg9aVHCc"
yt-dlp -f "bv*[height<=1080][ext=mp4]" --download-sections "*45:09-45:18" --force-keyframes-at-cuts -o data/clips/game5.mp4 "https://www.youtube.com/watch?v=JTIDg9aVHCc"
```

## Add your own video

Pick a YouTube video with one steady camera above the house.

Download the part you want (this one grabs 45:09 to 45:18):

```bash
yt-dlp -f "bv*[height<=1080][ext=mp4]" \
  --download-sections "*45:09-45:18" --force-keyframes-at-cuts \
  -o data/clips/game6.mp4 "https://www.youtube.com/watch?v=..."
```

Then make its config file. Easiest way: ask Claude.

> "Make configs/game6.yaml for data/clips/game6.mp4"

## Play it

Commentary only:

```bash
.venv/bin/python src/main.py configs/game6.yaml
```

With the video and tracked stones on screen (press q to quit):

```bash
.venv/bin/python src/main.py configs/game6.yaml --track
```

## Demo
https://drive.google.com/file/d/1jWx-BgZ5ZcSU583CoNvKsyGypjsGAB1j/view?usp=drive_link
