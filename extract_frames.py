"""
extract_frames.py
-----------------
Extracts frames from memory model videos for Plan A+C visualization.

For each video listed in MEMORY_VIDEOS:
  - 6 timeline strip frames at evenly spaced timestamps  → strip_N.jpg
  - 1 early "memory reference" frame (t ≈ 3s)           → early.jpg
  - 1 late  "current scene" frame   (t ≈ 85% duration)  → late.jpg

Output: static/imgs/memory_frames/<key>_strip_N.jpg, <key>_early.jpg, <key>_late.jpg
        where <key> = folder + "_" + basename, e.g. "30s_031", "1min_091"

Usage:
    python extract_frames.py
"""

import cv2
import os

# Thumbnails are resized to this width (keeps aspect ratio)
THUMB_W = 320

# Strip: number of evenly-spaced frames to capture per video
N_STRIP = 6

# ── Videos to process ────────────────────────────────────────────────────────
MEMORY_VIDEOS = {
    # key          : path
    # First-person 30s
    "30s_031"  : "static/videos/30s/031.mp4",
    "30s_010_2"  : "static/videos/30s/010_2.mp4",
    "30s_004"  : "static/videos/30s/004.mp4",
    "30s_076"  : "static/videos/30s/076.mp4",
    "30s_002"  : "static/videos/30s/002.mp4",
    "30s_021"  : "static/videos/30s/021.mp4",
    "30s_091"  : "static/videos/30s/091.mp4",
    "30s_025"  : "static/videos/30s/025.mp4",
    # First-person 1min
    "1min_091" : "static/videos/1min/091.mp4",
    "1min_094" : "static/videos/1min/094.mp4",
    "1min_097" : "static/videos/1min/097.mp4",
    "1min_000" : "static/videos/1min/000.mp4",
    "1min_001" : "static/videos/1min/001.mp4",
    # Third-person
    # "tp_91"  : "static/videos/tp_demo/video_lingbot-I2V-A14B_91_action.mp4",
    # "tp_99"  : "static/videos/tp_demo/video_lingbot-I2V-A14B_99_action.mp4",
    # "tp_112" : "static/videos/tp_demo/video_lingbot-I2V-A14B_112_action.mp4",
    # "tp_34"  : "static/videos/tp_demo/video_lingbot-I2V-A14B_34_action.mp4",
    # "tp_48"  : "static/videos/tp_demo/video_lingbot-I2V-A14B_48_action.mp4",
    # "tp_54"  : "static/videos/tp_demo/video_lingbot-I2V-A14B_54_action.mp4",
    # "tp_61"  : "static/videos/tp_demo/video_lingbot-I2V-A14B_61_action.mp4",
    # "tp_72"  : "static/videos/tp_demo/video_lingbot-I2V-A14B_72_action.mp4",
    # "tp_80"  : "static/videos/tp_demo/video_lingbot-I2V-A14B_80_action.mp4",
    # "tp_7"   : "static/videos/tp_demo/video_lingbot-I2V-A14B_7_action.mp4",
}

OUT_DIR = "static/imgs/memory_frames"
os.makedirs(OUT_DIR, exist_ok=True)


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    new_h = int(h * width / w)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def extract_frame_at(cap, total_frames, t_ratio):
    """Seek to t_ratio ∈ [0,1] of the video and return the frame."""
    pos = int(total_frames * t_ratio)
    pos = max(0, min(pos, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    return frame if ret else None


def process_video(key, path):
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return

    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps

    # ── Strip frames (6 evenly spaced) ────────────────────────────────────
    strip_ratios = [i / (N_STRIP - 1) for i in range(N_STRIP)]
    for i, ratio in enumerate(strip_ratios):
        frame = extract_frame_at(cap, total_frames, ratio)
        if frame is not None:
            thumb = resize_to_width(frame, THUMB_W)
            out_path = os.path.join(OUT_DIR, f"{key}_strip_{i}.jpg")
            cv2.imwrite(out_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    # import pdb; pdb.set_trace()
    # ── Early frame (memory reference, t ≈ 3s or 5% of duration) ─────────
    t_early = min(3.0 / duration, 0.05) if duration > 0 else 0.03 #duration=31.0625
    frame_early = extract_frame_at(cap, total_frames, t_early)
    if frame_early is not None:
        thumb = resize_to_width(frame_early, THUMB_W)
        cv2.imwrite(os.path.join(OUT_DIR, f"{key}_early.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ── Late frame (current scene at 95% of duration) ────────────────────
    frame_late = extract_frame_at(cap, total_frames, 0.95)
    if frame_late is not None:
        thumb = resize_to_width(frame_late, THUMB_W)
        cv2.imwrite(os.path.join(OUT_DIR, f"{key}_late.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

    cap.release()
    dur_str = f"{duration:.1f}s"
    print(f"  [OK] {key:12s}  ({dur_str:8s})  strip ×{N_STRIP}  early  late")


def main():
    print(f"Extracting frames → {OUT_DIR}/\n")
    for key, path in MEMORY_VIDEOS.items():
        process_video(key, path)
    print("\nDone.")


if __name__ == "__main__":
    main()
