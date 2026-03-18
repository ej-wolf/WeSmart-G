# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

""" code """
import os
from pathlib import Path
# from json_processing import json_to_box_frames, process_json_folder #, play_frames
from visual_util import play_frames
# from  temporal_segment_BB_frames import  segment_all_clips, make_labels_file, make_train_val_ds
from my_local_utils import collection, clear_dir


" Default config, later should be move to designated config"
DRAW_CENTER = True


def play_all_clips(frames_root, x: float = 1.0, event_color=False):
    """ Iterate over all sub dirs under frames_root and play their frames.
    frames_root: root directory containing per-clip frame folders.
    x: playback speed factor (1.0 = original, 2.0 = 2x faster, 0.5 = 2x slower).
    event_color: passed directly to play_frames (False/ True/ (r,g,b)).
    """
    frames_root = Path(frames_root)

    if not frames_root.is_dir():
        print(f"[ERROR] {frames_root} is not a directory")
        return

    # Iterate over subdirs in sorted order
    clip_dirs = sorted(p for p in frames_root.iterdir() if p.is_dir())
    if not clip_dirs:
        print(f"[INFO] No subdirectories found under {frames_root}")
        return

    for clip_dir in clip_dirs:
        print(f"\n=== Playing clip: {clip_dir.name} ===")
        play_frames(clip_dir, x=x, event_color=event_color)


if __name__ == "__main__":

    CWD = Path(os.getcwd())
    # Example manual usage; adapt as needed
    json_dirs = ['json_ds','usual_jsons_from_cams', 'usual_jsons_from_events' ]
    json_dirs = ['Jsons']
    main_path = Path("/mnt/local-data/Projects/Wesmart/data/")
    main_path = Path("/mnt/local-data/Projects/Wesmart/datasets/")

    pass
