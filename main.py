# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import os
from pathlib import Path
from data_processing import json_to_box_frames, process_json_folder, play_frames
import data_processing as jsn
from  temporal_segment import  segment_all_clips
from my_local_utils import collection


" Default config, later should be move to designated config"
DRAW_CENTER = True

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def test_data_proc():
    json_path = Path("/mnt/local-data/Projects/Wesmart/data/json")
    frames_path = Path("/mnt/local-data/Python/Projects/weSmart/data/frames")
    # json_to_box_frames(json_path/"8_1.json", frames_path, draw_center=True)
    process_json_folder(json_path, frames_path, draw_center=True)
    # json_to_box_frames(json_path/"8_1.json", frames_path, draw_center=True)


def process_data(raw_data_path, data_root:Path = None):

    raw_data_path = collection(raw_data_path)
    data_root = CWD/'data' if data_root is  None else Path(data_root)

    for path in raw_data_path:
        process_json_folder(path, Path(data_root)/'frames',
                            draw_center=DRAW_CENTER)

    return
    segments = segment_all_clips(data_root/'frames', data_root/'cache')


    # cwd = Path(os.getcwd())
    # src_path = cwd/'data'/'frames'
    # dst_path = cwd/'cache'

    ann_path = data_root/'cache'/'all_windows.txt'
    with ann_path.open("w", encoding="utf-8") as f:
        for name, n_frames, label in segments:
            f.write(f"{name} {n_frames} {label}\n")
    print(f"[INFO] Wrote annotations to {ann_path}")




def play_all_clips(frames_root, x: float = 1.0, event_color=False):
    """ Iterate over all sub dirs under frames_root and play their frames.
    frames_root: root directory containing per-clip frame folders.
    x: playback speed factor (1.0 = original, 2.0 = 2x faster, 0.5 = 2x slower).
    event_color: passed directly to play_frames (False / True / (r,g,b)).
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
    json_path = [Path("/mnt/local-data/Projects/Wesmart/data/json"),
                 Path("/mnt/local-data/Projects/Wesmart/data/jsons_corrected")]
    # data_root = Path("/mnt/local-data/Python/Projects/weSmart/data/frames")
    data_root = CWD/"data/frames"
    #
    # # json_to_box_frames(json_path/"8_1.json", frames_path, draw_center=True)
    # process_json_folder(json_path, data_root, draw_center=True)

    process_data(json_path)
    play_all_clips(data_root, x=3.0, event_color=True)


# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('weSmart')
#
#     # test_data_proc()
#     play_all_clips()

    pass




