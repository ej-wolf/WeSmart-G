# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

""" code """
import os
from pathlib import Path
from json_processing import json_to_box_frames, process_json_folder #, play_frames
from visual_util import play_frames
import json_processing as jsn
from  temporal_segment import  segment_all_clips, make_labels_file, make_train_val_ds
from my_local_utils import collection, clear_dir


" Default config, later should be move to designated config"
DRAW_CENTER = True

def test_data_proc():
    json_path = Path("/mnt/local-data/Projects/Wesmart/data/json")
    frames_path = Path("/mnt/local-data/Python/Projects/weSmart/data/frames")
    # json_to_box_frames(json_path/"8_1.json", frames_path, draw_center=True)
    process_json_folder(json_path, frames_path, draw_center=True)


def process_data(json_path, data_root:Path = None, sub_dirs=None):

    json_path = collection(json_path)
    data_root = CWD/'data' if data_root is  None else Path(data_root)
    data_root = data_root/'frames'
    data_root.mkdir(exist_ok=True)

    if sub_dirs and len(json_path) == len(sub_dirs):
        data_paths = zip(json_path, sub_dirs)
    else:
        data_paths = zip(json_path, ['.']*len(json_path))

    # for path in json_path:
    for j, f in data_paths:
        # print(f" Input : {j} \n Output: {data_root/f}")
        process_json_folder(j, data_root/f, use_file_name=True, draw_center=DRAW_CENTER)
    return

    segments = segment_all_clips(data_root/'frames', data_root/'cache')

    ann_path = data_root/'cache'/'all_windows.txt'
    with ann_path.open("w", encoding="utf-8") as f:
        for name, n_frames, label in segments:
            f.write(f"{name} {n_frames} {label}\n")
    print(f"[INFO] Wrote annotations to {ann_path}")



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


    #* Set the main data dir inside the python project
    data_root = CWD/"data"

    #* Convret jsons to frames
    #process_data(json_path)
    json_path_ls = [main_path/d for d in json_dirs]
    # process_data(json_path_ls, sub_dirs=['train', 'test', 'test'])
    process_data(json_path_ls)

    #* Make segmentations
    clear_dir(data_root/'cache')
    segments = segment_all_clips(data_root/'frames', data_root/'cache', win_len=15, stride=5)
    # segments = segment_all_clips(data_root/'frames'/'train', data_root/'cache', win_len=15, stride=5)
    # tst_seg = segment_all_clips(data_root/'frames'/'test' , data_root/'cache', win_len=15, stride=5)

    make_labels_file(segments)
    # make_labels_file(tst_seg)
    make_train_val_ds(segments, data_root/'cache')
    # play_all_clips(data_root/'frames'/'test', x=3.0, event_color=True)

    #* crete primitive test
    # make_labels_file(segments,data_root/'labels'/'fake_test_01.txt')

    pass




