""" json_utils
Utilities for loading JSON annotation files into a unified,
firm internal structure.

Public API:
    load_json_data(file: str | Path, format: str = 'type1') -> dict

Returned structure:

json_data = {'header': {'video_file': ...,
                        'fps': ...,
                        'sampling': ...},
             'frames': [{'f': ...,
                        't': ...,
                        'group_events': ...,
                        'detections_list': [{'class': ...,
                                             'conf': ...,
                                             'bbox': [...],
                                             'key_pts': [17 pts x 3]},
                                            ...]
                        }, ...]
             }
"""

from pathlib import Path
import json

from my_local_utils import print_color
# --------------------------------------------------
# * Public loader
# --------------------------------------------------

def load_json_data(file:str|Path, j_type='type_1'):
    """ Load JSON file and normalize into firm internal structure.
        Param:
        file str|Path: Path to JSON file
        j_type str: Loader type ('type1', 'type2', ...)
        Returns
        json_data dict: Unified structured representation
    """
    file = Path(file)

    try:
        if j_type == 'type_1':
            return load_type_1(file)
        elif j_type in ['type_2', '2', 2]:
            return load_type_2(file)
        else:
            print_color(f"Warning: Unknown Json format: {j_type}", 'y')
            return None
            #raise ValueError(f"Unknown JSON format: {format}")
    except :
        raise ValueError(f"Error: Failed to load {file.name};  format: {j_type}")



# * Type 1 loader (current format)
def load_type_1(file: Path):
    """   Load JSON type 1 (old) files  """
    # * old format had 2 variations for detection_list
    DET_LIST = 'bbs_list_of_keypoints'  # 'list_of_bbs_keypoints'

    with open(file, 'r') as f:
        raw = json.load(f)

    header = {'video_file': raw.get('video'),
              'fps': raw.get('fps'),
              'sampling': raw.get('step'),
              'version': '1.0'}

    frames_out = []
    for frame in raw.get('frames', []):
        detections = []

        for bb in frame.get(DET_LIST, []):
            det = {'class': bb[0],'conf': bb[1],
                   'bbox': bb[2:6], 'key_pts': bb[6]}
            detections.append(det)

        frame_struct = {'f': frame.get('f'),
                        't': frame.get('t'),
                        'group_events': frame.get('group_events', []),
                        'detections_list': detections,
                        }
        frames_out.append(frame_struct)

    return {'header': header, 'frames': frames_out}
    # return json_data


#* Type 2 loader (future format placeholder)
def load_type_2(file: Path):
    """ Load JSON files in the new format (type2).
    Differences vs type1:
    - 'detection_list' instead of 'list_of_bbs_keypoints'
    - Each detection is already a dict
    - 'key_points' contains 17 keypoints * 3 values (x, y, conf) flattened as [x0, y0, c0, x1, y1, c1, ..., x16, y16, c16]
    """
    with open(file, 'r') as f:
        raw = json.load(f)

    header = {'video_file': raw['video'], 'fps': raw['fps'], 'sampling': raw['step'], 'version': '2.0'}

    frames_out = []
    for frame in raw.get('frames', []):
        detections = []

        for det in frame.get('detection_list', []):
            key_pts_raw = det.get('key_points',[])
            # Ensure list length consistency (should be 51 = 17 * 3)
            if key_pts_raw and len(key_pts_raw) % 3 != 0:
                print_color(f"[WARN] key_points length not divisible by 3 in frame {frame.get('f')}",'y')

            #det_struct = { 'class': det['class'],
            detections.append({'class': det['class'],
                               'conf': det['conf'],
                               'bbox': det.get('bbox', []),
                               'key_pts': key_pts_raw, #* keep flattened format for compatibility with other code
                               })
            # detections+= [det_struct]

        frame_struct = {'f': frame.get('f'),
                        't': frame.get('t'),
                        'group_events': frame.get('group_events', []),
                        'detections_list': detections,
                        }
        frames_out += [frame_struct]

    # json_data = {'header': header, 'frames': frames_out}
    return {'header': header, 'frames': frames_out}

#129(,,4) -> 88(,,2)
