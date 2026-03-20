import json, random
from collections.abc import Iterable
from pathlib import Path


#*  .pth files Utils
# -----------------------------------------------------------------------------
def get_epoch_pth(dir_path: str | Path, epoch: int | str | None = 'best') -> str | None:
    """Pick a checkpoint from `dir_path` by epoch policy.

    `epoch='best'`: newest file containing 'best' in its name.
    `epoch='last'`: highest `epoch_XX.pth`.
    `epoch=int`: exact match if exists, else closest available epoch file.
    """

    def _epoch_of(name: str) -> int | None:
        if name.startswith('epoch_') and name.endswith('.pth'):
            num_part = name[len('epoch_'):-len('.pth')]
            try:
                return int(num_part)
            except ValueError:
                return None

    def all_pairs():
        epc_pairs = [(p.name, _epoch_of(p.name)) for p in pth_files]
        return [(p, e) for p, e in epc_pairs if e is not None]

    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: {dir_path}')

    pth_files = sorted(dir_path.glob('*.pth'))
    if not pth_files:
        print(f"[Error] No pth files found in {dir_path}")
        return None

    if (isinstance(epoch, str) and epoch == 'best') or epoch is None:
        pth = [p for p in pth_files if 'best' in p.name]
        if pth:
            return str(max(pth, key=lambda p: p.stat().st_mtime).name)
        print("[WARN] No best checkpoint found, falling back to last epoch.")
        epoch = 'last'

    if isinstance(epoch, str) and epoch == 'last':
        epoch_pairs = all_pairs()
        if not epoch_pairs:
            return str(max(pth_files, key=lambda p: p.stat().st_mtime))
        last_pth, _ = max(epoch_pairs, key=lambda pe: pe[1])
        return last_pth

    if not isinstance(epoch, int):
        print(f"Unsupported epoch spec: {epoch!r}")

    pth = f'epoch_{epoch}.pth'
    if (dir_path / pth).is_file():
        return pth

    epoch_pairs = all_pairs()
    closest_pth, _ = min(epoch_pairs, key=lambda pe: (abs(pe[1] - epoch), -pe[1]))
    return str(closest_pth)

def tst_get_pth(tst_path: str | None = None, **kwargs):
    """ Print checkpoint resolution examples for a test directory."""
    tst_ls = kwargs.get('tst_ls', [None, 'best', 'last', 10, 25, 17, 51, 81, -1])
    d = Path(tst_path)
    for t in tst_ls:
        print(get_epoch_pth(d, t))
    d = d.parent
    print(get_epoch_pth(d))
    print(get_epoch_pth(d, epoch=10))


#*  JSON Utils
# -----------------------------------------------------------------------------

def _normalize_keys(keys):
    """Return keys as a list. Accepts None, str, or iterable."""
    if keys is None:
        return None
    if isinstance(keys, str):
        return [keys]
    if isinstance(keys, Iterable):
        return list(keys)
    return [keys]


def _find_first_key_value(obj, target_key):
    """Find first occurrence of `target_key` by DFS and return (found, value)."""
    if isinstance(obj, dict):
        if target_key in obj:
            return True, obj[target_key]
        for val in obj.values():
            found, out = _find_first_key_value(val, target_key)
            if found:
                return True, out
    elif isinstance(obj, list):
        for item in obj:
            found, out = _find_first_key_value(item, target_key)
            if found:
                return True, out
    return False, None


def _json_files_equal(f1: str | Path, f2: str | Path, keys=None) -> bool:
    """Compare two JSON files.

    If `keys` is None: compare full JSON objects.
    If `keys` is set: compare only first-found subtrees for each key.
    """
    with open(f1) as a, open(f2) as b:
        j1, j2 = json.load(a), json.load(b)

    keys = _normalize_keys(keys)
    if not keys:
        return j1 == j2

    for key in keys:
        found1, v1 = _find_first_key_value(j1, key)
        found2, v2 = _find_first_key_value(j2, key)
        if not (found1 and found2):
            return False
        if v1 != v2:
            return False
    return True


def compare_json_dirs(d1, d2, soft_compare=False, **kwargs):
    """Compare same-name JSON files in two directories.

    `soft_compare=False` requires identical filename sets.
    `soft_compare=True` compares only common filenames.
    Pass `keys=[...]` to compare selected key subtrees only.
    """
    d1, d2 = Path(d1), Path(d2)

    if not d1.is_dir():
        return False, f"can't find 1st path: {d1.name}"
    if not d2.is_dir():
        return False, f"can't find 2nd path: {d2.name}"

    files1 = {p.name: p for p in d1.rglob('*.json')}
    files2 = {p.name: p for p in d2.rglob('*.json')}

    if not soft_compare:
        if set(files1.keys()) != set(files2.keys()):
            return False, 'Different file sets'
        compare_list = files1.keys()
    else:
        compare_list = set(files1.keys()) & set(files2.keys())

    if not compare_list:
        return False, 'No common JSON files to compare'

    list1 = [files1[name] for name in compare_list]
    list2 = [files2[name] for name in compare_list]

    stats = compare_json_samples(list1, list2,1.0,
                                 keys=kwargs.get('keys'),
                                 print_list=kwargs.get('print_list', False),
                                 print_summary=False,
                                 return_stats=True,
                                 stop_on_mismatch=kwargs.get('break', False),
                                )

    if stats['first_mismatch'] and kwargs.get('break', False):
        return False, f"Mismatch in {stats['first_mismatch']}"

    unequal_count = stats['found'] - stats['identical']
    if unequal_count > 0:
        return False, f" {stats['identical']} out of {stats['found']} are equal"

    return True, "Directories identical (based on comparison mode)"


def compare_json_samples(l1, l2, smp, **kwargs):
    """Compare a random sample of same-name JSON files.

    `smp` can be ratio (0..1) or absolute count.
    Pass `keys=[...]` to compare selected key subtrees only.
    """
    list1 = list(Path(l1).rglob('*.json')) if isinstance(l1, (str, Path)) else l1
    list2 = list(Path(l2).rglob('*.json')) if isinstance(l2, (str, Path)) else l2

    if not list1:
        print('No files in first input')
        return

    if isinstance(smp, float) and (0 <= smp <= 1):
        n_draw = max(1, int(len(list1) * smp)) if smp > 0 else 0
    elif isinstance(smp, int) and 0 < smp:
        n_draw = min(int(smp), len(list1))
    else:
        raise ValueError('smp must a float between 0 and 1 or positive int ')

    sampled = random.sample(list1, n_draw) if n_draw > 0 else []
    map2 = {p.name: p for p in list2}

    if kwargs.get('print_list', False):
        print("\t File: \t\t |\tFound\t|\tEqual ")
    found, identical = 0, 0
    first_mismatch = None
    for p1 in sampled:
        p2 = map2.get(Path(p1).name)
        eql = False
        if p2:
            found += 1
            eql = _json_files_equal(p1, p2, keys=kwargs.get('keys'))
            if eql:
                identical += 1
            elif first_mismatch is None:
                first_mismatch = Path(p1).name
                if kwargs.get('stop_on_mismatch', False):
                    break
        if kwargs.get('print_list', False):
            print(f"{Path(p1).name:20}- {'Yes' if p2 else '---'}\t{eql} ")

    percent = 100 * (identical / found) if found > 0 else 0
    if kwargs.get('print_summary', True):
        print(f"\nComparison result:\n {n_draw} files drawn\n"
              f" {found} found in second set\n  {identical} identical ({percent:.1f}%)")

    if kwargs.get('return_stats', False):
        return {'n_draw': n_draw, 'found': found, 'identical': identical,
                'first_mismatch': first_mismatch, 'percent': percent, }


def test_json_comparison( dir1: str | Path, dir2: str | Path,  frame_numbers=(0, 1),  one_based=True):
    """ Run 3 checks on same-name file pairs: full JSON, `frames`, selected frame indices. """
    dir1, dir2 = Path(dir1), Path(dir2)
    files1 = {p.name: p for p in dir1.rglob("*.json")}
    files2 = {p.name: p for p in dir2.rglob("*.json")}

    common_names = sorted(set(files1.keys()) & set(files2.keys()))
    only_1 = sorted(set(files1.keys()) - set(files2.keys()))
    only_2 = sorted(set(files2.keys()) - set(files1.keys()))

    full_equal = []
    full_diff = []
    frames_equal = []
    frames_diff = []
    selected_equal = []
    selected_diff = {}

    idxs = [(n - 1) if one_based else n for n in frame_numbers]
    labels = list(frame_numbers)

    for name in common_names:
        with open(files1[name]) as a, open(files2[name]) as b:
            j1, j2 = json.load(a), json.load(b)

        if j1 == j2:
            full_equal.append(name)
        else:
            full_diff.append(name)

        fr1, fr2 = j1.get("frames"), j2.get("frames")
        if fr1 == fr2:
            frames_equal.append(name)
        else:
            frames_diff.append(name)

        bad = []
        if not isinstance(fr1, list) or not isinstance(fr2, list):
            bad = labels[:]  # can't index non-list frames payload
        else:
            for idx, label in zip(idxs, labels):
                if idx < 0 or idx >= len(fr1) or idx >= len(fr2):
                    bad.append(label)
                elif fr1[idx] != fr2[idx]:
                    bad.append(label)

        if bad:
            selected_diff[name] = bad
        else:
            selected_equal.append(name)

    print("\n====== test_json_comparison ======\n")
    print(f"dir 1: {dir1}\ndir 2:{dir2}")
    print(f"\nCommon files: {len(common_names)}")
    if only_1:
        print(f"only in dir1: {len(only_1)}")
    if only_2:
        print(f"only in dir2: {len(only_2)}")

    print(f"\n1) Full JSON equality\nequal: {len(full_equal)/len(common_names)}")
    print(f"\n2) 'frames' field equality \nequal: {len(frames_equal)/len(common_names)}")
    print(f"\n3) Selected frames equality {tuple(labels)}\nequal: {len(selected_equal)/len(common_names)}")

    if full_diff:
        print(f"\nfull-json mismatches (first 10): {full_diff[:10]}")
    if frames_diff:
        print(f"frames mismatches (first 10): {frames_diff[:10]}")
    if selected_diff:
        first10 = list(selected_diff.items())[:10]
        print(f"selected-frame mismatches (first 10): {first10}")

    return {'common_files': common_names,
            'only_dir1': only_1, 'only_dir2': only_2,
            'full_equal': full_equal, 'full_diff': full_diff,
            'frames_equal': frames_equal, 'frames_diff': frames_diff,
            'selected_equal': selected_equal, 'selected_diff': selected_diff,
            }

if __name__ == "__main__":

    tst_d1 = Path("/mnt/local-data/Python/Projects/weSmart/data/json_files/RWF-2000/V-ds_01")
    tst_d2 = Path("/mnt/local-data/Python/Projects/weSmart/data/json_files/RWF-2000/V-ds_02")
    test_json_comparison(tst_d1, tst_d2, frame_numbers=(6, 11, 22), one_based=True)

#310(,,1)
