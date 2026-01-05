#* run_test_safe.py   (place in project root: weSmart/)
import os
import sys
from pathlib import Path

import subprocess
from torch.serialization import add_safe_globals
from mmengine.logging.history_buffer import HistoryBuffer
import numpy as np
from numpy.dtypes import Float64DType, Int64DType

# ---- module-level constants ----
DEFAULT_SCORE = "test-score.pkl"
MMACTION_PATH = Path("extern/mmaction2")
# MMACTION_TEST = MMACTION_PATH/"tools/test.py"
MMACTION_TEST = "tools/test.py"

#* Robust import for NumPy's _reconstruct (from numpy._core.multiarray) as np_reconstruct
try:
    #* "Public" path – should exist on modern NumPy
    from numpy.core.multiarray import _reconstruct as np_reconstruct
except Exception:
    #* Fallback to the internal path seen in the pickle error
    from numpy._core.multiarray import _reconstruct as np_reconstruct


def main():
    """ Wrapper around mmaction2/tools/test.py with robust path handling.
    Design goals:
    - config + checkpoint are mandatory
    - Make NO assumption about current working directory
    - paths may be:
        absolute,  relative to CWD or relative to the script location
    - --dump rules as previously defined
    """

    # Allow this class to be unpickled by torch's safe loader
    add_safe_globals([  HistoryBuffer, np_reconstruct, getattr,
                        np.ndarray, np.dtype, np._core.multiarray.scalar,
                        Float64DType, Int64DType  ])

    argv = sys.argv
    launch_cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent

    # print(argv)
    # return
    if len(argv) < 3:
        raise RuntimeError("Usage: python run_test_safe.py <config.py> <checkpoint.pth> [--dump scores.pkl]")

    # ---- robust path resolution (no cwd assumptions) ----
    def resolve_existing(p:str, what:str) -> Path:
        p = Path(p)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([launch_cwd/p, script_dir/p,])
        #* all paths are converted to absolute
        # for c in candidates:
        #     c = c.resolve()
        #     if c.is_file():
        #         return c
        return next((c.resolve() for c in candidates if c.resolve().is_file()), None)
        raise FileNotFoundError(f"{what} not found. Tried: " + ", ".join(str(c) for c in candidates))

    cfg_path  = resolve_existing(argv[1], "Config file")
    ckpt_path = resolve_existing(argv[2], "Checkpoint file")

    argv[1] = str(cfg_path)
    argv[2] = str(ckpt_path)

    #* ---- handle --dump (optional, extended rules) ----
    # semantics:
    # --dump <dir>       -> <dir>/test-score.pkl
    # --dump <file.pkl>  -> use as-is
    # --dump no|0        -> disable dumping
    # no --dump          -> default: <ckpt_dir>/test-score.pkl

    dump_path = None
    # dump_disabled = False
    # if "--dump" in argv:
    #     i = argv.index("--dump")
    #     if i + 1 >= len(argv): #* --dump without value -> raise error
    #         raise RuntimeError("--dump specified without value")
    #     val = argv[i + 1]
    #     if val.lower() in {'no', 'No', '0'}:
    #         dump_disabled = True
    #         argv.pop(i + 1)
    #         argv.pop(i)
    #     else:
    #         dump_path = Path(val)
    # else:
    #     dump_path = ckpt_path.parent/DEFAULT_SCORE

    if '--dump' not in argv:
        dump_path = ckpt_path.parent/DEFAULT_SCORE
    elif  argv.index('--dump')+1 >= len(argv):
        argv.pop()  #* --dump without value -> remove it
    else:
        i = argv.index('--dump')
        if argv[i+1].lower() in {'0', 'no', 'none'}:
            #* --dump no|No|0  ->  remove dumping
            del argv[i:i+2]
        else:
            #* --dump <dir>|<*.pkl>
            dump_path = Path(argv[i+1])

    if dump_path is not None: #  and not dump_disabled:
        if not dump_path.is_absolute():
            dump_path = (launch_cwd/dump_path).resolve()
        if dump_path.suffix == "" or dump_path.is_dir():
            dump_path = dump_path/DEFAULT_SCORE

        parent = dump_path.parent
        if not parent.exists():
            if not parent.parent.exists():
                raise FileNotFoundError(f"Failed create dump directory, {parent.parent} does not exist")
            parent.mkdir()

        if "--dump" in argv:
            j = argv.index("--dump")
            argv[j + 1] = str(dump_path)
        else:
            argv.extend(["--dump", str(dump_path)])


    # ---- locate mmaction2 root by walking upwards ----
    mmaction_root = None
    for base in [script_dir, launch_cwd]:
        for p in [base] + list(base.parents):
            candidate = p/MMACTION_PATH/MMACTION_TEST
            if candidate.is_file():
                mmaction_root = candidate.parent.parent
                break
        if mmaction_root:
            break

    if mmaction_root is None:
        raise FileNotFoundError("Cannot locate extern/mmaction2/tools/test.py from script or CWD")

    test_file = mmaction_root/"tools/test.py"

    # Change CWD so mmaction imports work
    os.chdir(mmaction_root)

    code = compile(test_file.read_text(encoding="utf-8"), str(test_file), "exec")
    globals_dict = {"__name__": "__main__", "__file__": str(test_file)}

    exec(code, globals_dict)


def run_train(cfg_f:str|Path, **kwargs):

    cfg_f = Path(kwargs.get('cfg_dir', '../../configs' ))/cfg_f
    # " python tools/train.py ../../configs/tsm_R50_MMA_RWF.py
    cmd = [ 'python', 'tools/train.py', str(cfg_f)]
    subprocess.run(cmd, check=True, cwd=kwargs.get('run_in', "extern/mmaction2"))


def launch_wrapper(**kwargs):
    prfx = Path(kwargs.get('relative_path', '../..' ))
    # out_dir = prfx/kwargs.get('out_dir', kwargs['checkpoint_path'])
    score_file = (prfx/kwargs.get('out_dir', Path(kwargs['checkpoint_path']).parent/'test_eval')/
                       kwargs.get('score_file', DEFAULT_SCORE))

    cmd = [ 'python', kwargs.get('wrapper','run_safe_test.py'), #* the wrapper script
            str(prfx/kwargs['config_path']),
            str(prfx/kwargs['checkpoint_path']),
            kwargs.get('pkl_flag', '--dump'), str(score_file),
            "--launcher", "none",]

    subprocess.run(cmd, check=True, cwd=kwargs.get('run_in','.'))


if __name__ == "__main__":
    main()

#188(2,8,3)
