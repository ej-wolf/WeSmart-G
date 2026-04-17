#* run_test_safe.py   (place in project root: weSmart/)
import os
from pathlib import Path
import numpy as np
import subprocess
from torch.serialization import add_safe_globals
from mmengine.logging.history_buffer import HistoryBuffer

from numpy.dtypes import Float64DType, Int64DType

#* Robust import for NumPy's _reconstruct (from numpy._core.multiarray) as np_reconstruct
try:
    #* "Public" path – should exist on modern NumPy
    from numpy.core.multiarray import _reconstruct as np_reconstruct
except Exception:
    #* Fallback to the internal path seen in the pickle error
    from numpy._core.multiarray import _reconstruct as np_reconstruct


def main():
    # Allow this class to be unpickled by torch's safe loader
    add_safe_globals([HistoryBuffer, np_reconstruct, getattr,
                      np.ndarray, np.dtype,
                      np._core.multiarray.scalar,
                      Float64DType, Int64DType])

    #* Locate the mmaction repo and test.py
    repo_root = Path(__file__).resolve().parent
    mmaction_root = repo_root/"extern/mmaction2"
    test_file = mmaction_root/"tools/test.py"

    if not test_file.is_file():
        raise FileNotFoundError(f"Cannot find {test_file}")

    #* Change CWD so mmaction imports (import mmaction, etc.) work as before
    os.chdir(mmaction_root)
    #* Execute tools/test.py in *this* process as __main__
    code = compile(test_file.read_text(encoding="utf-8"), str(test_file), "exec")
    globals_dict = { "__name__": "__main__", "__file__": str(test_file), }
    #* Add for debugging
    print("code:\n", code, "globals:\n", globals_dict)
    # print(globals_dict)

    exec(code, globals_dict)


def run_train(cfg_f:str|Path, **kwargs):

    cfg_f = Path(kwargs.get('cfg_dir', '../../configs' ))/cfg_f
    # " python tools/train.py ../../configs/tsm_R50_MMA_RWF.py
    cmd = [ 'python', 'tools/train.py', str(cfg_f)]
    subprocess.run(cmd, check=True, cwd=kwargs.get('run_in', "../extern/mmaction2"))




def launch_wrapper(**kwargs):
    prfx = Path(kwargs.get('relative_path', '../../..'))
    # out_dir = prfx/kwargs.get('out_dir', kwargs['checkpoint_path'])
    score_file = (prfx/kwargs.get('out_dir', Path(kwargs['checkpoint_path']).parent/'test_eval')/
                       kwargs.get('score_file', "test_scores.pkl"))

    cmd = [ 'python', kwargs.get('', 'wrapper/run_safe_test.py'),  #* the wrapper script
            str(prfx/kwargs['config_path']),
            str(prfx/kwargs['checkpoint_path']),
            kwargs.get('pkl_flag', '--dump'), str(score_file),
            "--launcher", "none", ]

    subprocess.run(cmd, check=True, cwd=kwargs.get('run_in', '..'))



if __name__ == "__main__":
    main()
    cfg_f = "tsm_R50_MMA_RWF.py"
    # run_train(cfg_f)


#49(2,5,3)
