from __future__ import annotations

import os
import pickle
import subprocess
from pathlib import Path
# import run_safe_test
from my_local_utils import get_epoch_pth


def launch_wrapper(**kwargs):
    prfx = Path(kwargs.get('relative_path', '../..' ))
    # out_dir = prfx/kwargs.get('out_dir', kwargs['checkpoint_path'])
    score_file = (prfx/kwargs.get('out_dir', Path(kwargs['checkpoint_path']).parent/'test_eval')/
                       kwargs.get('score_file', "test_scores.pkl"))

    cmd = [ 'python', kwargs.get('wrapper','run_safe_test.py'), #* the wrapper script
            str(prfx/kwargs['config_path']),
            str(prfx/kwargs['checkpoint_path']),
            kwargs.get('pkl_flag', '--dump'), str(score_file),
            "--launcher", "none",]

    subprocess.run(cmd, check=True, cwd=kwargs.get('run_in','.'))


def set_with_defaults(run_tag:str, **kwargs)-> dict:


    pth = get_epoch_pth(Path(os.getcwd())/'work_dirs'/run_tag, kwargs.get('pth', 'best'))
    return {'wrapper': kwargs.get('testing_file','run_safe_test.py'),
            'config_path': f"configs/{run_tag}.py",
            'checkpoint_path': f"work_dirs/{run_tag}/{pth}"}



def print_metrics(metrics:dict):
    print(f"Samples  :{metrics['num_samples']};  Labels {metrics['support']}\n"
          f"Accuracy : {metrics['accuracy']:6f}\n"
          f"Recall   : {metrics['recall']:6f}\n"
          f"Confusion matrix: {metrics['confusion_matrix'][0]}\n"
          f"                  {metrics['confusion_matrix'][1]}")



if __name__ == "__main__":

    test_params = {'wrapper': 'run_safe_test.py',
                   'config_path': "configs/TRN/trn_r50_bbfrm_02.py",
                   'checkpoint_path': "work_dirs/tsn_R50_bbrfm/epoch_25.pth",
                   }
    test_params = {'wrapper': 'run_safe_test.py',
                   'config_path': "configs/tsm_R50_MMA_RWF.py",
                   'checkpoint_path': "work_dirs/tsm_R50_MMA_RLVS/best_acc_top1_epoch_5.pth",
                   }

    # test_params = set_with_defaults('tsm_r50_bbfrm')

    # launch_wrapper (**test_params)

#256( ,2,15)
