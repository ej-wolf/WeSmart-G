# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

""" code """
import os
from pathlib import Path





if __name__ == "__main__":

    CWD = Path(os.getcwd())
    # Example manual usage; adapt as needed
    json_dirs = ['json_ds','usual_jsons_from_cams', 'usual_jsons_from_events' ]
    json_dirs = ['Jsons']
    main_path = Path("/mnt/local-data/Projects/Wesmart/data/")
    main_path = CWD/"datasets"

    pass

