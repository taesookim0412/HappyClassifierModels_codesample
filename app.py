from pathlib import Path
import os

def root():
    cwd = Path.cwd()
    project_root = os.path.abspath([new_parent for new_parent in cwd.parents if new_parent.parts[-1] == "HappyClassifierModels"][0])
    return project_root
