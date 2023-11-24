# TUNE YOLO DETECT MODEL
import os
from pathlib import Path
from ultralytics import YOLO
import wandb
from dotenv import load_dotenv
import yaml
import argparse
from pprint import pprint
import torch

def setup(project_dir: Path, data_name: str = 'data') -> None:
    os.chdir(project_dir)
    
    load_dotenv()
    wandb.login(key = os.getenv('WANDB_KEY'))
    
def tune_with_wb(
    id: str,  
    model: str,  
    data_yaml: str,  
    tune_dir: str,
    project: str = 'capstone',
    epochs: int = 5,
    batch: int = 16,
    iterations: int = 10,
    save: bool = True,
    exist_ok: bool = True,
    **kwargs
) -> dict:
    model.tune(
        data = data_yaml,
        epochs = epochs,
        iterations = iterations,
        batch = batch,
        save = save,
        exist_ok = exist_ok,
        single_cls = True,
        cos_lr = True,
        amp = False,
        plots = False,
        name = id, 
        tune_dir = tune_dir,
        **kwargs
    )

if __name__ == '__main__':
    # take arg of which model to use
    PROJECT_DIR = Path('/home/camille/code/capstone')
    setup(PROJECT_DIR)
    model_id = 'yolo_tile_train'
    tune_id = model_id.replace('train', 'tune')
    dataset = 'cams_full'
    base_params = { 'optimizer': 'AdamW', 'degrees': 15 }
    model = YOLO(PROJECT_DIR / 'runs/detect' / model_id / 'weights' / 'best.pt')
    tune_with_wb(id = tune_id, 
                 model = model, 
                 data_yaml = PROJECT_DIR / 'data' / dataset / 'data.yaml', 
                 tune_dir = PROJECT_DIR / 'runs' / tune_id,
                 epochs = 2, 
                 iterations = 5, 
                 save = False,  
                 batch = 8,
                 **base_params)
    


# from ultralytics import YOLO 
# import yaml
# from pathlib import Path
# PROJECT_DIR = Path('/home/camille/code/capstone')
# best_yolo = YOLO(PROJECT_DIR / 'runs/detect/tune/weights/best.pt')
# with open(PROJECT_DIR / 'runs/detect/tune/best_hyperparameters.yaml') as f:
#     best_params = yaml.load(f, Loader = yaml.FullLoader)
# best_res = best_yolo.train(name = 'best_yolo_train',
#                            data = PROJECT_DIR / 'data/cams_full/data.yaml',
#                            amp = False,
#                            batch = 8,
#                            epochs = 10,
#                            optimizer = 'AdamW',
#                            **best_params)

from pathlib import Path
from ultralytics import YOLO
PROJECT_DIR = Path('/home/camille/code/capstone')
best_tile = YOLO(PROJECT_DIR / 'runs/detect/yolo_tile_train/weights/best.pt')
best_tile.train(name = 'tile_to_full_train', 
                data = '/home/camille/code/capstone/data/cams_full/data.yaml',
                amp = False,  
                freeze = 21,
                batch = 8, 
                epochs = 5,
                optimizer = 'AdamW',
                single_cls = True)