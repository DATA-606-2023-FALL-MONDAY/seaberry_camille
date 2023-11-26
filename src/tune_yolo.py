# TUNE YOLO DETECT MODEL
import os
from pathlib import Path
from ultralytics import YOLO, RTDETR
import wandb
from dotenv import load_dotenv
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
    val: bool = False,
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
        val = val,
        name = id, 
        # tune_dir = tune_dir,
        **kwargs
    )
    
def get_model_type(run: str, model_type: str):
    if model_type is not None:
        model_type = model_type 
    else:
        model_type = run.split('_')[0]
    if model_type == 'yolo':
        model = YOLO
    elif model_type == 'detr':
        model = RTDETR
    else:
        raise ValueError(f'Invalid model type {model_type}')
    return model

if __name__ == '__main__':
    # take arg of which model to use
    prsr = argparse.ArgumentParser()
    prsr.add_argument('run', type = str, help = 'Name of run to tune---must be in runs/detect')
    prsr.add_argument('-m', '--model_type', type = str, help = 'Type of model used, either yolo or detr; if None, inferred from run name')
    prsr.add_argument('-D', '--dataset', type = str, help = 'Name of dataset, which contains a data.yaml file')
    prsr.add_argument('-p', '--project_dir', type = str, default = './', help = 'Project directory')
    prsr.add_argument('-d', '--data_name', type = str, default = 'data', help = 'Name of data directory')
    prsr.add_argument('-e', '--tune_epochs', type = int, default = 10, help = 'Number of epochs for tuning')
    prsr.add_argument('-E', '--train_epochs', type = int, default = 40, help = 'Number of epochs for training')
    prsr.add_argument('-b', '--batch', type = int, default = 16, help = 'Batch size')
    prsr.add_argument('-i', '--iterations', type = int, default = 10, help = 'Number of iterations for tuning')
    prsr.add_argument('-t', '--train', action = 'store_true', help = 'Whether to train after tuning')
    args = prsr.parse_args()
    
    PROJECT_DIR = Path(args.project_dir)
    
    # get model type based on run name
    model_id = args.run
    weights = PROJECT_DIR / 'runs/detect' / model_id / 'weights' / 'best.pt'
    model = get_model_type(model_id, args.model_type)(weights)
    tune_id = model_id.replace('train', 'tune')
    dataset = args.dataset
    
    setup(PROJECT_DIR)
    
    base_params = { 'optimizer': 'AdamW' }
    tune_res = tune_with_wb(id = tune_id, 
                 model = model, 
                 data_yaml = PROJECT_DIR / args.data_name / dataset / 'data.yaml', 
                 tune_dir = PROJECT_DIR / 'runs' / tune_id,
                 epochs = args.tune_epochs, 
                 iterations = args.iterations, 
                 save = False,  
                 batch = args.batch,
                 val = False,
                 **base_params)
    print(tune_res)
    
    if args.train:
        # get best params from tune
        best_params = tune_res['best_params']
        # train with best params
        best_res = model.train(name = f'best_{model_id}',
                               data = PROJECT_DIR / args.data_name / dataset / 'data.yaml',
                               amp = False,
                               batch = args.batch,
                               epochs = args.train_epochs,
                               optimizer = 'AdamW',
                               **best_params)
    
        print(best_res)
