# TUNE YOLO DETECT MODEL
import os
from pathlib import Path
import ultralytics
from ultralytics import YOLO, RTDETR
import wandb
from dotenv import load_dotenv
import argparse
from pprint import pprint
import torch
import yaml

def setup(project_dir: Path, data_name: str = 'data') -> None:
    os.chdir(project_dir)
    
    load_dotenv()
    wandb.login(key = os.getenv('WANDB_KEY'))
    
def tune_with_wb(
    id: str,  
    model: str,  
    dataset: Path,  
    # tune_dir: str,
    project: str = 'capstone',
    epochs: int = 5,
    batch: int = 16,
    save: bool = True,
    exist_ok: bool = True,
    log: bool = True,
    iterations: int = 10,
    val: bool = False,
    **kwargs
) -> None:
    data_path = dataset / 'data.yaml'
    if log:
        log_mode = 'online'
    else:
        log_mode = 'offline'
    print(f'\n TUNING MODEL {id} ::::::::')
    with wandb.init(project = project, name = f'{id}_tune', magic = True, mode = log_mode) as run:
        model.tune(
            data = data_path,
            epochs = epochs,
            iterations = iterations,
            batch = batch,
            save = save,
            exist_ok = exist_ok,
            single_cls = True,
            amp = False,
            plots = False,
            val = val,
            name = f'{id}_tune', 
            # tune_dir = tune_dir,
            **kwargs
        )
    

def train_with_wb(
    id: str,  
    model: ultralytics.engine.model,
    dataset: Path,
    project: str = 'capstone',
    epochs: int = 5,
    batch: int = 16,
    save: bool = True,
    exist_ok: bool = True,
    log: bool = True,
    **kwargs
) -> dict:
    data_path = dataset / 'data.yaml'
    if log:
        log_mode = 'online'
    else:
        log_mode = 'offline'
    print(f'\n TRAINING MODEL {id} ::::::::')
    with wandb.init(project = project, name = f'{id}_best', magic = True, mode = log_mode) as run:
        res = model.train(
            data = data_path,
            epochs = epochs,
            batch = batch,
            save = save,
            exist_ok = exist_ok,
            single_cls = True,
            amp = False,  
            plots = False,
            name = f'{id}_best',
            **kwargs
        )
        recall = res.results_dict['metrics/recall(B)']
        map50 = res.results_dict['metrics/mAP50(B)']
        wandb.log({'recall': recall, 'map50': map50})
    return res
    
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
    prsr.add_argument('-v', '--val', action = 'store_true', help = 'Whether to validate after training')
    prsr.add_argument('-L', '--no_log', action = 'store_true', help = 'Do not post logs to wandb online')
    args = prsr.parse_args()
    pprint(args)
    
    PROJECT_DIR = Path(args.project_dir)
    data_dir = PROJECT_DIR / args.data_name
    dataset = data_dir / args.dataset
    
    # get model type based on run name
    model_id = args.run
    short_id = model_id.replace('_train', '')
    weights = PROJECT_DIR / 'runs/detect' / model_id / 'weights' / 'best.pt'
    model = get_model_type(model_id, args.model_type)(weights)
    # best_id = model_id.replace('train', 'best')
    # tune_dir = PROJECT_DIR / 'runs' / tune_id
    log = not args.no_log
    
    setup(PROJECT_DIR)
    
    tune_with_wb(
        id = short_id, 
        model = model, 
        dataset = dataset, 
        epochs = args.tune_epochs, 
        batch = args.batch,
        save = False,  
        iterations = args.iterations, 
        val = False,
        log = log,
        degrees = 15.0,
        hsv_h = 0.4,
        cos_lr = True,
        optimizer = 'AdamW')
    
    # need to get best params from tune based on last run
    if args.train:
        # get best params from tune
        last_run = PROJECT_DIR / 'runs/detect/tune' 
        best_yaml = last_run / 'best_hyperparameters.yaml'
        with open(best_yaml, 'r') as f:
            best_params = yaml.safe_load(f)
        best_res = train_with_wb(
            id = short_id,
            model = model,
            dataset = dataset,
            epochs = args.train_epochs,
            batch = args.batch,
            save = True,
            log = log,
            cos_lr = True,
            optimizer = 'AdamW',
            **best_params
        )
        pprint(best_res)
        
        if args.val:
            model.val()
    
    wandb.finish()
