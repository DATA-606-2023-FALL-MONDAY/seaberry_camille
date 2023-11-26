# TRAIN YOLO MODELS
import os
import roboflow
from roboflow import Roboflow
from pathlib import Path
from ultralytics import YOLO, RTDETR
import wandb
from pathlib import Path
from dotenv import load_dotenv
import yaml
from ultralytics import settings as ul_settings
import argparse
from pprint import pprint
import torch


def setup(project_dir: Path, data_name: str = 'data') -> roboflow.core.project.Project:
    """Setup working directory, environment variables, ultralytics dataset directory, roboflow project, and wandb logging.

    Args:
        project_dir (Path): Base directory of the project.
        data_name (str, optional): Name of data directory. Defaults to 'data'.

    Returns:
        Project: A roboflow project object
    """    
    os.chdir(project_dir)
    
    load_dotenv()
    wandb.login(key = os.getenv('WANDB_KEY'))
    
    data_dir = project_dir / data_name
    ul_settings.update({ 'datasets_dir': str(data_dir) })
    print('\nULTRALYTICS SETTINGS::::')
    pprint(ul_settings)
    
    rf = Roboflow(api_key = os.getenv('ROBOFLOW_KEY'))
    # update project name
    proj = rf.workspace('seaberry').project('comb-detect')
    return proj

def download_rf_imgs(proj: roboflow.core.project.Project, 
                     v: int, 
                     data_dir: str | Path, 
                     format: str = 'yolov8', 
                     overwrite: bool = False) -> roboflow.core.dataset.Dataset:
    """Download images from a roboflow project, given its version number.

    Args:
        proj (Project): roboflow project
        v (int): Version number to download
        data_dir (str | Path): Directory to download train, valid, and test images
        format (str, optional): Download format. Defaults to 'yolov8'.
        overwrite (bool, optional): Whether to overwrite folders. Defaults to False.

    Returns:
        Dataset: A roboflow dataset object
    """    
    dataset = proj.version(v).download(model_format = format, location = str(data_dir), overwrite = overwrite)
    return dataset

def fix_data_yaml(dir: str) -> None:
    """Paths to image directories have been inconsistent. This overwrites the datasets' yaml files 
    to give them consistent paths such as train/images, rather than longer relative ones.

    Args:
        dir (str): Directory containing train, val, and test folders.
    """    
    dir = Path(dir)
    with open(dir / 'data.yaml', 'r') as f:
        data = yaml.safe_load(f)
    for split in ['train', 'val', 'test']:
        long_dir = Path(data[split])
        data[split] = long_dir.parent.name + '/' + long_dir.name
    with open(dir / 'data.yaml', 'w') as f:
        yaml.dump(data, f)
        
def prep_datasets(proj: roboflow.core.project.Project, 
                  versions: list,
                  dirs: list,
                  ids: list,
                  overwrite: bool = False) -> dict:
    """Download roboflow datasets, both full and tiled versions, fix their yaml, and return them.

    Args:
        proj (Project): A roboflow project object
        versions (list, optional): Version numbers to download. Defaults to [1, 3].
        dirs (list, optional): Directories to download to. Defaults to ['data/cams_full', 'data/cams_tile'].
        ids (list, optional): Dataset names to use as keys. Defaults to ['full', 'tile'].
        overwrite (bool, optional): Whether to overwrite folders. Defaults to False.

    Returns:
        dict: A dict of Dataset objects.
    """    
    datasets = {}
    
    for i, v in enumerate(versions):
        data_dir = Path(ul_settings['datasets_dir']) / dirs[i]
        id = ids[i]
        dataset = download_rf_imgs(proj=proj,
                                   v=v,
                                   data_dir=data_dir,
                                   overwrite=overwrite)
        fix_data_yaml(dataset.location)
        datasets[id] = dataset
    
    return datasets

def model_with_wb(
    id: str,
    model: str,
    weights: str,
    dataset: roboflow.core.dataset.Dataset,
    project: str = 'capstone',
    imgsz: int = 640,
    patience: int = 10,
    epochs: int = 5,
    batch: int = 16,
    freeze: int = 0,
    save: bool = True,
    exist_ok: bool = True,
    **kwargs
) -> None:
    """Train a YOLO model and log to wandb. In order to not crash, this clears the cuda cache after each run.

    Args:
        id (str): ID for this run, used for naming run folders.
        model (str): Type of model to train, either 'yolo' or 'detr'.
        weights (str): Name of weights file to use.
        dataset (roboflow.Dataset): Dataset to use for training
        project (str, optional): Name of project on wandb. Defaults to 'capstone'.
        
    Args to pass to YOLO trainer:
        imgsz (int, optional): Image size. Defaults to 640.
        patience (int, optional): Number of epochs to wait before early stopping. Defaults to 10.
        epochs (int, optional): Number of epochs. Defaults to 5.
        batch (int, optional): Batch size. Defaults to 16.
        freeze (int, optional): Number of layers to freeze. Defaults to 0.
        save (bool, optional): Whether to save artifacts of training. Defaults to True.
        exist_ok (bool, optional): Whether to overwrite a previous run of the same name. Defaults to True.
        **kwargs: Additional named args to pass to YOLO trainer.
    """    
    data_path = f'{dataset.location}/data.yaml'
    print(f'\n TRAINING MODEL {id} ::::::::')
    with wandb.init(project = project, name = id) as run:
        if model == 'yolo':
            model = YOLO(weights)
        elif model == 'detr':
            model = RTDETR(weights)
        else: 
            raise ValueError(f'Invalid model {model}')
        model.train(data = data_path,
                        imgsz = imgsz,
                        patience = patience,
                        epochs = epochs,
                        batch = batch,
                        freeze = freeze,
                        save = save,
                        exist_ok = exist_ok,
                        single_cls = True,
                        lr0 = 1e-3,
                        # optimizer = 'AdamW',
                        degrees = 15,
                        cos_lr = True,
                        amp = False,
                        name = f'{id}_train',
                        **kwargs)
        # clear torch cache
        # torch.cuda.empty_cache()

if __name__ == '__main__':
    prsr = argparse.ArgumentParser(prog = 'train_yolo.py', description = 'Train YOLO models on roboflow datasets.')
    prsr.add_argument('-p', '--project_dir', type = str, default = './', help = 'Project directory')
    prsr.add_argument('-d', '--data_name', type = str, default = 'data', help = 'Name of data directory')
    prsr.add_argument('-e', '--epochs', type = int, default = 40, help = 'Number of epochs')
    prsr.add_argument('-b', '--batch', type = int, default = 16, help = 'Batch size')
    prsr.add_argument('-o', '--overwrite', action = 'store_true', help = 'Overwrite existing datasets')
    prsr.add_argument('-a', '--use_plain', action = 'store_true', default = True, help = 'Include basic runs (no freezing or tiling)')
    prsr.add_argument('-z', '--use_freeze', action = 'store_true', help = 'Include runs with frozen layers')
    prsr.add_argument('-f', '--freeze', type = int, default = 20, help = 'Number of layers to freeze')
    prsr.add_argument('-t', '--use_tile', action = 'store_true', help = 'Train on tiled images')
    args = prsr.parse_args()
    pprint(args)
    
    # needs at least one type of run
    if not args.use_plain and not args.use_freeze and not args.use_tile:
        raise ValueError('Must include at least one type of run')
    # set project directory and setup project
    PROJECT_DIR = Path(args.project_dir)
    proj = setup(PROJECT_DIR, args.data_name)
    datasets = prep_datasets(proj, 
                             versions = [8, 9],
                             dirs = ['cams_full', 'cams_tile'],
                             ids = ['full', 'tile'],
                             overwrite = args.overwrite)
    
    # define params for runs
    weights = { 'yolo': 'yolov8s.pt', 'detr': 'rtdetr-l.pt'}
    base_params = { 'epochs': args.epochs, 'batch': args.batch }
    params = {}
    
    if args.use_plain:
        params['yolo_full'] = { 'dataset': datasets['full'], 'model': 'yolo', 'weights': weights['yolo'] }
        params['detr_full'] = { 'dataset': datasets['full'], 'model': 'detr', 'weights': weights['detr'] }
    
    if args.use_tile:
        params['yolo_tile'] = { 'dataset': datasets['tile'], 'model': 'yolo', 'weights': weights['yolo'] }
        params['detr_tile'] = { 'dataset': datasets['tile'], 'model': 'detr', 'weights': weights['detr'] }
    
    if args.use_freeze:
        params['yolo_full_frz'] = { 'dataset': datasets['full'], 'model': 'yolo', 'weights': weights['yolo'], 'freeze': args.freeze }
        params['detr_full_frz'] = { 'dataset': datasets['full'], 'model': 'detr', 'weights': weights['detr'], 'freeze': args.freeze }
    
    if args.use_tile and args.use_freeze:
        params['yolo_tile_frz'] = { 'dataset': datasets['tile'], 'model': 'yolo', 'weights': weights['yolo'], 'freeze': args.freeze }
        params['detr_tile_frz'] = { 'dataset': datasets['tile'], 'model': 'detr', 'weights': weights['detr'], 'freeze': args.freeze }
        
    for id, ps in params.items():
        run_params = { **base_params, **ps }
        # check whether id contains the string 'detr'--cut batch since it needs more ram
        if 'detr' in id:
            run_params['batch'] = int(run_params['batch'] / 2)
        pprint(run_params)
        model_with_wb(id, **run_params)
    
    wandb.finish()


