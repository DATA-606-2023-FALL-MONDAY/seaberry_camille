# TRAIN YOLO MODELS
import os
import roboflow
from roboflow import Roboflow
from pathlib import Path
import ultralytics
from ultralytics import YOLO, RTDETR
import wandb
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import settings as ul_settings
import argparse
from pprint import pprint
import shutil
import warnings

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
    proj = rf.workspace('seaberry').project('cap-class')
    
    # ignore warnings with corrupt image
    warnings.filterwarnings(action = 'ignore', module = 'ultralytics')
    return proj

def download_rf_imgs(proj: roboflow.core.project.Project, 
                     v: int, 
                     data_dir: str | Path, 
                     format: str = 'folder', 
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

def fix_data_dirs(dir: str) -> None:
    """Paths to image directories have been inconsistent. This changes the name of a "valid" folder to "val".

    Args:
        dir (str): Directory containing train, val, and test folders.
    """    
    dir = Path(dir)
    # check whether valid folder exists
    if (dir / 'valid').exists():
        # rename it to val
        shutil.move(dir / 'valid', dir / 'val')
        # shutil.rmtree(dir / 'valid')
        
def prep_datasets(proj: roboflow.core.project.Project, 
                  versions: list,
                  dirs: list,
                  ids: list,
                  overwrite: bool = False) -> dict:
    """Download roboflow datasets, both full and tiled versions, fix their yaml, and return them.

    Args:
        proj (Project): A roboflow project object
        versions (list, optional): Version numbers to download. Defaults to [2].
        dirs (list, optional): Directories to download to. Defaults to ['cams_crop'].
        ids (list, optional): Dataset names to use as keys. Defaults to ['crop'].
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
        fix_data_dirs(dataset.location)
        datasets[id] = dataset
    
    return datasets

def model_with_wb(
    id: str,
    model: YOLO | RTDETR,
    dataset: roboflow.core.dataset.Dataset,
    project: str = 'capstone',
    imgsz: int = 32,
    patience: int = 0,
    epochs: int = 5,
    batch: int = 16,
    freeze: int = 0,
    save: bool = True,
    exist_ok: bool = True,
    log: bool = True,
    **kwargs
) -> None:
    """Train a YOLO model and log to wandb.

    Args:
        id (str): ID for this run, used for naming run folders.
        model (YOLO | RTDETR): Model to train
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
    data_path = f'{dataset.location}' # no data.yaml--that's just for detection
    if log:
        log_mode = 'online'
    else:
        log_mode = 'offline'
    print(f'\n TRAINING MODEL {id} ::::::::')
    with wandb.init(project = project, name = id, mode = log_mode, job_type = 'classify') as run:
        model.train(data = data_path,
                    imgsz = imgsz,
                    patience = patience,
                    epochs = epochs,
                    batch = batch,
                    freeze = freeze,
                    save = save,
                    exist_ok = exist_ok,
                    close_mosaic = 0,
                    cos_lr = True,
                    amp = False,
                    name = f'{id}_train',
                    **kwargs)

def main(
    project_dir: str,  
    data_name: str,  
    epochs: int,   
    batch: int,
    overwrite: bool,
    use_freeze: bool,
    freeze: int | None,
    no_log: bool
) -> None:
    PROJECT_DIR = Path(project_dir)
    proj = setup(PROJECT_DIR, data_name)
    log = not no_log
    datasets = prep_datasets(proj = proj,
                             versions = [2],
                             dirs = ['cams_crop'],
                             ids = ['crop'],
                             overwrite = overwrite)
    wts = { 'yolo_s': 'yolov8s-cls.pt', 'yolo_m': 'yolov8m-cls.pt', 'yolo_l': 'yolov8l-cls.pt' }
    base_params = { 'epochs': epochs, 'batch': batch }
    params = {}
    
    for key, wt in wts.items():
        mod = f'{key}_class'
        params[mod] = { 'dataset': datasets['crop'], 'model': YOLO(wt, task = 'classify') }
    
    for key, ps in params.items():
        run_params = { **base_params, **ps }
        model_with_wb(key, log = log, **run_params)
        
    wandb.finish()

if __name__ == '__main__':
    prsr = argparse.ArgumentParser(prog = 'train_yolo.py', description = 'Train YOLO models on roboflow datasets.')
    prsr.add_argument('-p', '--project_dir', type = str, default = './', help = 'Project directory')
    prsr.add_argument('-d', '--data_name', type = str, default = 'data', help = 'Name of data directory')
    prsr.add_argument('-e', '--epochs', type = int, default = 40, help = 'Number of epochs')
    prsr.add_argument('-b', '--batch', type = int, default = 16, help = 'Batch size')
    prsr.add_argument('-o', '--overwrite', action = 'store_true', help = 'Overwrite existing datasets')
    prsr.add_argument('-z', '--use_freeze', action = 'store_true', help = 'Include runs with frozen layers')
    prsr.add_argument('-f', '--freeze', type = int, default = 7, help = 'Number of layers to freeze')
    prsr.add_argument('-L', '--no_log', action = 'store_true', help = 'Do not post logs to wandb')
    args = prsr.parse_args()
    pprint(args)
    
    main(**vars(args))


