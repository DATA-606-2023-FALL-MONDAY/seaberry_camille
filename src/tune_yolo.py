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
    wandb.login(key=os.getenv('WANDB_KEY'))


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
    task: str = 'detect',
    **kwargs
) -> str:
    if task == 'detect':
        data_path = dataset / 'data.yaml'
        single_cls = True
    else: 
        data_path = dataset
        single_cls = False
        
    if log:
        log_mode = 'online'
    else:
        log_mode = 'offline'
        
    print(f'\n TUNING MODEL {id} ::::::::')
    with wandb.init(job_type=f'{id}_tune', mode=log_mode, reinit=True, project = project) as run:
        model.tune(
            data=data_path,
            epochs=epochs,
            iterations=iterations,
            batch=batch,
            save=save,
            exist_ok=exist_ok,
            single_cls=single_cls,
            amp=False,
            plots=False,
            val=val,
            name=f'{id}_tune',
            # tune_dir = tune_dir,
            **kwargs
        )
        print(run)
    # last_run = ultralytics.utils.files.get_latest_run()
    # print(f'\n LAST RUN: {last_run}')
    # return Path(last_run).parent.parent
    


def train_with_wb(
    id: str,
    model: ultralytics.engine.model,
    dataset: Path,
    project: str = 'capstone',
    epochs: int = 5,
    batch: int = 16,
    patience: int = 10,
    save: bool = True,
    exist_ok: bool = True,
    log: bool = True,
    task: str = 'detect',
    **kwargs
) -> dict:
    if task == 'detect':
        data_path = dataset / 'data.yaml'
        single_cls = True
    else: 
        data_path = dataset
        single_cls = False
        
    if log:
        log_mode = 'online'
    else:
        log_mode = 'offline'
    print(f'\n TRAINING MODEL {id} ::::::::')
    with wandb.init(name=f'{id}_best', job_type='best', mode=log_mode, reinit=True) as run:
        res = model.train(
            data=data_path,
            epochs=epochs,
            batch=batch,
            patience=patience,
            save=save,
            save_period=10,
            exist_ok=exist_ok,
            single_cls=single_cls,
            amp=False,
            plots=True,
            name=f'{id}_best',
            **kwargs
        )
        wandb.log(res.results_dict)
        # recall = res.results_dict['metrics/recall(B)']
        # map50 = res.results_dict['metrics/mAP50(B)']
        # wandb.log({'recall': recall, 'map50': map50, **kwargs})
    return res


def get_model_type(run: str, model_type: str | None):
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


def main(
    run: str,
    model_type: str,
    dataset: str,
    project_dir: str,
    data_name: str,
    tune_epochs: int,
    train_epochs: int,
    batch: int,
    iterations: int,
    train: bool,
    val: bool,
    no_log: bool,
    label: str,
    task: str
):
    PROJECT_DIR = Path(project_dir)
    setup(PROJECT_DIR)
    data_dir = PROJECT_DIR / data_name
    dataset = data_dir / dataset

    # get model type based on run name
    model_id = run
    if label is None:
        short_id = model_id.replace('_train', '')
    else:
        short_id = label
    weights = PROJECT_DIR / 'runs' / task / model_id / 'weights' / 'best.pt'
    # creates e.g. YOLO('train/weights/best.pt')
    model = get_model_type(model_id, model_type)(str(weights))
    print(model.info())

    log = not no_log

    tune_with_wb(
        id=short_id,
        model=model,
        dataset=dataset,
        epochs=tune_epochs,
        batch=batch,
        save=False,
        iterations=iterations,
        exist_ok=True,
        val=False,
        task = task,
        log=log)
        #    degrees=15.0,
        #    hsv_h=0.4,
        # cos_lr=True,
        # optimizer='AdamW')
    last_run = PROJECT_DIR / 'runs' / task / 'tune'

    # need to get best params from tune based on last run
    if train:
        # get best params from tune
        # last_run = PROJECT_DIR / 'runs/detect/tune'
        best_yaml = last_run / 'best_hyperparameters.yaml'
        with open(best_yaml, 'r') as f:
            best_params = yaml.safe_load(f)
        best_res = train_with_wb(id=short_id,
                                 model=model,
                                 dataset=dataset,
                                 epochs=train_epochs,
                                 batch=batch,
                                 save=True,
                                 exist_ok=True,
                                 task = task,
                                 log=log,
                                #  cos_lr=True,
                                 optimizer='AdamW',
                                 **best_params)
        pprint(best_res)

        if val:
            model.val()

    wandb.finish()


if __name__ == '__main__':
    # take arg of which model to use
    prsr = argparse.ArgumentParser()
    prsr.add_argument(
        'run', type=str, help='Name of run to tune---must be in runs/detect')
    prsr.add_argument('-m', '--model_type', type=str,
                      help='Type of model used, either yolo or detr; if None, inferred from run name')
    prsr.add_argument('-D', '--dataset', type=str,
                      help='Name of dataset, which contains a data.yaml file')
    prsr.add_argument('-p', '--project_dir', type=str,
                      default='./', help='Project directory')
    prsr.add_argument('-d', '--data_name', type=str,
                      default='data', help='Name of data directory')
    prsr.add_argument('-e', '--tune_epochs', type=int,
                      default=10, help='Number of epochs for tuning')
    prsr.add_argument('-E', '--train_epochs', type=int,
                      default=40, help='Number of epochs for training')
    prsr.add_argument('-b', '--batch', type=int, default=16, help='Batch size')
    prsr.add_argument('-i', '--iterations', type=int,
                      default=10, help='Number of iterations for tuning')
    prsr.add_argument('-t', '--train', action='store_true',
                      help='Whether to train after tuning')
    prsr.add_argument('-v', '--val', action='store_true',
                      help='Whether to validate after training')
    prsr.add_argument('-L', '--no_log', action='store_true',
                      help='Do not post logs to wandb online')
    prsr.add_argument('-l', '--label', type=str,
                      help='Custom label to override run name')
    prsr.add_argument('-k', '--task', type = str, default = 'detect', help = 'Task to perform, either detect or classify',
                      choices = ['detect', 'classify'])
    args = prsr.parse_args()
    pprint(args)

    main(**vars(args))

    # PROJECT_DIR = Path(args.project_dir)
    # setup(PROJECT_DIR)
    # data_dir = PROJECT_DIR / args.data_name
    # dataset = data_dir / args.dataset

    # # get model type based on run name
    # model_id = args.run
    # short_id = model_id.replace('_train', '')
    # weights = PROJECT_DIR / 'runs/detect' / model_id / 'weights' / 'best.pt'
    # model = get_model_type(model_id, args.model_type)(weights) # creates e.g. YOLO('train/weights/best.pt')

    # log = not args.no_log

    # last_run = tune_with_wb(
    #     id = short_id,
    #     model = model,
    #     dataset = dataset,
    #     epochs = args.tune_epochs,
    #     batch = args.batch,
    #     save = False,
    #     iterations = args.iterations,
    #     exist_ok = False,
    #     val = False,
    #     log = log,
    #     degrees = 15.0,
    #     hsv_h = 0.4,
    #     cos_lr = True,
    #     optimizer = 'AdamW')

    # # need to get best params from tune based on last run
    # if args.train:
    #     # get best params from tune
    #     # last_run = PROJECT_DIR / 'runs/detect/tune'
    #     best_yaml = last_run / 'best_hyperparameters.yaml'
    #     with open(best_yaml, 'r') as f:
    #         best_params = yaml.safe_load(f)
    #     best_res = train_with_wb(
    #         id = short_id,
    #         model = model,
    #         dataset = dataset,
    #         epochs = args.train_epochs,
    #         batch = args.batch,
    #         save = True,
    #         exist_ok = True,
    #         log = log,
    #         cos_lr = True,
    #         optimizer = 'AdamW',
    #         **best_params
    #     )
    #     pprint(best_res)

    #     if args.val:
    #         model.val()

    # wandb.finish()
