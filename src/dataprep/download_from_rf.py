'''
Download datasets from Roboflow. 
All were uploaded and augmented online. 
To get cropped images for classification, I create a cropped version in the UI,
download it with this script as a temporary folder, then manually upload it as a new project.
This is because projects are setup as either detection or classification.
'''

from roboflow import Roboflow
from dotenv import load_dotenv
from pathlib import Path
import os
import shutil

def download_rf_imgs(proj, v: int, data_dir: str, format: str = 'coco', overwrite: bool = False):
    dataset = proj.version(v).download(model_format = format, location = data_dir, overwrite = overwrite)
    return dataset

if __name__ == '__main__':
    load_dotenv()
    rf = Roboflow(api_key = os.getenv('ROBOFLOW_KEY'))
    proj_detect = rf.workspace('seaberry').project('cap-detect')
    proj_class  = rf.workspace('seaberry').project('cap-class')
    
    # going to work with data in yolo format but already have code based on coco
    download_rf_imgs(proj_detect, v = 2, data_dir = 'data/cams_full_coco', format = 'coco', 
                     overwrite = True)
    download_rf_imgs(proj_detect, v = 2, data_dir = 'data/cams_full', format = 'yolov5',
                     overwrite = True)
    # temporary folder of cropped images
    # download_rf_imgs(proj_detect, v = 3, data_dir = 'data/cams_crop_tmp')
    # cropped images for classification
    download_rf_imgs(proj_class, v = 1, data_dir = 'data/cams_crop', format = 'folder', 
                     overwrite = True)
    
    # shutil.rmtree('data/cams_crop_tmp')