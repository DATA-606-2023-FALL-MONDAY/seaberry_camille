'''
clean up both streetview & object365 datasets to put together metadata for upload to roboflow. 
because of how differently both sets of annotations are formatted and difficulties in changing formats, 
these get processed differently--Obj365 is close to COCO format and can get finalized with SAHI; 
Streetview is more difficult, and easiest thing is to get into VoTT csv format. 
Roboflow can process either.
'''

from glob import glob
from pycocotools.coco import COCO as PyCOCO
import sys
import tarfile
import sahi
from sahi.utils.coco import Coco as SahiCOCO
from sahi.utils.file import save_json
import pandas as pd
import json
import os
from pathlib import Path
import multiprocessing as mp


class Obj365Cleanup:
    '''
    Class for Object365 dataset cleanup.
    First, get into proper COCO format with `get_coco_imgs` & `extract_imgs` 
    to pull images from tar files without having to unopen them entirely. 
    Then use SAHI to extract just surveillance camera images. 
    Output: json file with COCO format annotations for surveillance imgs only.
    '''

    def __init__(self, split: str, base_dir: str):
        self.split = split
        self.base_dir = Path(base_dir)  # e.g. data/obj365
        self.ann_orig = self.base_dir / f'{split}-ann.json'
        self.ann_out = self.base_dir / f'{split}-coco.json'
        self.img_paths = []

    def get_imgs_from_ann(self):
        coco = PyCOCO(self.ann_orig)
        cat_ids = coco.getCatIds(catNms=['Surveillance Camera'])
        img_ids = coco.getImgIds(catIds=cat_ids)
        imgs = coco.loadImgs(img_ids)
        return imgs

    def get_img_patch(self, img_path: Path):
        patch = img_path.parts[-2]
        img = img_path.parts[-1]
        return f'{patch}/{img}'

    def extract_imgs(self, tar_path: str):
        v_set = tar_path.parts[-2] # v1 or v2
        v_path = self.base_dir / 'images' / v_set
        with tarfile.open(tar_path, 'r') as tar_obj:
            member_objs = tar_obj.getmembers()
            members = [m for m in member_objs if m.name in self.img_paths] # patch/img.jpg
            
            # path gives where to extract to, members gives which files to extract
            tar_obj.extractall(path = v_path, members = members)
            print(f'patch {tar_path}: {len(members)} files')
        return [m.name for m in members]
    
    def safe_extract_imgs(self, tar_path: str):
        try:
            members = self.extract_imgs(tar_path)
        except Exception as error:
            print(f'Error in {tar_path}', error)
    

    def write_coco(self):
        coco = SahiCOCO.from_coco_dict_or_path(
            str(self.ann_orig), clip_bboxes_to_img_dims=False, ignore_negative_samples=True
        )
        coco.update_categories({'Surveillance Camera': 2})  # let directed, globe be 0 & 1
        save_json(coco.json, self.ann_out)
        self.total = len(coco.json['images'])

    def prep_coco(self):
        imgs = self.get_imgs_from_ann()
        img_paths = [self.get_img_patch(self.base_dir / i['file_name']) for i in imgs] # patch/img.jpg
        self.img_paths = img_paths

        tar_files = list(self.base_dir.glob('images/*/*.tar.gz'))
        
        pool = mp.Pool(mp.cpu_count() - 1)
        pool.map(self.safe_extract_imgs, tar_files)

        self.write_coco()

class StreetCleanup():
    '''
    Class for Street View cleanup.
    Read metadata csv, drop rows where images are blank (small file size),
    then flatten into VoTT csv format.
    TODO: find a way to get this into coco json format instead. 
    Output: csv file with columns image, xmin, ymin, xmax, ymax, label
    '''
    def __init__(self, ann_orig: str, base_dir: str):
        self.base_dir = Path(base_dir) # data/streetview
        self.ann_orig = Path(ann_orig) # data/sv_meta.csv
        self.ann_out = self.base_dir / 'sv_clean.csv'
        
    def get_file_size(self, path: Path):
        return path.stat().st_size / 1024
    
    def fix_path(self, path: Path):
        # edit gsv_image_path--this is a holdover from Stanford project
        return self.base_dir / 'images' / path.name
    
    def read_json(self, x: str):
        # replace single quotes with double quotes in order to read json
        x = x.replace('\'', '"')
        return json.loads(x)
    
    def extract_col(self, row, col):
        # extract an item from a dict column in order to explode
        return [x[col] for x in row]
    
    def expand_col(self, col: pd.Series, new_names: list):
        # expand a column into multiple
        return pd.DataFrame(col.to_list(), columns = new_names)
            
    def clean_meta(self):
        ann = pd.read_csv(self.ann_orig)
        ann['path'] = ann['gsv_image_path'].apply(lambda x: self.fix_path(Path(x)))
        ann['image'] = ann['path'].apply(lambda x: x.name)
        ann['annotations'] = ann['annotations'].apply(self.read_json)
        
        # drop rows with no annotations
        ann = ann.loc[ann['annotations'].apply(lambda x: len(x)) > 0]
        # drop rows with small file size
        ann = ann.loc[ann['path'].apply(lambda x: self.get_file_size(x) >= 10)]
        
        # unnest bbox & label columns
        ann['bbox'] = ann['annotations'].apply(lambda row: self.extract_col(row, 'bbox'))
        ann['label'] = ann['annotations'].apply(lambda row: self.extract_col(row, 'category_id'))
        ann = ann.loc[:, ['image', 'bbox', 'label']].explode(['bbox', 'label'])
        
        # get bbox column (list) into 4 columns of coords
        bbox_df = self.expand_col(ann['bbox'], ['xmin', 'ymin', 'xmax', 'ymax'])
        ann_out = pd.concat([ann.reset_index(), bbox_df.reset_index()], axis = 1)
        ann_out = ann_out.loc[:, ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label']]
        ann_out.to_csv(self.ann_out, index = False)
        self.total = len(ann_out)

if __name__ == '__main__':
    obj365 = Obj365Cleanup(split='val', base_dir='data/obj365')
    obj365.prep_coco()
    
    sv = StreetCleanup(ann_orig = 'data/sv_meta.csv', base_dir = 'data/streetview')
    sv.clean_meta()
    
    print(f'total obj365 images: {obj365.total}')
    print(f'total streetview images: {sv.total}')
