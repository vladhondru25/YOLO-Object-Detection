import os
import json
from collections import OrderedDict
import numpy as np
from PIL import Image
import skimage.io as io
import urllib3
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils

from pycocotools.coco import COCO

from utility.transformations import *


class CocoDatasetAPITrain(Dataset):
    def __init__(self):
        # self.images_train_dir = "data/images/train"
        self.images_train_dir = "data/images/val"
        self.annotation_dir = "data/annotations/train_val"
        
        self.coco = COCO(os.path.join(self.annotation_dir, "instances_val2017.json"))
        self.catIds = self.coco.getCatIds()
        
        self.imgIds = self.coco.getImgIds()
        self.dataset_length = len(self.imgIds)
        
    def __getitem__(self, idx):
        image_data = self.coco.loadImgs(self.imgIds[idx])[0]
        
        image = io.imread(image_data['coco_url'])
        
        if image.shape[-1] != 3:
            return None
        
        annIds = self.coco.getAnnIds(imgIds=image_data['id'])
        anns = self.coco.loadAnns(annIds)
        
        
        bboxes = np.zeros(shape=(len(anns),6))
        try:
            for i, ann in enumerate(anns):
                categ_id = map_category(ann['category_id'])
                bboxes[i,1:] = np.array([categ_id] + ann['bbox'])
        
            image, bboxes = ToTensor()(image, bboxes)
            
            # Scale bounding boxes according to resize image
            image_h, image_w = image.shape[1:]
            
            bboxes[:,2] = bboxes[:,2] * 640.0 / image_w
            bboxes[:,4] = bboxes[:,4] * 640.0 / image_w
            bboxes[:,3] = bboxes[:,3] * 416.0 / image_h
            bboxes[:,5] = bboxes[:,5] * 416.0 / image_h
            
            image = transforms.Resize(size=(416,640), interpolation=Image.NEAREST)(image)
        except:
            print("ERROR: CANNOT TRANSFORM DATA \n")
            return None
        
        return image, bboxes
    
    def collate_fn(self, batch):
        batch = [data for data in batch if data is not None]
        
        images, bboxes = list(zip(*batch))
        
        images = torch.stack(images)
        
        for i, boxes in enumerate(bboxes):
            boxes[:, 0] = i
        bboxes = torch.cat(bboxes, 0)
        
        return images, bboxes
        
    def __len__(self):
        return self.dataset_length
        
        
def map_category(cat):
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return cat
    
        
def load_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)