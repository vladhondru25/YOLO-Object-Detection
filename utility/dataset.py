import os
import json
from collections import OrderedDict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils

from utility.transformations import *


class CocoDatasetTrain(Dataset):
    def __init__(self):
        # self.images_train_dir = "data/images/train"
        self.images_train_dir = "data/images/val"
        self.annotation_dir = "data/annotations/train_val"
        
        self.length = len(os.listdir(self.images_train_dir))
        
        annotations_json = open(os.path.join(self.annotation_dir, "instances_val2017.json"))
        annotations =  json.load(annotations_json)
        
        self.dataset = OrderedDict()
        self.images_ids = []
        for image in annotations['images']:
            self.dataset[image['id']] = {'image': image['file_name'], 'boxes': []}
            self.images_ids.append(image['id'])
        
        for annotation in annotations['annotations']:
            self.dataset[annotation['image_id']]['boxes'].append( [annotation['category_id']] + annotation['bbox'] )  
        
    def __getitem__(self, idx):
        entry = self.dataset[self.images_ids[idx]]
        
        img_path = os.path.join(self.images_train_dir, entry['image'])
        image = Image.open(img_path).convert('RGB') #TODO: float or uint8?
        
        bboxes = np.zeros(shape=(len(entry['boxes']),6))
        try:
            bboxes[:,1:] = np.array(entry['boxes'])
        
            image, bboxes = ToTensor()(image, bboxes)
            image = transforms.Resize(size=(480,640), interpolation=Image.BILINEAR)(image)
        except:
            return None
        
        return image, bboxes
    
    def collate_fn(self, batch):
        batch = [data for data in batch if data is not None]
        
        images, bboxes = list(zip(*batch))
        
        images = torch.stack(images)
        
        for i, boxes in enumerate(bboxes):
            boxes[:, 0] = i
        bboxes = torch.cat(bboxes, 1).squeeze(0)
        
        return images, boxes
        
    def __len__(self):
        return self.length
        
        
def load_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)
        
        
if __name__ == "__main__":
    ds = CocoDatasetTrain()
    print(len(ds))
    
    entry = ds[2]
    
    # print(entry[1].shape)
    # plt.axis('off')
    # plt.imshow(entry[0])
    # plt.show()