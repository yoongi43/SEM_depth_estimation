from fcntl import DN_DELETE
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
import cv2


opj = os.path.join


class SEMdataset(Dataset):
    def __init__(self, dir='./AI_challenge_data/', task='Train', transform=None):
        print("SEM Dataset Loading from :", dir)
        
        if task not in ['Train', 'Test', 'Validation']:
            raise NotImplementedError("Task should be 'Train' or 'Test' or 'Validation'")
        self.dir = dir
        self.task = task
        self.img_path = opj(dir, task, 'SEM')
        self.depth_path = opj(dir, task, 'Depth')
        self.transform = transform

        self.imgs = sorted(glob(opj(self.img_path, '*.png')))
        self.depths = sorted(glob(opj(self.depth_path, '*.png')))
        
        num_imgs = len(self.imgs)
        num_depths = len(self.depths)
        self.img_names = [os.path.basename(os.path.splitext(self.imgs[idx])[0]) for idx in range(num_imgs)]
        self.depth_names = [os.path.basename(os.path.splitext(self.depths[idx])[0]) for idx in range(num_depths)]
        
        
        "Image-Depth pair test"
        if self.task != 'Test':
            if num_imgs//4 != num_depths:
                print('image - depth 4:1 pair did not match')
            for idx in range(num_imgs):
                if self.img_names[idx][:-5] != self.depth_names[idx//4]:
                    print("Image - depth unpaired!!!!!!")
            print('Parining Test Complete')
                
                
    def __len__(self):
        return len(self.imgs)
        
        
    def __getitem__(self, idx):
        sample = {}
        
        img = Image.open(self.imgs[idx]).convert('L')
        sample['image'] = transforms.ToTensor()(img)
        
        # img = cv2.imread(self.imgs[idx], flags=cv2.IMREAD_GRAYSCALE)
        # sample['image'] = torch.as_tensor(img, dtype=torch.float)
        
        img_name = self.img_names[idx]
        sample['name img'] = img_name
        if self.task != 'Test':
            depth_name = self.depth_names[idx//4]
            sample['name depth'] = depth_name
            
            depth = Image.open(self.depths[idx//4])
            sample['depth']= transforms.ToTensor()(depth)
            
            # depth = cv2.imread(self.depths[idx//4], flags=cv2.IMREAD_GRAYSCALE)
            # sample['depth'] = torch.as_tensor(depth, dtype=torch.float)
        
        if self.transform:
            sample = self.transform(sample)            
        
        return sample
    
if __name__=="__main__":
    from torchvision import transforms
    import numpy as np

    ds = SEMdataset(task='Train')
    print(len(ds))
    print(ds[0]['image'].shape)