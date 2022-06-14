import torch
import torch.nn
from torchvision import transforms
from dataset import SEMdataset
from PIL import Image
import numpy as np
from glob import glob
import cv2
from einops import rearrange



dataset = SEMdataset(task='Validation')
# print(dataset[0]['depth'].mean())

paths = glob('./AI_challenge_data/Train/Depth/*.png')
img = Image.open(paths[0])
img_totensor = transforms.ToTensor()(img)
print('tensor shape:', img_totensor.shape)

img = cv2.imread(paths[0])
img = np.load(paths[0])
print(img.dtype)

# img_astensor = torch.as_tensor(img)
# print('astensor : ', img_astensor.shape)
# x = img_astensor.tile((512,1, 1))
# print('here', x.shape)
# x = rearrange(x, 'b h w -> b 1 h w')  # if cv2.read!
# print(x.shape)
