import torch
# from model import DepthEstimation
from model import DepthEstimation2 as DepthEstimation
# from model import SimpleConv as DepthEstimation
import os
from torch.utils.data import DataLoader
from dataset import SEMdataset
from tqdm import tqdm
from PIL import Image
from data_utils import *


opj = os.path.join


def evaluate(args):
    dataset = SEMdataset(dir=args.data_dir, task='Test')
    test_loader = DataLoader(dataset=dataset,
                             batch_size=1,
                             num_workers=4
                             )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = opj(args.save_dir, "ckpt", str(args.epoch).zfill(4) +'.pt')
    save_test = opj(args.save_dir, "test", str(args.epoch).zfill(4))
    os.makedirs(save_test, exist_ok=True)
    
    model = DepthEstimation(n_conformer=2)
    # print('Current model:')
    # print(model)
    # print('Loading checkpoint from :', ckpt_dir)
    ckpt = torch.load(ckpt_dir, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    
    model = model.to(device)
    print('Dataset len:', dataset.__len__())
    
    for idx, data in enumerate(tqdm(test_loader)):
        img = data['image'].to(device)
        pred = model(img)
        pred_np = dcu_numpy(pred)[0][0]
        im = Image.fromarray(pred_np).convert('L')
        im.save(opj(save_test, data['name img'][0]+'.png'))
        
if __name__=="__main__":
    import argparse
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--data-dir', type=str, default='./AI_challenge_data')
    parser.add_argument('--epoch', type=int)
    
    args = parser.parse_args()
    
    evaluate(args)
    
    
        
    