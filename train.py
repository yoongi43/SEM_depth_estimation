from lib2to3.pgen2.tokenize import generate_tokens
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from data_utils import *

from dataset import SEMdataset

# from model import DepthEstimation
from model import DepthEstimation2 as DepthEstimation
# from model import SimpleConv as DepthEstimation

import wandb
from datetime import datetime
import os
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image



opj = os.path.join

class Solver:
    def __init__(self, args):
        # super().__init__()
        
        self.datasets = dict(
            train=SEMdataset(dir=args.dir, task='Train', transform=None),
            valid=SEMdataset(dir=args.dir, task='Validation', transform=None)
        )
        
        self.model = DepthEstimation(ic=1, n_conformer=args.n_conformer)
        self.mseloss = nn.MSELoss(reduction='sum')
        # self.loss_fn = nn.L1Loss()
        self.loss_fn = self.mseloss
        self.optim = optim.AdamW(params=self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=args.lr_step, gamma=args.lr_decay)
        self.img_size = 45*66
        
        
        
        if args.nowstr is None:
            """ New run """
            if args.debug:
                save_dir = 'debug'
            else: 
                args.nowstr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                save_dir = args.nowstr
            
            args.save_dir = opj(args.base_dir, save_dir)
            os.makedirs(args.save_dir, exist_ok=True)
            for d in ['train', 'valid', 'test', 'ckpt', 'wandb']:
                os.makedirs(opj(args.save_dir, d), exist_ok=True)
        else:
            """ resume """
            args.resume = True
            save_dirs = glob(opj(args.base_dir, args.nowstr + '*'))
            assert len(save_dirs) == 1
            args.save_dir = save_dirs[0]
            print("Resuming ---- savedir: ", args.save_dir)
            
            
            map_location = torch.device('cpu')
            ckpt = torch.load(
                opj(args.save_dir, "ckpt", str(args.epoch).zfill(4) + ".pt"),
                map_location=map_location)
            self.model.load_state_dict(ckpt['model'])
            # self.optim.load_state_dict(ckpt['optim'])
            
        self.args = args
        
        
    # def solve(self, model, data, device):
    #     img_input = data['image']
    #     pred = model(img_input)
    #     return pred
    
    def train(self):
        args = self.args
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print("Current Device: ", device)
        
        """ Wandb Setup """
        if not args.debug:
            os.environ["WANDB_START_METHOD"] = "thread"
            args.wandb_id = args.nowstr if args.nowstr is not None else None
            wandb_data = wandb.init(
                project = args.project,
                id=wandb.util.generate_id() if args.wandb_id is None else args.wandb_id,
                dir = opj(args.save_dir),
                resume=False if args.wandb_id is None else True,
                config=args,
            )
        """ Data """
        train_loader = DataLoader(dataset=self.datasets["train"], 
                                  batch_size=args.batch_size, 
                                  num_workers=args.num_workers,
                                  shuffle=True)
        valid_loader = DataLoader(dataset=self.datasets['valid'],
                                  batch_size=1,
                                  num_workers=args.num_workers,
                                  shuffle=True)
        print("Data Loaded")
        
        mseloss = self.mseloss.to(device)
        optim = self.optim
        scheduler = self.scheduler
        model = self.model.to(device)
        
        num_train = self.datasets['train'].__len__()
        num_valid = self.datasets['valid'].__len__()
        
        print('Num train: ', num_train)
        print('Num vlaid: ', num_valid)
        
        if not args.debug:
            wandb.watch(model, log='all')

        
        if args.resume:
            for state in optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        
        for epoch in range(args.epoch, 100000):
            print(f"\n#########START Training Loop: Epoch {epoch}#########\n")
            
            """ Training Loop """
            train_dir = opj(args.save_dir, 'train', str(epoch).zfill(4))
            os.makedirs(train_dir, exist_ok=True)
            model.train()
            
            total_mse_loss = 0
            
            for idx, data in enumerate(tqdm(train_loader)):
                img = data['image'].to(device)
                target = data['depth'].to(device)
                
                pred = model(img)
                
                optim.zero_grad()
                
                # loss = mseloss(pred, target) / img.size(0)
                # loss = torch.sqrt(loss+torch.finfo(torch.float32).eps)
                # loss = self.loss_fn(pred, target)
                
                L2 = (pred*255-target*255)**2
                L2 = L2.sum(axis=(1,2))/self.img_size
                rmse = torch.sqrt(L2)
                loss = torch.mean(rmse)
                
                loss.backward()
                optim.step()
                
                # total_mse_loss += mseloss(pred, target).detach()
                
                # total_loss += loss
                
                if not args.debug:
                    wandb.log({'train loss':loss})
                
                if args.debug:
                    if idx > 5:
                        break
                    
            scheduler.step()
            if not args.debug:
                wandb.log({'Learning rate': optim.param_groups[0]['lr']})
                wandb.log({'epoch': epoch})
            
            if epoch % args.print_image_per==0:
                with torch.no_grad():
                    img_np = dcu_numpy(img[0])[0]
                    depth_np = dcu_numpy(target[0])[0]
                    pred_np = dcu_numpy(pred[0])[0]
                    if not args.debug:
                        wandb.log({"train image":[wandb.Image(dcu_numpy(img[0]), caption='Image')],
                                "train depth":[wandb.Image(dcu_numpy(target[0]), caption="Depth")],
                                "train pred":[wandb.Image(dcu_numpy(pred[0]), caption="Pred depth")]})
                        concat = np.hstack([img_np, pred_np, depth_np])
                        im = Image.fromarray(concat).convert('L')
                        im.save(opj(train_dir, data['name img'][0]+'||'+data['name depth'][0]+'.jpeg'))

               
            # total_loss = total_loss / len(train_loader)
            # total_loss = dcu_numpy(total_loss)
            # total_rmse = dcu_numpy(torch.sqrt(total_loss))
            # total_rmse = torch.sqrt(total_mse_loss / num_train)
            # print(f"Epoch end: {epoch}, Loss: {total_rmse}")
            
            
            if epoch % args.valid_per == 0:
                print("Is it valid per?", epoch % args.valid_per)
                print('Valid loop start')
                valid_dir = opj(args.save_dir, 'valid', str(epoch).zfill(4))
                os.makedirs(valid_dir, exist_ok=True)

                with torch.no_grad():
                    model.eval()
                    total_loss = 0
                    for idx, data in enumerate(tqdm(valid_loader)):
                        img = data['image'].to(device)
                        target = data['depth'].to(device)
                        
                        pred = model(img)
                        
                        # loss = mseloss(pred, target)
                        L2 = (pred*255-target*255)**2
                        L2 = L2.sum(axis=(-2, -1))
                        rmse = torch.sqrt(L2 / self.img_size)
                        loss = torch.sum(rmse)
                        total_loss += loss
                    total_loss = total_loss / len(valid_loader)
                    # total_loss = torch.sqrt(total_loss)
                    total_loss = dcu_numpy(total_loss)
                        
                    print("Valid loss: ", total_loss)
                    if not args.debug:
                        img_np = dcu_numpy(img[0])[0]
                        depth_np = dcu_numpy(target[0])[0]
                        pred_np = dcu_numpy(pred[0])[0]
                        wandb.log({"valid image":[wandb.Image(dcu_numpy(img[0]), caption="Valid image")],
                                "valid depth":[wandb.Image(dcu_numpy(target[0]), caption='Valid depth')],
                                "valid pred":[wandb.Image(dcu_numpy(pred[0]), caption='Valid pred')]})

                        wandb.log({'valid loss': total_loss})
                        
                        concat = np.hstack([img_np, pred_np, depth_np])
                        im = Image.fromarray(concat).convert('L')
                        im.save(opj(valid_dir, data['name img'][0]+data['name depth'][0]+'.jpeg'))
                        
                    
            if epoch % args.ckpt_per == 0:
                ckpt = dict(
                    model=model.state_dict(),
                    optim=optim.state_dict()
                )
                torch.save(ckpt, opj(args.save_dir, 'ckpt', str(epoch).zfill(4)+'.pt'))
                    

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    """ Wandb"""
    parser.add_argument('--project', type=str, default="SEM depth estimation")
    parser.add_argument('--wandb-id', type=str, default=None)
    
    """ Resume """
    parser.add_argument('--nowstr', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.set_defaults(resume=False)
    parser.add_argument('--epoch', type=int, default=0)
    
    """ Data """
    parser.add_argument('--dir', type=str, default='./AI_challenge_data')
    parser.add_argument('--base_dir', type=str, default='./logs')
    
    """ Training """
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--train-report-batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr-step', type=int, default=50)
    parser.add_argument('--lr-decay', type=float, default=0.9)
    # parser.add_argument('--loss', type=str, default='rmse')
    
    parser.add_argument('--n-conformer', type=int, default=4)
    
    """ Validation """
    parser.add_argument('--print-image-per', type=int, default=30)
    parser.add_argument('--valid-per', type=int, default=10)
    parser.add_argument('--ckpt-per', type=int, default=10)
    
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    
                    
    args = parser.parse_args()
    
    solver = Solver(args)
    solver.train()
    