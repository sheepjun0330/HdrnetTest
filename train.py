import os
import sys
from test import test
import gc
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torch.optim import SGD, Adam, RAdam, RMSprop
from torch.utils.data import DataLoader
import adabound
import criteria
import wandb
from dataset import HDRDataset
from metrics import psnr
from model import HDRPointwiseNN, L2LOSS
from utils import load_image, save_params, get_latest_ckpt, load_params

torch.manual_seed(13)
random.seed(13)

torch.cuda.empty_cache()
gc.collect()

def train(params=None):
    os.makedirs(params['ckpt_path'], exist_ok=True)
    wandb.init(project="HdrnetV3")
    config = {
      "learning_rate": params['lr'],
      "lam1": params['lam1'],
      "lam2": params['lam2'],
      "lam3": params['lam3'],
      "lam4": params['lam4'],
      "epochs": params['epochs'],
      "batch_size": params['batch_size']
    }
    wandb.config.update(config)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    train_dataset = HDRDataset(params['dataset'], params=params, suffix=params['dataset_suffix'])
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_dataset = HDRDataset(params['validset'], params=params, suffix=params['validset_suffix'])
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'])
    alpha = params['lambda']
    model = HDRPointwiseNN(params=params)
    ckpt = get_latest_ckpt(params['ckpt_path'])
    if ckpt:
        print('Loading previous state:', ckpt)
        state_dict = torch.load(ckpt)
        state_dict,_ = load_params(state_dict)
        model.load_state_dict(state_dict)
    model.to(device)

    mseloss = torch.nn.MSELoss()#L2LOSS()#torch.nn.MSELoss()#torch.nn.SmoothL1Loss()#
    cliploss = criteria.CLIPLoss(device)
    optimizer = Adam(model.parameters(), params['lr'], eps=1e-7)#, weight_decay=1e-5)
    # optimizer = SGD(model.parameters(), params['lr'], momentum=0.9)
    # optimizer = adabound.AdaBound(model.parameters(), lr=params['lr'], final_lr=0.1)
    
    train_losses = {"mse_loss":[], "clip_loss":[], "patch_loss":[], "total_loss":[]}
    valid_losses = {"mse_loss":[], "clip_loss":[], "patch_loss":[], "total_loss":[]}
    count = 0
    count2 = 0
    lam1 = params['lam1']
    lam2 = params['lam2']
    lam3 = params['lam3']
    lam4 = params['lam4']
    batch_size = params['batch_size']
    for e in range(params['epochs']):
        model.train()
        for i, (low, full) in enumerate(train_loader):
            optimizer.zero_grad()

            low = low.to(device)
            full = full.to(device)
            res = model(low, full)
            text_direction = cliploss.compute_text_direction("Normal photo", "Warm photo")
            mse_loss = mseloss(full,res)
            clip_loss = cliploss.clip_directional_loss(full, res, text_direction)
            patch_loss = cliploss.patch_loss(full, res, text_direction, batch_size)
            tv_loss = cliploss.tv_loss(res)

            total_loss = lam1*mse_loss + lam2*clip_loss + lam3*patch_loss + lam4*tv_loss
            total_loss.backward()
            train_losses["mse_loss"].append(mse_loss.item())
            train_losses["clip_loss"].append(clip_loss.item())
            train_losses["patch_loss"].append(patch_loss.item())
            train_losses["total_loss"].append(total_loss.item())
            if (count+1) % params['log_interval'] == 0:
                loss = total_loss.item()
                print(e, count, loss)
                train_mse_loss = np.mean(train_losses["mse_loss"])
                train_clip_loss = np.mean(train_losses["clip_loss"])
                train_patch_loss = np.mean(train_losses["patch_loss"])
                train_total_loss = np.mean(train_losses["total_loss"])
                wandb.log({"Training MSE Loss": train_mse_loss, "Training CLIP Loss": train_clip_loss, "Training Patch Loss":train_patch_loss, "Training Total Loss": train_total_loss})
                del train_losses
                torch.cuda.empty_cache()
                train_losses = {"mse_loss":[], "clip_loss":[], "patch_loss":[], "total_loss":[]}
            
            if (count+1) % params['ckpt_interval'] == 0:
                print('@@ MIN:',torch.min(res),'MAX:',torch.max(res))
                model.eval().cpu()
                ckpt_model_filename = "ckpt_"+str(e)+'_' + str(count) + ".pth"
                ckpt_model_path = os.path.join(params['ckpt_path'], ckpt_model_filename)
                state = save_params(model.state_dict(), params)
                torch.save(state, ckpt_model_path)
                #test(ckpt_model_path)
                model.to(device).train()
            count += 1
        model.eval()
        with torch.no_grad():
            for i, (low, full) in enumerate(valid_loader):
                low = low.to(device)
                full = full.to(device)
                res = model(low, full)
                text_direction = cliploss.compute_text_direction("Normal photo", "Warm photo")
                mse_loss = mseloss(full,res)
                clip_loss = cliploss.clip_directional_loss(full, res, text_direction)
                patch_loss = cliploss.patch_loss(full,res,text_direction,batch_size)
                tv_loss = cliploss.tv_loss(res)
                total_loss = lam1*mse_loss + lam2*clip_loss + lam3*patch_loss +lam4*tv_loss
                valid_losses["mse_loss"].append(mse_loss.item())
                valid_losses["clip_loss"].append(clip_loss.item())
                valid_losses["patch_loss"].append(patch_loss.item())
                valid_losses["total_loss"].append(total_loss.item())
                if (count2+1) % params['log_interval'] == 0:
                    valid_mse_loss = np.mean(valid_losses["mse_loss"])
                    valid_clip_loss = np.mean(valid_losses["clip_loss"])
                    valid_patch_loss = np.mean(valid_losses["patch_loss"])
                    valid_total_loss = np.mean(valid_losses["total_loss"])
                    wandb.log({"Validation MSE Loss": valid_mse_loss, "Validation CLIP Loss": valid_clip_loss, "Validation Patch Loss": valid_patch_loss, "Validation Total Loss": valid_total_loss})
                    del valid_losses
                    torch.cuda.empty_cache()
                    valid_losses = {"mse_loss":[], "clip_loss":[], "patch_loss":[], "total_loss":[]}
                count2 += 1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HDRNet Inference')
    parser.add_argument('--ckpt-path', type=str, default='./ch', help='Model checkpoint path')
    parser.add_argument('--test-image', type=str, dest="test_image", help='Test image path')
    parser.add_argument('--test-out', type=str, default='out.png', dest="test_out", help='Output test image path')

    parser.add_argument('--luma-bins', type=int, default=8)
    parser.add_argument('--channel-multiplier', default=1, type=int)
    parser.add_argument('--spatial-bin', type=int, default=16)
    parser.add_argument('--guide-complexity', type=int, default=16)
    parser.add_argument('--batch-norm', action='store_true', help='If set use batch norm')
    parser.add_argument('--net-input-size', type=int, default=256, help='Size of low-res input')
    parser.add_argument('--net-output-size', type=int, default=512, help='Size of full-res input/output')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--ckpt-interval', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='', help='Dataset path with input/output dirs', required=True)
    parser.add_argument('--dataset-suffix', type=str, default='', help='Add suffix to input/output dirs. Useful when train on different dataset image sizes')
    parser.add_argument('--validset', type=str, default='', help='Valid Dataset path with input/output dirs', required=True)
    parser.add_argument('--validset-suffix', type=str, default='', help='Add suffix to input/output dirs. Useful when train on different dataset image sizes')
    parser.add_argument('--lambda', type=float, default=0.5, help='parameter between mse & clip')
    
    parser.add_argument('--lam1', type=float, default=1, help='mse loss')
    parser.add_argument('--lam2', type=float, default=1, help='clip loss')
    parser.add_argument('--lam3', type=float, default=0, help='patch loss')
    parser.add_argument('--lam4', type=float, default=0, help='tv loss')

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)

    train(params=params)
