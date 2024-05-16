import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

import math
import clip
from PIL import Image

import time

class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)

class CLIPLoss(torch.nn.Module):
    def __init__(self, device, direction_loss_type='cosine', clip_model='RN50'): # 'ViT-B/32', 'RN50'
        super(CLIPLoss, self).__init__()

        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose(clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                             clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        self.cos = torch.nn.CosineSimilarity()
        
        self.direction_loss = DirectionLoss(direction_loss_type)
        self.model.requires_grad_(False)

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)
    
    def get_text_features(self, class_str: str, norm: bool = True) -> torch.Tensor:
        template_text = [class_str]

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction
            
            
    def clip_directional_loss(self, src_img: torch.Tensor, target_img: torch.Tensor, target_direction: torch.Tensor) -> torch.Tensor:
        src_encoding    = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        if edit_direction.sum() == 0:
            target_encoding = self.get_image_features(target_img + 1e-6)
            edit_direction = (target_encoding - src_encoding)

        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True))
        
        return self.direction_loss(edit_direction, target_direction).mean()
        
    def forward(self, src_img: torch.Tensor, target_img: torch.Tensor, target_direction: torch.Tensor):
        clip_loss = 0.0
        clip_loss += self.clip_directional_loss(src_img, target_img, target_direction)
        return clip_loss
    
    def patch_loss(self, src_img: torch.Tensor, target_img: torch.Tensor, target_direction: torch.Tensor, thresh=0.7, num_crops=64, crop_size = 64):
        def clip_normalize(image, device):
            image = F.interpolate(image,size=224,mode='bicubic')
            mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
            std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
            mean = mean.view(1,-1,1,1)
            std = std.view(1,-1,1,1)

            image = (image-mean)/std
            return image
        cropper = transforms.Compose([
            transforms.RandomCrop(crop_size)
        ])
        augment = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
            transforms.Resize(224)
        ])
        '''
        with torch.no_grad():
            source_features = self.model.encode_image(clip_normalize(src_img,self.device))
            source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
        '''
        patch_loss=0 
        img_proc =[]
        for n in range(num_crops):
            target_crop = cropper(target_img)
            target_crop = augment(target_crop)
            img_proc.append(target_crop)

        img_proc = torch.cat(img_proc,dim=0)
        img_aug = img_proc

        #image_features = self.model.encode_image(clip_normalize(img_aug, self.device))
        #image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
        
        image_features = self.get_image_features(clip_normalize(img_aug,self.device))
        source_features = self.get_image_features(clip_normalize(src_img,self.device))
        print("size")
        print(source_features.size())
        print(image_features.size())       

        img_direction = (image_features-source_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        loss_temp = (1- torch.cosine_similarity(img_direction, target_direction, dim=1))
        loss_temp[loss_temp<thresh] =0
        patch_loss+=loss_temp.mean()

        return patch_loss.item()
