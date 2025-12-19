import torch

from PIL import Image
import torchvision.transforms as transforms

from torch.nn import functional as F

import os

from tqdm.auto import trange, tqdm
from einops import rearrange, repeat

import numpy as np
import pandas as pd

from random import randint

from torchvision import transforms

from einops import rearrange
import numpy as np
import random
from einops import rearrange, repeat
import json
import pdb
from openai import OpenAI
import httpx
import torch
import requests
import base64
import re

from ..assets.gme_inference import GmeQwen2VL

def cosine_similarity(a, b):
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

class ZeroInsDataset(torch.utils.data.Dataset):

    def __init__(self, text_path, image_folder, clip_path, dino_path):
        self.text_path = text_path
        self.all_images = []
        for subdir, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(subdir, file)
                    self.all_images.append(image_path)
        self.all_goals = [path.split("_done.txt")[0] for path in os.listdir("./step_data/GT_description")]
        
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        self.tokenizer = self.clip_processor.tokenizer
        self.clip_image_processor = self.clip_processor.image_processor
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_path)

        
    def __getitem__(self, index):
        cur_image = self.all_images[index]

        cur_goal = os.path.dirname(cur_image).split("/")[-1].replace("_", " ")
        #answer_path = os.path.join(self.text_path, 'sub_step_folder')
        answer_path = self.text_path
        answer_file = os.path.join(answer_path,os.path.dirname(cur_image).split("/")[-1]+'_done.txt') #+'.txt')#
        
        with open(answer_file, 'r') as file:
            answer = file.read()
        answer = answer.split("*")[:-1]
        
        try:
            cur_answer_index = int(cur_image.split("_")[-1].split(".")[0])-1
            cur_step = answer[cur_answer_index].split("Action:")[0].split("Image Description:")[1]
            #cur_answer_index = int(cur_image.split("_")[-1].split(".")[0])-1
            #cur_step = answer[cur_answer_index].split("Action:")[1]
        except Exception as e:
            print(e)
            print("Error in ",cur_image)#, "cur_index: ", cur_answer_index)
            #print("answer: ",answer[cur_answer_index])
            print("answer length:", len(answer))
            #cur_step = answer[cur_answer_index-1]

        
        previous_step = int(cur_image.split("_")[-1].split(".")[0])-1
        previous_image = os.path.join(os.path.dirname(cur_image),"Step_"+str(previous_step)+".png")
        has_previous = True
        if not os.path.exists(previous_image):
            previous_image=self.dino_processor(images=Image.open(cur_image), return_tensors="pt")['pixel_values']
            has_previous=False
        else:
            previous_image=self.dino_processor(images=Image.open(previous_image), return_tensors="pt")['pixel_values']

        cur_image_clip = self.clip_processor(images=Image.open(cur_image), return_tensors="pt")['pixel_values']
        cur_image_kosmos = Image.open(cur_image) #self.kosmos_processor(images=Image.open(cur_image), return_tensors="pt")['pixel_values']
        #pdb.set_trace()
        cur_image_dino = self.dino_processor(images=Image.open(cur_image), return_tensors="pt")['pixel_values']
        
        out = {'cur_image_dino': cur_image_dino, 
               'cur_image_clip': cur_image_clip, 
               'cur_image_kosmos': cur_image_kosmos, 
               'cur_step':cur_step,
               'cur_goal':cur_goal,
               'has_previous':has_previous,
               'previous_image': previous_image,
               'cur_image_path': cur_image,
               "previou_step": previous_step}
        
        return out
    

    def __len__(self):
        return len(self.all_images)


def custom_collate_fn(batch):
    #pdb.set_trace()
    cur_image_dino = torch.cat([item['cur_image_dino'] for item in batch],dim=0)
    cur_image_clip = torch.cat([item['cur_image_clip'] for item in batch],dim=0)
    cur_image_kosmos = [item['cur_image_kosmos'] for item in batch] #torch.cat([item['cur_image_kosmos'] for item in batch],dim=0)
    cur_step = [item['cur_step'] for item in batch]  # list format
    cur_goal = [item['cur_goal'] for item in batch]  
    has_previous = [item['has_previous'] for item in batch] 
    previous_image = torch.cat([item['previous_image'] for item in batch],dim=0)
    cur_image_path = [item['cur_image_path'] for item in batch]  

    return {'cur_image_dino': cur_image_dino, 
            'cur_image_clip': cur_image_clip, 
            'cur_image_kosmos': cur_image_kosmos, 
            'has_previous':has_previous,
            'cur_step':cur_step,
            'cur_goal':cur_goal,
            'previous_image': previous_image,
            'cur_image_path': cur_image_path}


if __name__ == '__main__':
    # Shuffle the tensor
    torch.manual_seed(42)  # Set a seed for reproducibility
    random.seed(42)

    dataset = ZeroInsDataset(text_path="./step_data/GT_description",
                    image_folder="./step_data/results",
                    clip_path="./clip-vit-large-patch14",
                    dino_path="./dinov2-large")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=128,
        collate_fn=custom_collate_fn,
        num_workers=12
    )

    device="cuda:0"

    qwen_model = GmeQwen2VL('Alibaba-NLP/gme-Qwen2-VL-2B-Instruct')

    torch.set_grad_enabled(False)


    # batched image based rating
    #dino_score = []
    overall_goal_score = []
    all_clip_step_score = []
    all_clip_goal_score = []
    all_step_bert_P = []
    all_step_bert_R = []
    all_step_bert_F1 = []
   
    for batch_num, batch in enumerate(tqdm(dataloader)): 
        torch.cuda.empty_cache()
        cur_image_dino = batch['cur_image_dino'].cuda()  
        cur_image_clip = batch['cur_image_clip'].cuda()  
        cur_image_kosmos = batch['cur_image_kosmos']#.cuda()  
        has_previous = batch['has_previous']
        previous_image = batch['previous_image'].cuda()  
        previous_step = batch['previous_step']
        cur_step = batch['cur_step']
        cur_goal = batch['cur_goal']
        cur_image_path = batch['cur_image_path']
        

        if has_previous: 
            cur_step_image = cur_image_path
            previous_step_text = previous_step
            previous_step_image = previous_image

            prev_embedding = qwen_model.embed(["describe this image after applying : " + previous_step_text], [previous_step_image])
            cur_embedding = qwen_model.embed(["describe this image"], [cur_step_image])
            cur_step_similarity = cosine_similarity(prev_embedding, cur_embedding)

            all_clip_step_score.append(cur_step_similarity)


    print("Qwen2 step score:",torch.mean(torch.cat(all_clip_step_score).float()),"\n",)