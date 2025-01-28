"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw, ImageFont

from src.core import YAMLConfig
import os

labels_text=["Person","Handgun"]

fnt = ImageFont.truetype("arial.ttf", 20)

def draw(images, labels, boxes, scores, name, output, thrh):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{labels_text[lab[j].item()]} - {round(scrs[j].item(), 3)}", fill='blue', font=fnt)

        os.makedirs(output, exist_ok=True)
        name_ = os.path.basename(name)
        im.save(f'{output}/{name_}')

def inference(args, ):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    if args.folder:
        images = os.listdir(args.folder)
        print(len(images))
    
    if args.im_file:
        images = [args.im_file]

    for i in images:
        if args.folder:
            _, file_extension = os.path.splitext(args.folder + '/' + i)
            
            if file_extension != '.jpg':
                continue
        
            im_pil = Image.open(args.folder + '/' + i).convert('RGB')

        im_pil = Image.open(i).convert('RGB')

        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

        im_data = transforms(im_pil)[None].to(args.device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        draw([im_pil], labels, boxes, scores, i, args.output, args.thrh)

def main(args, ):
    """main
    """
    if args.folder:
        images = os.listdir(args.folder)
        print(len(images))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-i', '--folder', type=str)
    parser.add_argument('-o', '--output', type=str, default='./output/output_images')
    parser.add_argument('-t', '--thrh', type=float, default=0.50)
    args = parser.parse_args()
    inference(args)
