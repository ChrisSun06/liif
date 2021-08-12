import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--split_accel', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = list(map(int, args.resolution.split(',')))
    if not args.split_accel:
        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
            coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    else:
        coord, coord_seqs = make_coord((h,w), with_split=True, flatten=False, device=device)
        cell = torch.tensor([2/h, 2/w]).cuda()
        pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
            coord.unsqueeze(0), cell.unsqueeze(0), coord_seqs=coord_seqs, bsize=30000)
    
    pred = (pred * 0.5 + 0.5).clamp(0, 1).reshape(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
