import argparse
import os
import math
from functools import partial
from typing import List

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils


def batched_predict(model, inp, coord, cell, bsize, coord_seqs=None): 
    # coord_seqs to trigger split_accel
    if coord_seqs is None:
        with torch.no_grad():
            model.gen_feat(inp)
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
    else:
        with torch.no_grad():
            model.gen_feat(inp, cell=cell, split_cache=True)
            rel_coord_seqs, coord_feat_seqs, feat_id_seqs = model.prep_coords_split(coord_seqs)
            
            # TODO tile-wise batching, not flattening coord. The batch is based on #feat
            try:
                x_bsize, y_bsize = bsize # bsize should be a tuple or list
            except:
                x_bsize, y_bsize = bsize, bsize
            x_len, y_len = coord.shape[1:3]
            preds = []
            for xi in range(0, x_len, x_bsize):
                preds_row = []
                for yi in range(0, y_len, y_bsize):
                    rel_coord_seqs_ = (rel_coord_seqs[0][xi:xi+x_bsize], rel_coord_seqs[1][yi:yi+y_bsize])
                    coord_feat_seqs_ = (coord_feat_seqs[0][xi:xi+x_bsize], coord_feat_seqs[1][yi:yi+y_bsize])
                    feat_id_seqs_ = (feat_id_seqs[0][xi:xi+x_bsize], feat_id_seqs[1][yi:yi+y_bsize])
                    pred = model.query_rgb_split(rel_coord_seqs_, coord_feat_seqs_, feat_id_seqs_)
                    preds_row.append(pred)
                preds.append(torch.cat(preds_row, 1)) #
            pred = torch.cat(preds, 0)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
            elif isinstance(v, list): # for seqs
                batch[k] = [vi.squeeze().cuda() for vi in v]
        if 'coord_seqs' not in batch: batch['coord_seqs'] = None
        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(model, inp,
                batch['coord'], batch['cell'], eval_bsize, batch['coord_seqs'])
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].numel()//2 / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--split_accel', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], 
                args={'dataset': dataset, 'split_accel': args.split_accel})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=0, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))
