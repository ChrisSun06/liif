import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord, one_d_sample


@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None: # implicit mlp
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            split_list = [imnet_in_dim, 1, 1]
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
                split_list.append(2)
            args_update = {'in_dim': imnet_in_dim}
            if imnet_spec['name'] == 'split_mlp': 
                args_update['split_list'] = split_list
            self.channel_dict = {'feat': 0, 'x': 1, 'y': 2, 'cell': 3}
            self.imnet = models.make(imnet_spec, args=args_update)
        else:
            self.imnet = None

    def gen_feat(self, inp, split_cache=False, cell=None):
        self.feat = self.encoder(inp)
        if self.feat_unfold:
            self.feat = F.unfold(self.feat, 3, padding=1).view(
                self.feat.shape[0], self.feat.shape[1] * 9, self.feat.shape[2], self.feat.shape[3])
        if split_cache:
            self.feat = self.imnet.forward_channel(
                    self.feat.permute(0,2,3,1), self.channel_dict['feat'])
            if cell is not None and self.cell_decode:
                cell_feat = self.imnet.forward_channel(cell, self.channel_dict['cell'])
                self.feat = self.imnet.fusion_op([self.feat, cell_feat])
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # # TODO redundant
        # if self.feat_unfold:
        #     feat = F.unfold(feat, 3, padding=1).view(
        #         feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # coord_ = coord.clone()
                # coord_[:, :, 0] += vx * rx + eps_shift
                # coord_[:, :, 1] += vy * ry + eps_shift
                coord_ = coord + torch.tensor([vx * rx, vy * ry]).cuda() + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
    
    # only consider coord_seqs.
    # 2000-d coord vector will generate a feature tensor of 1.95 MB --> totally acceptable
    def prep_coords_split(self, coord_seqs): #, ret_feat=False, ret_id=False): 
        feat = self.feat
        feat_coord_seqs = make_coord(feat.shape[1:-1], flatten=False, only_split=True, device=feat.device)
        feat_id_seqs = [torch.arange(i, device=feat.device).float() for i in feat.shape[1:-1]]
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0
            
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        q_id_seqs = []
        rel_coord_seqs = [] # nested list
        coord_feat_seqs = []

        for vx in vx_lst:
            for vy in vy_lst:
                coord_seqs_ = [
                    (coord_seqs[0] + vx*rx + eps_shift).clamp(-1 + 1e-6, 1 - 1e-6), 
                    (coord_seqs[1] + vy*ry + eps_shift).clamp(-1 + 1e-6, 1 - 1e-6)
                ]
                q_id = one_d_sample(feat_id_seqs, coord_seqs_)
                q_id = [q.long() for q in q_id]
                q_coord_seqs = [feat_coord_seqs[i][idx] for i,idx in enumerate(q_id)]
                rel_coords = [coord_seqs_[i] - q_coord_seqs[i] for i in range(2)]
                rel_coord_seqs.append(rel_coords)
                # if ret_feat:
                coord_feat_seqs.append([
                    self.imnet.forward_channel(rel_coords[0][:,None], self.channel_dict['x']),
                    self.imnet.forward_channel(rel_coords[1][:,None], self.channel_dict['y'])
                ])
                # if ret_id:
                q_id_seqs.append(q_id)
        
        return rel_coord_seqs, coord_feat_seqs, q_id_seqs
        # if ret_id:
        #     return rel_coord_seqs, q_id_seqs
        # else:
        #     return rel_coord_seqs

        # return rel_coord_seqs

    # TODO query_rgb for split acceleration
    def query_rgb_split(self, rel_coord_seqs, coord_feat_seqs, feat_id_seqs):
        feat = self.feat

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        preds = []
        areas = []
        cnt_id = 0
        for vx in vx_lst:
            for vy in vy_lst:
                feat_id = feat_id_seqs[cnt_id]
                x_feat, y_feat = coord_feat_seqs[cnt_id]
                x_rel, y_rel = rel_coord_seqs[cnt_id]
                idx = torch.meshgrid(*feat_id)
                feat_ =feat[0, idx[0], idx[1], :]
                fusion_feat = self.imnet.fusion_op([feat_, x_feat[:, None, :], y_feat[None,:]])
                pred = self.imnet.post_fusion(fusion_feat)
                preds.append(pred)

                area = torch.abs(x_rel[:,None] * y_rel[None,:])
                areas.append(area)
        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret