import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

from models import register


@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

@register('split_mlp')
class SplitMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, split_list, fusion_layer=2):
        super().__init__()
        layers = []
        self.out_dim = out_dim
        self.split_list = split_list
        self.fusion_layer = fusion_layer
        self.in_layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(feat, hidden_list[0]), nn.ReLU())
                for feat in self.split_list]
        )
        # self.in_act = nn.ReLU()
        lastv = hidden_list[0]
        for hidden in hidden_list[1:]:
            layers.append(nn.Sequential(nn.Linear(lastv, hidden), nn.ReLU()))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        xs = torch.split(x, self.split_list, dim=-1)
        xo_list = []
        for i, xi in enumerate(xs):
            xo = self.in_layer[i](xi)
            xo_list.append(xo)
        xs = torch.stack(xo_list, -2)
        for i in range(len(self.layers)-self.fusion_layer):
            xs = self.layers[i](xs)
        # fusion
        x = self.fusion_op(xs)
        x = self.post_fusion(x)
        return x
    
    def forward_channel(self, x, channel_id):
        x = self.in_layer[channel_id](x)
        for i in range(len(self.layers)-self.fusion_layer):
            x = self.layers[i](x)
        return x

    def fusion_op(self, xs):
        if isinstance(xs, torch.Tensor):
            x = xs.prod(-2)
        else: # list
            x = xs[0]
            for xi in xs[1:]:
                x = x * xi

        if self.fusion_layer == 0:
            x = x.reshape(*x.shape[:-1], self.out_dim, -1).sum(-1)
        return x

    def post_fusion(self, x):
        for i in range(len(self.layers)-self.fusion_layer, len(self.layers)):
            x = self.layers[i](x)
        return x