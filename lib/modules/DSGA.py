import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from modules.model_resnet import ResidualBlock
from torch.nn.modules.module import Module
import torch.nn.utils.spectral_norm as spectral_norm

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, output_dim):
        #600 600 1024
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, output_dim)
        
        #self.linear = nn.Linear(tgt_dim, tgt_dim, bias=True)
    def forward(self, x, adj):    
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x

#  batch, len, dim = output.size()
       
#         uh = nn.Linear(len, 10, bias=True)
#         uh = uh.view(batch, 1, len, dim)
#         output = nn.Tanh(uh)
#         print(output.size())
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True, CUDA=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        # print('in_features',in_features)
        # print('out_features',out_features)
        self.out_features = out_features
        
        if CUDA:
            self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            if CUDA:
                self.bias = Parameter(torch.cuda.FloatTensor(out_features))
            else:
                self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
  
        
    def forward(self, input, adj, CUDA=True):
        # support = torch.mm(input, self.weight)
        if CUDA:
            input = input.cuda()
            adj = adj.cuda()
        else:
            input=input
            adj=adj     
        support = torch.matmul(input, self.weight)
        support = support.float()
        adj = adj.float()
        # output = torch.spmm(adj, support)
        output = torch.matmul(adj,support)#[4*18*600]
       
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def add_pool(x, nd_to_sample):
    dtype, device = x.dtype, x.device
    batch_size = torch.max(nd_to_sample) + 1
    pooled_x = torch.zeros(batch_size, *x.shape[1:]).float().to(device)
    pool_to = nd_to_sample.view(-1, 1, 1, 1).expand_as(x).to(device)
    pooled_x = pooled_x.scatter_add(0, pool_to, x)
    return pooled_x

def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=False, batch_norm=True):
    block = []

    
    conv = torch.nn.Conv2d
    tconv = torch.nn.ConvTranspose2d

    if upsample:
        if spec_norm:
            block.append(spectral_norm(tconv(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True)))
        else:
            block.append(tconv(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True))
    else:
        if spec_norm:
            block.append(spectral_norm(conv(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True)))
        else:        
            block.append(conv(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True))
    if batch_norm:
        block.append(nn.InstanceNorm2d(out_channels))
        # block.append(nn.BatchNorm2d(out_channels))
    if "leaky" in act:
        block.append(torch.nn.LeakyReLU(0.1, inplace=True))
    elif "relu" in act:
        block.append(torch.nn.ReLU(inplace=True))
    elif "tanh":
        block.append(torch.nn.Tanh())
    return block

class CMP(nn.Module):
    def __init__(self, in_channels):
        super(CMP, self).__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            ResidualBlock(3*in_channels, 3*in_channels, 3, resample=None), 
            ResidualBlock(3*in_channels, 3*in_channels, 3, resample=None), 
            *conv_block(3*in_channels, in_channels, 3, 1, 1, act="relu"))

    def forward(self, feats, edges=None):
        
        # allocate memory
        dtype, device = feats.dtype, feats.device
        edges = edges.view(-1, 3)
        V, E = feats.size(0), edges.size(0)
        pooled_v_pos = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        pooled_v_neg = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        
        # pool positive edges
        pos_inds = torch.where(edges[:, 1] > 0)
        pos_v_src = torch.cat([edges[pos_inds[0], 0], edges[pos_inds[0], 2]]).long()
        pos_v_dst = torch.cat([edges[pos_inds[0], 2], edges[pos_inds[0], 0]]).long()
        pos_vecs_src = feats[pos_v_src.contiguous()]
        pos_v_dst = pos_v_dst.view(-1, 1, 1, 1).expand_as(pos_vecs_src).to(device)
        pooled_v_pos = pooled_v_pos.scatter_add(0, pos_v_dst, pos_vecs_src)
        
        # pool negative edges
        neg_inds = torch.where(edges[:, 1] < 0)
        neg_v_src = torch.cat([edges[neg_inds[0], 0], edges[neg_inds[0], 2]]).long()
        neg_v_dst = torch.cat([edges[neg_inds[0], 2], edges[neg_inds[0], 0]]).long()
        neg_vecs_src = feats[neg_v_src.contiguous()]
        neg_v_dst = neg_v_dst.view(-1, 1, 1, 1).expand_as(neg_vecs_src).to(device)
        pooled_v_neg = pooled_v_neg.scatter_add(0, neg_v_dst, neg_vecs_src)
        
        # update nodes features
        enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], 1)
        out = self.encoder(enc_in)
        return out
    

class DSGA(nn.Module):
    """
    input:
    feats edges


    output:
    y(t)
    attention
    """
    def __init__(self, hidden_dim=16,in_dim=64):
        super(DSGA, self).__init__()
        
    
        self.cmp_1 = CMP(in_channels=hidden_dim)
        self.cmp_2 = CMP(in_channels=hidden_dim)
        self.cmp_3 = CMP(in_channels=hidden_dim)
        self.cmp_4 = CMP(in_channels=hidden_dim)

        self.l1 = nn.Sequential(nn.Linear(18, 8 * in_dim ** self.dim))
        self.downsample_1 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')
        self.downsample_2 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')
        self.downsample_3 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')
        self.downsample_4 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')

        self.encoder = nn.Sequential(
            *conv_block(9, hidden_dim, 3, 1, 1, act="relu"),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample=None),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample=None))
        
        # define classification heads(rm last layer)
        self.head_local_cnn = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'))
    
    def forward(self, masks, nodes=None, edges=None, nd_to_sample=None):
        #cat gt
        S = masks.shape[-1]
        masks = masks.view(*([-1, 1] + [S]*self.dim))

        # include nodes
        if True:
            y = nodes
            y = self.l1(y)
            y = y.view(*([-1, 8] + [S]*self.dim))
            x = torch.cat([masks, y], 1)
        x = self.encoder(x).view(-1, *x.shape[1:])  
        x = self.downsample_1(x)
        x = self.cmp_2(x, edges).view(-1, *x.shape[1:])
        x = self.downsample_2(x)
        x = self.cmp_3(x, edges).view(-1, *x.shape[1:])
        x = self.downsample_4(x)
        x_l = self.cmp_4(x, edges).view(-1, *x.shape[1:])

        x_g = add_pool(x_l, nd_to_sample)
        x_g = self.head_global_cnn(x_g)
        
        return x_g



