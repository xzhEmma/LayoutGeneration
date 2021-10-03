from numpy.lib import index_tricks
import torch
import torch.nn as nn
import numpy as np
def _init_weights(module):
  if hasattr(module, 'weight'):
    if isinstance(module, nn.Linear):
      nn.init.kaiming_normal_(module.weight)

def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
  layers = []
  for i in range(len(dim_list) - 1):
    dim_in, dim_out = dim_list[i], dim_list[i + 1]
    layers.append(nn.Linear(dim_in, dim_out))
    final_layer = (i == len(dim_list) - 2)
    if not final_layer or final_nonlinearity:
      if batch_norm == 'batch':
        layers.append(nn.BatchNorm1d(dim_out))
      if activation == 'relu':
        layers.append(nn.ReLU())
      elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU())
    if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
  return nn.Sequential(*layers)

class GraphTripleConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none'):
    super(GraphTripleConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    #("input_dim",input_dim)
    
    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
    self.pooling = pooling
    net1_layers = [1500, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)
    
    net2_layers = [hidden_dim, hidden_dim, output_dim]
    self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    self.net2.apply(_init_weights)

    net3_layers = [input_dim, hidden_dim, hidden_dim]
    self.net3 = build_mlp(net3_layers, batch_norm=mlp_normalization)
    self.net3.apply(_init_weights)

  def forward(self,trip,rel_lens,index):
    O ,T= 18,  5
    Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim
    if len(trip.size())==2:
     trip = trip.unsqueeze(0)
    new_t_vecs = self.net1(trip)#[4,5,1024+600]
    # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
    # p vecs have shape (T, Dout)
    new_s_vecs = new_t_vecs[:,:, :H]
    new_p_vecs = new_t_vecs[:,:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:,:, (H+Dout):(2 * H + Dout)]
    #rongh
    pooled_obj_vecs = torch.zeros(O, H).cuda()
    if len(index.size())==2:
      index = index.unsqueeze(0)
    s_idx = index[:,:,0] #[4, 6]
    o_idx = index[:,:,1]
    if type(rel_lens)==int:
      rel_lens = [rel_lens]
    new_obj = []
    for i,l in enumerate(s_idx):
      l_r = l[:rel_lens[i]]
      ls = l_r.cpu().numpy()
      for j, ind in enumerate(ls):  
        p = np.where(ls == ind)
        if len(p)>1:
          for m, v in enumerate(p):
            update = 1/2*torch.cat(new_s_vecs[v],new_o_vecs[o_idx[i,j]])
            new_s_vecs[v] = update
            new_o_vecs[o_idx[i,j]] = update 
 
      s_exp = l_r.view(-1,1).expand_as(new_s_vecs[i,:rel_lens[i],:]).long().cuda()
      o_exp = o_idx[i,:rel_lens[i]].view(-1,1).expand_as(new_o_vecs[i,:rel_lens[i],:]).long().cuda()
      #print('o_exp.size()',o_exp.size(),s_exp.size())
      #[len]-->[len,512]   
      pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_exp.long(), new_s_vecs[i,:rel_lens[i],:])
      pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_exp, new_o_vecs[i,:rel_lens[i],:])#6,512
      
      # for m,ob in enumerate(pooled_obj):
      #   if ob == 0:
      #     pooled_obj_vecs[m,:]== self.net3(obj_vecs[i,m,:])

      new_obj.append(self.net2(pooled_obj_vecs).unsqueeze(0))
    new_obj_vecs = torch.cat(new_obj,0)    
    return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
  """ A sequence of scene graph convolution layers  """

  def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
               mlp_normalization='none'):
    super(GraphTripleConvNet, self).__init__()

    self.num_layers = num_layers
    self.gconvs = nn.ModuleList()
    gconv_kwargs = {
      'input_dim': input_dim,
      'hidden_dim': hidden_dim,
      'pooling': pooling,
      'mlp_normalization': mlp_normalization,
    }
    for _ in range(self.num_layers):
      self.gconvs.append(GraphTripleConv(**gconv_kwargs))

  def forward(self, trip,rel_lens,index):
    for i in range(self.num_layers):
      gconv = self.gconvs[i]
      obj_vecs, pred_vecs = gconv(trip,rel_lens,index)
    return obj_vecs, pred_vecs
