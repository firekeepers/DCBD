import numpy as np
import torch
from torch.nn import functional as F
# x = []
# z = torch.tensor([0.64])
# y = torch.tensor([0.64])
# x.append(z)
# x.append(y)
# x =torch.tensor(x)
# print(x)


x = torch.tensor([[[[ 0.0372,  0.0144],
          [ 0.0342,  0.0544],
          [-0.0048,  0.0177]]],


        [[[ 0.0285,  0.0368],
          [ 0.0561,  0.0143],
          [ 0.0342, -0.0027]]],


        [[[ 0.0261,  0.0378],
          [ 0.0471,  0.0874],
          [-0.0108,  0.0245]]],


        [[[-0.0002,  0.0303],
          [ 0.0435,  0.0355],
          [ 0.0190, -0.0297]]],


        [[[ 0.0481,  0.0350],
          [ 0.0300,  0.0499],
          [-0.0031,  0.0353]]],


        [[[ 0.0140,  0.0225],
          [ 0.0493,  0.0391],
          [ 0.0265, -0.0002]]],


        [[[ 0.0314,  0.0304],
          [ 0.0218,  0.0445],
          [ 0.0244,  0.0080]]],


        [[[ 0.0180,  0.0165],
          [ 0.0484,  0.0190],
          [ 0.0234,  100]]]])
y = torch.mean(x)

x = torch.randn((4,2,40,86))
y = torch.randn((4,2,40,86))
z = torch.randn((4,2,40,86))
loss = F.triplet_margin_loss(x,y,z,margin=0.5)
loss_D_img_t = F.binary_cross_entropy_with_logits(x,torch.FloatTensor(x.data.size()).fill_(0))
print(loss_D_img_t)
# x = x * (torch.tensor(1.0)-loss_D_img_t)
# print(torch.tensor(1.0)-loss_D_img_t)
# score,indx = x.sort(descending = True)
# score = score.reshape(-1,1)
# print(score,indx)
