import numpy as np
import torch
from torch.nn import functional as F

SMALL_NUM = np.log(1e-45)


class DCL(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()

class DCLW(DCL):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(DCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)

def cal_single_domain_loss(features,split_num=1):
    batch_size, _, w, h = features.shape
    new_batch, new_w, new_h = batch_size*split_num*split_num,w//split_num,h//split_num

    features = F.interpolate(features, size=(new_w*split_num, new_h*split_num))
    new_length = new_w*new_h
    input = features.reshape(new_batch, -1)

    shuffle_index = torch.randperm(new_length).repeat(new_batch, 1)
    input = torch.stack([input[i][shuffle_index[i]] for i in range(new_batch)])

    anchor = input[:, :new_length // 2]
    pos = input[:, -(new_length // 2):]

    # loss_fn = DCL(temperature=0.5)
    # loss = loss_fn(anchor, pos)  # loss = tensor(-0.2726, grad_fn=<AddBackward0>

    loss_fn = DCLW(temperature=0.5, sigma=0.5)
    loss = loss_fn(anchor, pos)  # loss = tensor(38.8402, grad_fn=<AddBackward0>)
    return loss

def cal_domain_loss(source_features,target_features,split_num=1,weight = 0.07):

    assert source_features.shape[0] == target_features.shape[0]

    if not source_features.shape==target_features:
        rs_size = (target_features.size(-2), target_features.size(-1))
        source_features = F.interpolate(source_features, size=rs_size)

    source_loss = cal_single_domain_loss(source_features,split_num)
    target_loss = cal_single_domain_loss(target_features,split_num)

    DCL_loss = (source_loss+target_loss)*weight
    return DCL_loss

# random_S_input = torch.rand((4, 1, 38, 75))
# # random_T_input = torch.rand((4, 1, 40, 80))
# random_T_input = random_S_input
# loss = cal_domain_loss(random_S_input,random_T_input,split_num=4)
# print(loss)
