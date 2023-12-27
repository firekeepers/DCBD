# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from detectron2.modeling.postprocessing import detector_postprocess
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList
# from .DecoupledContrastiveLoss import cal_domain_loss
#from nwd import NuclearWassersteinDiscrepancy

import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import cv2
from fvcore.nn import sigmoid_focal_loss_jit
SMALL_NUM = np.log(1e-45)
#####################_________DA FASTER RCNN______________________________
###################
##################
#################
class image_level_Discriminator(nn.Module):
    def __init__(self, in_feature):
        super(image_level_Discriminator, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(in_feature, int(in_feature/2), kernel_size = (1, 1) ,bias = False),  
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_feature/2), int(in_feature/4), kernel_size = (1, 1) ,bias = False),  
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(int(in_feature/4), 1, kernel_size=(1, 1), bias = False)
        ).cuda()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x, 1)
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x, domain_t, alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x, domain_s, alpha=0.25,gamma=2,reduction="mean")
        return {"loss_image_d": loss}
class DiscriminatorProposal(nn.Module):
    def __init__(self, in_feature):
        super(DiscriminatorProposal, self).__init__()
        self.reducer = nn.Sequential(
            nn.Linear(in_feature, in_feature, bias = False),  
            nn.ReLU(inplace=True),
            nn.Linear(in_feature, 1, bias = False)
        ).cuda()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x, domain_t, alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x, domain_s, alpha=0.25,gamma=2,reduction="mean")
        return {"loss_instance_d": loss}

class DiscriminatorProposalDC5(nn.Module):
    def __init__(self, in_feature):
        super(DiscriminatorProposalDC5, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(in_feature, in_feature, kernel_size = (1, 1) ,bias = False),  
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_feature, 1, kernel_size=(1, 1), bias = False)
        ).cuda()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_target = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x, 1)
        if domain_target:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x, domain_t, alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x, domain_s, alpha=0.25,gamma=2,reduction="mean")
        return {"loss_instance_d": loss}
##########________________________________________________
#########
########
######
class DCL(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, pos_weight_fn=None,neg_weight_fn=None,domain=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.pos_weight_fn = pos_weight_fn
        self.neg_weight_fn = neg_weight_fn
        self.domain = domain

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """

        
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.pos_weight_fn is not None:
            positive_loss = positive_loss * self.pos_weight_fn(z1, z2)
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
        #default weight of DCLW
        pos_weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        #changed weight
        neg_weight_fn = 1
        neg_weight_fn = lambda z1, z2: 1 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(DCLW, self).__init__(pos_weight_fn=pos_weight_fn, neg_weight_fn=neg_weight_fn, temperature=temperature)
#define triplet loss functionclass triplet_loss(nn.Module): 
def triple_loss(anchor,positive,negative,margin): 
    
    pos_dist = torch.nn.functional.pairwise_distance(anchor,positive, keepdim=True)
    # (anchor - positive).pow(2).sum(1) 
    neg_dist = torch.nn.functional.pairwise_distance(anchor,negative, keepdim=True)
    # (anchor - negative).pow(2).sum(1) 
    loss = F.relu(pos_dist - neg_dist + margin) 
    # torch.triplet_margin_loss
    return loss.mean()
#we can also use #torch.nn.functional.pairwise_distance(anchor,positive, keep_dims=True), which #computes the euclidean distance.

#-0.3
def cal_single_domain_loss_DCLW(source_features,target_features,split_num=1):
    # todo:
    # split_num=1
    batch_size, _, w, h = source_features.shape
    new_batch, new_w, new_h = batch_size*split_num*split_num,w//split_num,h//split_num

    source_features = F.interpolate(source_features, size=(new_w*split_num, new_h*split_num))
    target_features = F.interpolate(target_features, size=(new_w*split_num, new_h*split_num))
    # neg_features = F.interpolate(neg, size=(new_w*split_num, new_h*split_num))
    
    new_length = new_w*new_h
    input_S = source_features.reshape(new_batch, -1)
    input_T = target_features.reshape(new_batch, -1)

    features_S = torch.nn.functional.normalize(input_S,dim=1)
    features_T = torch.nn.functional.normalize(input_T,dim=1)

    Source_anchor = features_S[:, :new_length // 2]
    Source_positi = features_S[:, -(new_length // 2):]

    Target_anchor = features_T[:, :new_length // 2]
    Target_positi = features_T[:, -(new_length // 2):]

    features_1 = torch.concat((Source_anchor,Target_anchor),dim=0)
    features_2 = torch.concat((Source_positi,Target_positi),dim=0)

    loss_fn = DCLW(temperature=0.5, sigma=0.5)
    loss_1 = loss_fn(features_1,features_2)  # loss = tensor(38.8402, grad_fn=<AddBackward0>)
    loss_2 = loss_fn(features_2,features_1)

    # neg = neg.reshape(new_batch, -1)

    # shuffle_index = torch.randperm(new_length).repeat(new_batch, 1)
    # #todo:zheli guanle yixia shuffle
    # # input = torch.stack([input[i][shuffle_index[i]] for i in range(new_batch)])
    # features_1 = torch.nn.functional.normalize(features_1,dim=1)

    # features_1 = input[:, :new_length // 2]
    # features_2 = input[:, -(new_length // 2):]

    # features_12 ,features_22 = DCL_projector(features_1,features_2)

    # features_1 = torch.nn.functional.normalize(features_1,dim=1)
    # features_2 = torch.nn.functional.normalize(features_2,dim=1)
    # nega = neg[:new_batch, :new_length // 2]

    # loss_fn = DCL(temperature=0.5)
    # loss = loss_fn(anchor, pos)  # loss = tensor(-0.2726, grad_fn=<AddBackward0>

    
    #从一张图像中找到pos和anchor，然后其他样本都作为副样本，然后设法降低同一个域中样本的权重，可以给两个权重，前半部分是同一个域
    # loss_1 = loss_fn(features_1,features_2)  # loss = tensor(38.8402, grad_fn=<AddBackward0>)
    # loss_2 = loss_fn(features_2,features_1)
    # loss_xxx = loss_fn(features_12 ,features_22)*0
    loss = loss_1+loss_2
    # loss = triple_loss(anchor,pos,neg)
    return loss

#51.64
def cal_domain_intra_consistency_constraint(features,split_num = 2,weight=0.07):
    # split_num = 2
    batch_size, _, w, h = features.shape   #75*38*1
    new_batch, new_w, new_h = batch_size*split_num*split_num,w//split_num,h//split_num

    features = F.interpolate(features, size=(new_w*split_num, new_h*split_num))
    new_length = new_w*new_h
    input = features.reshape(new_batch, -1)                               
    # neg = neg.reshape(new_batch, -1)                                  

    shuffle_index = torch.randperm(new_length).repeat(new_batch, 1)
    # todo:zheli guanle yixia shuffle
    input = torch.stack([input[i][shuffle_index[i]] for i in range(new_batch)])

    features_1 = input[:, :new_length // 2]
    features_2 = input[:, -(new_length // 2):]

    features_1 = torch.nn.functional.normalize(features_1,dim=1)
        

    loss_fn = DCLW(temperature=0.5, sigma=0.5)
    loss_1 = loss_fn(features_1,features_2)  # loss = tensor(38.8402, grad_fn=<AddBackward0>)
    loss_2 = loss_fn(features_2,features_1)
    # loss_xxx = loss_fn(features_12 ,features_22)*0
    loss = (loss_1+loss_2)*weight
    # loss = triple_loss(anchor,pos,neg)
    return loss


def cal_single_domain_loss_additional_branch(DCL_projector,features_source,features_target,split_num=1):
    split_num=1
    batch_size, _, w, h = features.shape
    new_batch, new_w, new_h = batch_size*split_num*split_num,w//split_num,h//split_num

    features = F.interpolate(features, size=(new_w*split_num, new_h*split_num))
    # neg_features = F.interpolate(neg, size=(new_w*split_num, new_h*split_num))
    
    # new_length = new_w*new_h
    # input = features.reshape(new_batch, -1)
    # neg = neg.reshape(new_batch, -1)

    # shuffle_index = torch.randperm(new_length).repeat(new_batch, 1)
    #todo:zheli guanle yixia shuffle
    # input = torch.stack([input[i][shuffle_index[i]] for i in range(new_batch)])
    

    # features_1 = input[:, :new_length // 2]
    # features_2 = input[:, -(new_length // 2):]

    features_12 ,features_22 = DCL_projector(features_source,features_2)

    features_1 = torch.nn.functional.normalize(features_1,dim=1)
    features_2 = torch.nn.functional.normalize(features_2,dim=1)
    # nega = neg[:new_batch, :new_length // 2]

    # loss_fn = DCL(temperature=0.5)
    # loss = loss_fn(anchor, pos)  # loss = tensor(-0.2726, grad_fn=<AddBackward0>

    loss_fn = DCLW(temperature=0.5, sigma=0.5)
    #从一张图像中找到pos和anchor，然后其他样本都作为副样本，然后设法降低同一个域中样本的权重，可以给两个权重，前半部分是同一个域
    loss_1 = loss_fn(features_1,features_2)  # loss = tensor(38.8402, grad_fn=<AddBackward0>)
    loss_2 = loss_fn(features_2,features_1)
    # loss_xxx = loss_fn(features_12 ,features_22)*0
    loss = loss_1+loss_2
    # loss = triple_loss(anchor,pos,neg)
    return loss

def cal_domain_loss_DCLW(source_features,target_features,split_num=1,weight = 0.07):

    assert source_features.shape[0] == target_features.shape[0]

    if not source_features.shape==target_features:
        rs_size = (target_features.size(-2), target_features.size(-1))
        source_features = F.interpolate(source_features, size=rs_size)

    # domain_contrastive_feature = torch.cat((source_features,target_features),dim=0)
    # domain_contrastive_loss = cal_single_domain_loss_DCLW(DCL_projection_layer,domain_contrastive_feature,split_num)
    #todo:guided lua
    domain_contrastive_loss = cal_single_domain_loss_DCLW(source_features,target_features,split_num)
    DCL_loss = (domain_contrastive_loss)*weight

    #domain_intra consistency constraint
    # source_loss = cal_single_domain_loss(source_features,split_num)
    # target_loss = cal_single_domain_loss(target_features,split_num)
    # DCL_loss = (source_loss+target_loss)*weight
    
    return DCL_loss

def cal_single_domain_loss(features,split_num=1):
    batch_size, _, w, h = features.shape
    new_batch, new_w, new_h = batch_size*split_num*split_num,w//split_num,h//split_num

    features = F.interpolate(features, size=(new_w*split_num, new_h*split_num))
    # neg_features = F.interpolate(neg, size=(new_w*split_num, new_h*split_num))
    
    new_length = new_w*new_h
    input = features.reshape(new_batch, -1)
    # neg = neg.reshape(new_batch, -1)

    shuffle_index = torch.randperm(new_length).repeat(new_batch, 1)
    #todo:zheli guanle yixia shuffle
    # input = torch.stack([input[i][shuffle_index[i]] for i in range(new_batch)])
    input = torch.nn.functional.normalize(input)

    anchor = input[:, :new_length // 2]
    pos = input[:, -(new_length // 2):]
    # nega = neg[:new_batch, :new_length // 2]

    # loss_fn = DCL(temperature=0.5)
    # loss = loss_fn(anchor, pos)  # loss = tensor(-0.2726, grad_fn=<AddBackward0>

    loss_fn = DCLW(temperature=0.5, sigma=0.5)
    loss = loss_fn(anchor, pos)  # loss = tensor(38.8402, grad_fn=<AddBackward0>)

    # loss = triple_loss(anchor,pos,neg)
    return loss

def cal_domain_loss(source_features,target_features,split_num=1,weight = 0.07):

    assert source_features.shape[0] == target_features.shape[0]

    if not source_features.shape==target_features:
        rs_size = (target_features.size(-2), target_features.size(-1))
        source_features = F.interpolate(source_features, size=rs_size)

    # domain_contrastive_feature = torch.cat((source_features,target_features),dim=0)
    # domain_contrastive_loss = cal_single_domain_loss(domain_contrastive_feature,split_num)

    source_loss = cal_single_domain_loss(source_features,target_features,split_num)
    target_loss = cal_single_domain_loss(target_features,source_features,split_num)

    DCL_loss = (source_loss+target_loss)*weight
    return DCL_loss

############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        feature = self.leaky_relu(x)
        x = F.sigmoid(self.classifier(feature))
        # print(x.mean())
        # x = self.classifier(x)
        return x

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """
    def __init__(self, n_features):
        super(SimCLR, self).__init__()
        # Replace the fc layer with an Identity function
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.LeakyReLU(),
            # nn.Linear(self.n_features, projection_dim, bias=False),
            nn.Linear(n_features, n_features//2, bias=False),
        )

    def forward(self, x_i, x_j):

        z_i = self.projector(x_i)
        z_j = self.projector(x_j)
        return z_i, z_j
    

# class _InstanceDA(nn.Module):
#     def __init__(self, in_channel,ndf1=256, ndf2=128):
#         super(_InstanceDA, self).__init__()
#         self.dc_ip1 = nn.Conv2d(in_channel, ndf1, kernel_size=3, padding=1)
#         self.dc_relu1 = nn.ReLU()
#         self.dc_drop1 = nn.Dropout(p=0.5)

#         self.dc_ip2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
#         self.dc_relu2 = nn.ReLU()
#         self.dc_drop2 = nn.Dropout(p=0.5)

#         self.clssifer = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)


#     def forward(self, x):
#         # x = grad_reverse(x)
#         x = self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
#         x = self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
#         x = F.sigmoid(self.clssifer(x))
#         return x


class FCDiscriminator_proposal(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_proposal, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################
def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h
################ Gradient reverse function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

#######################

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        # dis_loss_weight: float = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.gaussian2D = gaussian2D(3)
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # @yujheli: you may need to build your discriminator here

        self.dis_type = dis_type
        #todo:oral_code
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) # Need to know the channel
        # self.DCL_projection_layer = SimCLR(1425) # Need to know the channel
        #todo:end
       ####################
        # self.image_level_discriminator =  image_level_Discriminator(self.backbone._out_feature_channels[self.dis_type])
        # self.discriminatorProposalDC5 = DiscriminatorProposalDC5(2048)
        ###########################DA_FASTER rcnn

       ######################
    # def build_discriminator(self):
    #     self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel
    # def build_DCL_Projector(self):
    #     self.DCL_projection_layer = SimCLR(1425).to(self.device)
    # def build_discriminator_ins(self):
    #     self.D_ins = FCDiscriminator_proposal(self.backbone._out_feature_channels[self.dis_type]).to(self.device)
    #     self.D_img = _InstanceDA(self.backbone._out_feature_channels[self.dis_type])
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            # "dis_loss_ratio": cfg.xxx,
        }

    def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
        """Generate 2D gaussian kernel.

        Args:
            radius (int): Radius of gaussian kernel.
            sigma (int): Sigma of gaussian function. Default: 1.
            dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
            device (str): Device of gaussian tensor. Default: 'cpu'.

        Returns:
            h (Tensor): Gaussian kernel with a
                ``(2 * radius + 1) * (2 * radius + 1)`` shape.
        """
        x = torch.arange(
            -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
        y = torch.arange(
            -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

        # h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()
        h = (-(x * x + y * y) / (5 * sigma * sigma)).exp()
        # h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        return h

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, [batched_inputs], image_sizes
        ):
            # height = image_size[0]
            # width = image_size[1]
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    def inference_demo(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (tensor): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image([batched_inputs])
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return self._postprocess(results, batched_inputs, images.image_sizes)
        # else:
        return results
    def forward(
        self, batched_inputs,gt_proposals='', branch="supervised", given_proposals=None, val_mode=False
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image. 
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        #todo::::::note there
        # if self.D_img == None:
        #     self.build_discriminator()
        # if self.DCL_projection_layer == None:
        #     self.build_DCL_Projector()

        if (not self.training) and (not val_mode):  # only conduct when testing mode
            # return self.inference(batched_inputs)
            #todo:only use when wanna visible results
            return self.inference_demo(batched_inputs)
        source_label = 0
        target_label = 1
        soft_source_label = 0
        soft_target_label = 1
        if branch == "supervised_validation_DCBD":
            # Region proposal network
            images = self.preprocess_image(batched_inputs)

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            with torch.no_grad():
                features = self.backbone(images.tensor)
                proposals_rpn, _ = self.proposal_generator(
                    images, features, gt_instances
                )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            # losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None
        if branch == "domain":
            # self.D_img.train()
            # images = self.preprocess_image(batched_inputs)
            images_s, images_t = self.preprocess_image_train(batched_inputs)
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            if "instances_unlabeled" in batched_inputs[0]:
                fake_instances = [x["instances_unlabeled"].to(self.device) for x in batched_inputs]
            else:
                fake_instances = None
            features = self.backbone(images_s.tensor)

            # for i in range(len(batched_inputs)):
            #     if batched_inputs[i]['file_name'] == 'datasets/cityscapes/leftImg8bit/train/bremen/bremen_000253_000019_leftImg8bit.png':
            #         print('hi_source')
            #     if batched_inputs[i]['file_name_unlabeled'] == 'datasets/cityscapes_foggy/leftImg8bit_foggy/train/hanover/hanover_000000_026804_leftImg8bit_foggy_beta_0.02.png':
            #         print("hi_target") 

            # import pdb
            # pdb.set_trace()
#todo:原来的东西的样子
            # features_s = grad_reverse(features[self.dis_type])
            # features_s_p6 = features["p6"]
            # D_img_out_s_p6 = self.D_img(features_s_p6)
            # loss_D_img_s_p6 = F.binary_cross_entropy_with_logits(D_img_out_s_p6,
            #                                                      torch.FloatTensor(D_img_out_s_p6.data.size()).fill_(
            #                                                          soft_source_label).to(self.device))
            # 改过的
            #todo:
            # features_s = features[self.dis_type]
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(soft_source_label).to(self.device))

            # features_s_p5 = grad_reverse(features["p5"])
            # D_img_out_s_p5 = self.D_img(features_s_p5)
            # loss_D_img_s_p5 = F.binary_cross_entropy_with_logits(D_img_out_s_p5, torch.FloatTensor(D_img_out_s_p5.data.size()).fill_(soft_source_label).to(self.device))
#改过的
#             # features_s_p3 = features["p3"]
#             # D_img_out_s_p3 = self.D_img(features_s_p3)
#             # loss_D_img_s_p3 = F.binary_cross_entropy_with_logits(D_img_out_s_p3,
#             #                                                      torch.FloatTensor(D_img_out_s_p3.data.size()).fill_(
#             #                                                          soft_source_label).to(self.device))
#
            # D_img_out_s_p4 = self.D_img(grad_reverse(features["p4"]))
            # loss_D_img_s_p4 = F.binary_cross_entropy_with_logits(D_img_out_s_p4,
            #                                                      torch.FloatTensor(D_img_out_s_p4.data.size()).fill_(
            #                                                          soft_source_label).to(self.device))
#
#             # D_img_out_s_p2 = self.D_img(features["p2"])
#             # loss_D_img_s_p2 = F.binary_cross_entropy_with_logits(D_img_out_s_p2,
#             #                                                      torch.FloatTensor(D_img_out_s_p2.data.size()).fill_(
#             #                                                          soft_source_label).to(self.device))
#
            # loss_D_img_s = loss_D_img_s_p5 + loss_D_img_s_p4 
                           # + loss_D_img_s_p3 \
                           # + loss_D_img_s_p2
                           # + loss_D_img_s_p6
            # features_ins_s = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.in_features],
            #                                            [x.gt_boxes for x in gt_instances])

            #todo:86.57
            # D_ins_out_s = self.D_ins(features_ins_s)
            # D_ins_out_s = torch.mul(D_ins_out_s, gaussian2D(3, 3, device=self.device)).permute(1, 2, 3, 0)
            # loss_D_ins_s = F.binary_cross_entropy_with_logits(D_ins_out_s,
            #                                                   torch.FloatTensor(D_ins_out_s.data.size()).fill_(
            #                                                       source_label).to(self.device))

            # D_ins_out_s = self.D_img(features_ins_s).view(features_ins_s.size(0), -1)
            # weights_s = torch.ones((D_ins_out_s.size(0), D_ins_out_s.size(1))).to(self.device) * gaussian2D(3, 3,
            #             device=self.device).view(-1)
            # # scores_s = torch.cat([x.scores for x in fake_instances], dim=0).view(-1, 1)
            # # weighss_gus_score = scores_s * weights_s
            # loss_D_ins_s = F.binary_cross_entropy_with_logits(D_ins_out_s,
            #                                                   torch.FloatTensor(D_ins_out_s.data.size()).fill_(
            #                                                       soft_source_label).to(self.device),
            #                                                   weight=weights_s)

            # features_proposal_s = [features[self.dis_type]]is_quantized = {bool} False
            # # features_proposal_s = grad_reverse(features_proposal_s)
            # features_proposal_s = self.roi_heads.box_pooler(features_proposal_s, [x.gt_boxes for x in gt_instances])
            # features_proposal_s = grad_reverse(features_proposal_s)
            # D_proposal_s = self.D_proposal(features_proposal_s)
            # #
            # loss_D_proposal_s = F.binary_cross_entropy_with_logits(D_proposal_s,
            #                                                        torch.FloatTensor(D_proposal_s.data.size()).fill_(
            #                                                            source_label).to(self.device))


#dis tgt feature w/o grl
            features_target = self.backbone(images_t.tensor)
            # features_t = grad_reverse(features_target[self.dis_type])
            # #todo:看看加grl会怎么样，会很差，2000+无了
            # #不加grl了
            # D_img_out_t = self.D_img(grad_reverse(features_target[self.dis_type]))
            # D_img_out_t_p6 = self.D_img(features_target["p6"])
            # #
            # # # features_t = grad_reverse(features_t['p2'])
            # # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t_p6 = F.binary_cross_entropy_with_logits(D_img_out_t_p6,
            #                                                      torch.FloatTensor(D_img_out_t_p6.data.size()).fill_(
            #                                                          soft_target_label).to(self.device))
            # todo:
            # features_t = features_target[self.dis_type]
            features_t = grad_reverse(features_target[self.dis_type])
            D_img_out_t = self.D_img(features_t)
            loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(soft_target_label).to(self.device))

            # D_img_out_t_p5 = self.D_img(grad_reverse(features_target["p5"]))
            # # #
            # # # # features_t = grad_reverse(features_t['p2'])
            # # # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t_p5 = F.binary_cross_entropy_with_logits(D_img_out_t_p5, torch.FloatTensor(D_img_out_t_p5.data.size()).fill_(soft_target_label).to(self.device))
            # #
            # # # D_img_out_t_p3 = self.D_img(grad_reverse(features_target["p3"]))
            # # # loss_D_img_t_p3 = F.binary_cross_entropy_with_logits(D_img_out_t_p3,
            # # #                                                   torch.FloatTensor(D_img_out_t_p3.data.size()).fill_(
            # # #                                                       soft_target_label).to(self.device))
            # # #域无关的p4
            # D_img_out_t_p4 = self.D_img(grad_reverse(features_target["p4"]))
            # loss_D_img_t_p4 = F.binary_cross_entropy_with_logits(D_img_out_t_p4,
            #                                                      torch.FloatTensor(D_img_out_t_p4.data.size()).fill_(
            #                                                          soft_target_label).to(self.device))

            # D_img_out_t_p2 = self.D_img(features_target["p2"])
            # loss_D_img_t_p2 = F.binary_cross_entropy_with_logits(D_img_out_t_p2,
            #                                                      torch.FloatTensor(D_img_out_t_p2.data.size()).fill_(
            #                                                          soft_target_label).to(self.device))

            # loss_D_img_t = loss_D_img_t_p5 + loss_D_img_t_p4
                           # + loss_D_img_t_p3 + loss_D_img_t_p2
                           # + loss_D_img_t_p6
            #TODO:ins_
#todo：
            # features_ins_t = self.roi_heads.box_pooler([features_target[f] for f in self.roi_heads.in_features],
            #                                            [x.pred_boxes for x in fake_instances])
            #todo:86.57
            # D_ins_out_t = self.D_ins(features_ins_t)
            # D_ins_out_t = torch.mul(D_ins_out_t, 1.3 - gaussian2D(3, 3, device=self.device)).permute(1, 2, 3, 0)
            # loss_D_ins_t = F.binary_cross_entropy_with_logits(D_ins_out_t,
            #                                                   torch.FloatTensor(D_ins_out_t.data.size()).fill_(
            #                                                       target_label).to(self.device),
            #                                                   weight=torch.cat([x.scores for x in fake_instances],
            #                                                                    dim=0))

            # D_ins_out_t = self.D_img(features_ins_t).view(features_ins_t.size(0), -1)
            # weights = torch.ones((D_ins_out_t.size(0), D_ins_out_t.size(1))).to(self.device)\
            #           * gaussian2D(3, 3, device=self.device).view( -1)
            # scores = torch.cat([x.scores for x in fake_instances], dim=0).view(-1, 1)
            # weighst_gus_score = scores * weights
            # loss_D_ins_t = F.binary_cross_entropy_with_logits(D_ins_out_t,
            #                                                   torch.FloatTensor(D_ins_out_t.data.size()).fill_(
            #                                                       soft_target_label).to(self.device),
            #                                                   weight=weighst_gus_score)

            # features_proposal_t = [features_target[self.dis_type]]
            # # features_proposal_s = grad_reverse(features_proposal_s)
            # features_proposal_t = self.roi_heads.box_pooler(features_proposal_t, [x.gt_boxes for x in gt_instances])
            # features_proposal_t = grad_reverse(features_proposal_t)
            # D_proposal_t = self.D_proposal(features_proposal_t)
            # #
            # loss_D_proposal_t = F.binary_cross_entropy_with_logits(D_proposal_t,
            #                                                        torch.FloatTensor(D_proposal_t.data.size()).fill_(
            #                                                            target_label).to(self.device))

            # import pdb
            # pdb.set_trace()

            # fwd = len(features_ins_s)
            # discrepancy = NuclearWassersteinDiscrepancy(self.roi_heads.box_predictor, fwd)
            #  
            # feat = torch.cat([features_ins_s, features_ins_t], dim=0)
            # discrepancy_loss = discrepancy(self.roi_heads.box_head(feat))
            # D_img_out_s_pro,D_img_out_t_pro = self.DCL_projection_layer(D_img_out_s,D_img_out_t)

            # loss_DCL = cal_domain_loss_DCLW(self.DCL_projection_layer,features_DCL_s,features_DCL_t,split_num=2,weight=0.02)

            # loss_DCL = cal_domain_loss_DCLW(D_img_out_s,D_img_out_t,split_num=2,weight=0.02) #0.02 for 51.02

            #todo:loss intra_consistency_constraint 51.64 0.02
            # loss_DCL_S = cal_domain_intra_consistency_constraint(D_img_out_s,split_num=2,weight=0.02)
            # loss_DCL_T = cal_domain_intra_consistency_constraint(D_img_out_t,split_num=2,weight=0.02)
            # loss_DCL = loss_DCL_S+loss_DCL_T
            #todo: cal_domain_intra_consistency_constraint

            #FPN IDCC
            # loss_DCL_S = cal_domain_intra_consistency_constraint(D_img_out_s_p5,split_num=2,weight=0.02)
            # loss_DCL_T = cal_domain_intra_consistency_constraint(D_img_out_t_p5,split_num=2,weight=0.02)
            # loss_DCL = loss_DCL_S+loss_DCL_T
            
 
            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s
            # losses["loss_D_ins_s"] = loss_D_ins_s*0.1
            # losses["loss_D_proposal_s"] = loss_D_proposal_s
            losses["loss_D_img_t"] = loss_D_img_t
            #todo:IDCC
            # losses["loss_IDCC"] = loss_DCL
            # print(loss_DCL)
            # losses["loss_nwd"] = discrepancy_loss * 0.01
            # losses["loss_D_ins_t"] = loss_D_ins_t*0.1
            # losses["loss_D_proposal_t"] = loss_D_proposal_t
            return losses, [], [], None

        # self.D_img.eval()
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
  
        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised":

            # features_s_p5 = grad_reverse(features["p5"])
            # D_img_out_s_p5 = self.D_img(features_s_p5)
            # loss_D_img_s_p5 = F.binary_cross_entropy_with_logits(D_img_out_s_p5,
            #                                                      torch.FloatTensor(D_img_out_s_p5.data.size()).fill_(
            #                                                          soft_source_label).to(self.device))
            # # 改过的
            # # features_s_p3 = features["p3"]
            # # D_img_out_s_p3 = self.D_img(features_s_p3)
            # # loss_D_img_s_p3 = F.binary_cross_entropy_with_logits(D_img_out_s_p3,
            # #                                                      torch.FloatTensor(D_img_out_s_p3.data.size()).fill_(
            # #                                                          soft_source_label).to(self.device))

            # D_img_out_s_p4 = self.D_img(grad_reverse(features["p4"]))
            # loss_D_img_s_p4 = F.binary_cross_entropy_with_logits(D_img_out_s_p4,
            #                                                      torch.FloatTensor(D_img_out_s_p4.data.size()).fill_(
            #                                                          soft_source_label).to(self.device))

            # D_img_out_s_p2 = self.D_img(features["p2"])
            # loss_D_img_s_p2 = F.binary_cross_entropy_with_logits(D_img_out_s_p2,
            #                                                      torch.FloatTensor(D_img_out_s_p2.data.size()).fill_(
            #                                                          soft_source_label).to(self.device))

            # loss_D_img_s = loss_D_img_s_p5 + loss_D_img_s_p4 



            # features_s = features[self.dis_type]
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            # print(D_img_out_s.mean())
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(soft_source_label).to(self.device))

# #todo:ins
#             # features_ins_s = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.in_features],
#             #                                            [x.gt_boxes for x in gt_instances])
#             #todo:86.57
#             # D_ins_out_s = self.D_ins(features_ins_s)
#             # D_ins_out_s = torch.mul(D_ins_out_s, gaussian2D(3, 3, device=self.device)).permute(1, 2, 3, 0)
#             # loss_D_ins_s = F.binary_cross_entropy_with_logits(D_ins_out_s,
#             #                                                   torch.FloatTensor(D_ins_out_s.data.size()).fill_(
#             #                                                       source_label).to(self.device))
#             loss_DCL = cal_domain_loss_DCLW(self.DCL_projection_layer,D_img_out_s,D_img_out_s,split_num=1,weight=0)

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
        
            losses["loss_D_img_s"] = (loss_D_img_s) * 0.02
            # losses["loss_D_DCL"] = loss_DCL
       

            return losses, [], [], None
        elif branch == "supervised_compare":
            # features_t = grad_reverse(features[self.dis_type])
            # x = features[self.dis_type]
            # D_img_out_t_p6 = self.D_img(features["p6"])
            # loss_D_img_t_p6 = F.binary_cross_entropy_with_logits(D_img_out_t_p6,
            #                                                      torch.FloatTensor(D_img_out_t_p6.data.size()).fill_(
            #                                                          soft_target_label).to(self.device))
            #todo:
            # features_t = features[self.dis_type]
            features_t = grad_reverse(features[self.dis_type])
            D_img_out_t = self.D_img(features_t)
            loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(soft_target_label).to(self.device))
            #just 
        #    loss_DCL = cal_domain_loss_DCLW(self.DCL_projection_layer,D_img_out_t,D_img_out_t,split_num=1,weight=0)
            # D_img_out_t_p5 = self.D_img(grad_reverse(features["p5"]))
            # loss_D_img_t_p5 = F.binary_cross_entropy_with_logits(D_img_out_t_p5,
            #                                                   torch.FloatTensor(D_img_out_t_p5.data.size()).fill_(
            #                                                       soft_target_label).to(self.device))
            # # #
            # # # # D_img_out_t_p3 = self.D_img(grad_reverse(features["p3"]))
            # # # # loss_D_img_t_p3 = F.binary_cross_entropy_with_logits(D_img_out_t_p3,
            # # # #                                                   torch.FloatTensor(D_img_out_t_p3.data.size()).fill_(
            # # # #                                                       soft_target_label).to(self.device))
            # # #
            # D_img_out_t_p4 = self.D_img(grad_reverse(features["p4"]))
            # loss_D_img_t_p4 = F.binary_cross_entropy_with_logits(D_img_out_t_p4,
            #                                                      torch.FloatTensor(D_img_out_t_p4.data.size()).fill_(
            #                                                          soft_target_label).to(self.device))

            # D_img_out_t_p2 = self.D_img(features["p2"])
            # loss_D_img_t_p2 = F.binary_cross_entropy_with_logits(D_img_out_t_p2,
            #                                                      torch.FloatTensor(D_img_out_t_p2.data.size()).fill_(
            #                                                          soft_target_label).to(self.device))

            # loss_D_img_t = loss_D_img_t_p5 + loss_D_img_t_p4
                           # + loss_D_img_t_p3  + loss_D_img_t_p2
                           # + loss_D_img_t_p6
#todo:ins
            # features_ins_t = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.in_features],
            #                                            [x.gt_boxes for x in gt_instances])
            #todo:86.57
            # D_ins_out_t = self.D_ins(features_ins_t)
            # D_ins_out_t = torch.mul(D_ins_out_t,1.3 - gaussian2D(3, 3, device=self.device)).permute(1, 2, 3, 0)
            # loss_D_ins_t = F.binary_cross_entropy_with_logits(D_ins_out_t,
            #                                                   torch.FloatTensor(D_ins_out_t.data.size()).fill_(
            #                                                       source_label).to(self.device))

            # D_ins_out_t = self.D_img(features_ins_t).view(features_ins_t.size(0), -1)
            # weights = torch.ones((D_ins_out_t.size(0), D_ins_out_t.size(1))).to(self.device) * gaussian2D(3, 3,
            #                                                                                               device=self.device).view(
            #     -1)
            # # scores = torch.cat([x.scores for x in fake_instances], dim=0).view(-1, 1)
            # # weighst_gus_score = scores * weights
            # loss_D_ins_t = F.binary_cross_entropy_with_logits(D_ins_out_t,
            #                                                   torch.FloatTensor(D_ins_out_t.data.size()).fill_(
            #                                                       soft_target_label).to(self.device),
            #                                                   weight=weights)

            # features_proposal_t = [features[self.dis_type]]
            # # features_proposal_s = grad_reverse(features_proposal_s)
            # features_proposal_t = self.roi_heads.box_pooler(features_proposal_t, [x.gt_boxes for x in gt_instances])
            # features_proposal_t = grad_reverse(features_proposal_t)
            # D_proposal_t = self.D_proposal(features_proposal_t)
            # #
            # loss_D_proposal_t = F.binary_cross_entropy_with_logits(D_proposal_t,
            #                                                        torch.FloatTensor(D_proposal_t.data.size()).fill_(
            #                                                         target_label).to(self.device))

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses["loss_D_img_t"] = loss_D_img_t * 0.02 #0.1 for sonar
            # losses["loss_D_DCL"] = loss_DCL
    
            return losses, [], [], None

        elif branch == "supervised_target":

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))
            #todo:domain_weights

            # with torch.no_grad():
            #     features_ins_s = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.in_features],
            #                                                [x.gt_boxes for x in gt_instances])
            #     domain_weights = [self.D_img(features).mean() for features in features_ins_s]
            #     index = 0
            #     # valid_map = [len(gt_instances[i]) > 0 for i in range(len(gt_instances))]
            #     for instance in gt_instances:
            #         conf = torch.tensor(instance.scores)
            #         dom_conf = torch.tensor(domain_weights[index:len(instance)+index]).to(torch.device("cuda"))
            #         instance.scores = conf*torch.sqrt(abs(0.5-dom_conf))
            #         index += len(instance)

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None
        elif branch == "compare_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            # proposals_rpn, _ = self.proposal_generator(
            #     images, features, None, compute_loss=False
            # )
            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!

            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                # proposals_rpn,
                gt_proposals,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return proposals_roih, ROI_predictions

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return {}, proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "domain_DRD":

            images_s, images_t = self.preprocess_image_train(batched_inputs)

            features = self.backbone(images_s.tensor)

            features_s = features[self.dis_type]
            # features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s,
                                                              torch.FloatTensor(D_img_out_s.data.size()).fill_(
                                                                  soft_source_label).to(self.device))
            # dis tgt feature w/o grl
            features_target = self.backbone(images_t.tensor)

            features_t = features_target[self.dis_type]
            # features_t = grad_reverse(features[self.dis_type])
            D_img_out_t = self.D_img(features_t)
            loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t,
                                                              torch.FloatTensor(D_img_out_t.data.size()).fill_(
                                                                  soft_target_label).to(self.device))

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s
            # losses["loss_D_ins_s"] = loss_D_ins_s*0.1
            # losses["loss_D_proposal_s"] = loss_D_proposal_s
            losses["loss_D_img_t"] = loss_D_img_t
            # losses["loss_nwd"] = discrepancy_loss * 0.01
            # losses["loss_D_ins_t"] = loss_D_ins_t*0.1
            # losses["loss_D_proposal_t"] = loss_D_proposal_t
            return losses

        elif branch == "domain_DID":

            images_s, images_t = self.preprocess_image_train(batched_inputs)

            features = self.backbone(images_s.tensor)

            # features_s = features[self.dis_type]
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s,
                                                              torch.FloatTensor(D_img_out_s.data.size()).fill_(
                                                                  soft_source_label).to(self.device))
            # dis tgt feature w/o grl
            features_target = self.backbone(images_t.tensor)

            # features_t = features_target[self.dis_type]
            features_t = grad_reverse(features_target[self.dis_type])
            D_img_out_t = self.D_img(features_t)
            loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t,
                                                              torch.FloatTensor(D_img_out_t.data.size()).fill_(
                                                                  soft_target_label).to(self.device))

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s

            losses["loss_D_img_t"] = loss_D_img_t
            return losses
        elif branch == "DA-faster":
            images_s, images_t = self.preprocess_image_train(batched_inputs)
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            if "instances_unlabeled" in batched_inputs[0]:
                fake_instances = [x["instances_unlabeled"].to(self.device) for x in batched_inputs]
            else:
                fake_instances = None

            need_backprop = torch.FloatTensor(1).cuda()
            tgt_need_backprop = torch.FloatTensor(1).cuda()

            features = self.backbone(images_s.tensor)

            source_proposals_rpn, source_proposal_losses = self.proposal_generator(
                images_s, features, gt_instances
            )

            # roi_head lower branch
            _, source_detector_losses = self.roi_heads(
                images_s,
                features,
                source_proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )
            # target_proposals_rpn, target_proposal_losses = self.proposal_generator(
            #     images_t, features_t, fake_instances
            # )

            # # roi_head lower branch
            # _, target_detector_losses = self.roi_heads(
            #     images_t,
            #     features_t,
            #     target_proposals_rpn,
            #     compute_loss=True,
            #     targets=fake_instances,
            #     branch=branch,
            # )

            features_s = grad_reverse(features[self.dis_type])

            features_ins_s = self.roi_heads.box_pooler([features_s],
                                                       [x.gt_boxes for x in gt_instances])
            
            base_score, base_label = self.RCNN_imageDA(features_ins_s, need_backprop)
            base_prob = F.log_softmax(base_score, dim=1)
            #loss 1
            DA_img_loss_cls = F.nll_loss(base_prob, base_label)
            
            instance_sigmoid, same_size_label = self.RCNN_instanceDA(features_ins_s, need_backprop)
            instance_loss = nn.BCELoss()
            #loss 2
            DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)
            consistency_prob = F.softmax(base_score, dim=1)[:,1,:,:]
            consistency_prob=torch.mean(consistency_prob)
            consistency_prob=consistency_prob.repeat(instance_sigmoid.size())
            #loss 3
            DA_cst_loss=self.consistency_loss(instance_sigmoid,consistency_prob.detach())

            # dis tgt feature w/o grl
            features_target = self.backbone(images_t.tensor)

            # features_t = features_target[self.dis_type]
            features_t = grad_reverse(features_target[self.dis_type])
            D_img_out_t = self.D_img(features_t)
            features_ins_t = self.roi_heads.box_pooler([features_t],
                                                       [x.gt_boxes for x in gt_instances])
            tgt_instance_sigmoid, tgt_same_size_label = \
                self.RCNN_instanceDA(features_ins_t, tgt_need_backprop)
            tgt_instance_loss = nn.BCELoss()
            tgt_base_score, tgt_base_label = \
                self.RCNN_imageDA(features_t, tgt_need_backprop)
            tgt_base_prob = F.log_softmax(tgt_base_score, dim=1)
            #loss 1
            tgt_DA_img_loss_cls = F.nll_loss(tgt_base_prob, tgt_base_label)
            #loss 2
            tgt_DA_ins_loss_cls = \
                tgt_instance_loss(tgt_instance_sigmoid, tgt_same_size_label)
            tgt_consistency_prob = F.softmax(tgt_base_score, dim=1)[:, 0, :, :]
            tgt_consistency_prob = torch.mean(tgt_consistency_prob)
            tgt_consistency_prob = tgt_consistency_prob.repeat(tgt_instance_sigmoid.size())
            #loss 3
            tgt_DA_cst_loss = self.consistency_loss(tgt_instance_sigmoid, tgt_consistency_prob.detach())

            losses = {}
            losses.update(source_detector_losses)
            losses.update(source_proposal_losses)
            #source
            losses["loss_DA_S_img"] = DA_img_loss_cls
            losses["loss_DA_S_ins"] = DA_ins_loss_cls
            losses["loss_DA_S_cst"] = DA_cst_loss
            #target
            losses["loss_DA_S_img"] = tgt_DA_img_loss_cls
            losses["loss_DA_S_ins"] = tgt_DA_ins_loss_cls
            losses["loss_DA_S_cst"] = tgt_DA_cst_loss

            # losses["loss_D_img_t"] = loss_D_img_t
            return losses

        elif branch == "t-SNE":

            features_ins_t = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.in_features],
                                                       [x.gt_boxes for x in gt_instances])
            box_features = self.roi_heads.box_head(features_ins_t)
            cls_features = self.roi_heads.box_predictor(box_features)[0]
            lenss = len(box_features)
            return cls_features,lenss,[],[]


        elif branch == "val_loss":
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch



@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)
        # return inference_demo(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None



# import numpy as np
# from detectron2.modeling.postprocessing import detector_postprocess
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
# from detectron2.config import configurable
# # from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# # from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
# import logging
# from typing import Dict, Tuple, List, Optional
# from collections import OrderedDict
# from detectron2.modeling.proposal_generator import build_proposal_generator
# from detectron2.modeling.backbone import build_backbone, Backbone
# from detectron2.modeling.roi_heads import build_roi_heads
# from detectron2.utils.events import get_event_storage
# from detectron2.structures import ImageList

# ############### Image discriminator ##############
# class FCDiscriminator_img(nn.Module):
#     def __init__(self, num_classes, ndf1=256, ndf2=128):
#         super(FCDiscriminator_img, self).__init__()

#         self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
#         self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.leaky_relu(x)
#         x = self.conv2(x)
#         x = self.leaky_relu(x)
#         x = self.conv3(x)
#         x = self.leaky_relu(x)
#         x = self.classifier(x)
#         return x

# class FCDiscriminator_proposal(nn.Module):
#     def __init__(self, num_classes, ndf1=256, ndf2=128):
#         super(FCDiscriminator_proposal, self).__init__()

#         self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
#         self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.leaky_relu(x)
#         x = self.conv2(x)
#         x = self.leaky_relu(x)
#         x = self.conv3(x)
#         x = self.leaky_relu(x)
#         x = self.classifier(x)
#         return x
# #################################
# def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
#     """Generate 2D gaussian kernel.

#     Args:
#         radius (int): Radius of gaussian kernel.
#         sigma (int): Sigma of gaussian function. Default: 1.
#         dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
#         device (str): Device of gaussian tensor. Default: 'cpu'.

#     Returns:
#         h (Tensor): Gaussian kernel with a
#             ``(2 * radius + 1) * (2 * radius + 1)`` shape.
#     """
#     x = torch.arange(
#         -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
#     y = torch.arange(
#         -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

#     h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

#     h[h < torch.finfo(h.dtype).eps * h.max()] = 0
#     return h
# ################ Gradient reverse function
# class GradReverse(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.neg()

# def grad_reverse(x):
#     return GradReverse.apply(x)

# #######################

# @META_ARCH_REGISTRY.register()
# class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

#     @configurable
#     def __init__(
#         self,
#         *,
#         backbone: Backbone,
#         proposal_generator: nn.Module,
#         roi_heads: nn.Module,
#         pixel_mean: Tuple[float],
#         pixel_std: Tuple[float],
#         input_format: Optional[str] = None,
#         vis_period: int = 0,
#         dis_type: str,
#         # dis_loss_weight: float = 0,
#     ):
#         """
#         Args:
#             backbone: a backbone module, must follow detectron2's backbone interface
#             proposal_generator: a module that generates proposals using backbone features
#             roi_heads: a ROI head that performs per-region computation
#             pixel_mean, pixel_std: list or tuple with #channels element, representing
#                 the per-channel mean and std to be used to normalize the input image
#             input_format: describe the meaning of channels of input. Needed by visualization
#             vis_period: the period to run visualization. Set to 0 to disable.
#         """
#         super(GeneralizedRCNN, self).__init__()
#         self.backbone = backbone
#         self.proposal_generator = proposal_generator
#         self.roi_heads = roi_heads
#         self.gaussian2D = gaussian2D(3)
#         self.input_format = input_format
#         self.vis_period = vis_period
#         if vis_period > 0:
#             assert input_format is not None, "input_format is required for visualization!"

#         self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
#         self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
#         assert (
#             self.pixel_mean.shape == self.pixel_std.shape
#         ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
#         # @yujheli: you may need to build your discriminator here

#         self.dis_type = dis_type
#         self.D_img = None
#         self.D_ins = None
#         # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels['res4']) # Need to know the channel
#         # self.D_proposal = None
#         # self.D_img = None
#         self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) # Need to know the channel
#         # self.D_ins = FCDiscriminator_proposal(self.backbone._out_feature_channels[self.dis_type])
#         # self.D_proposal = FCDiscriminator_proposal(self.backbone._out_feature_channels[self.dis_type])
#         # self.bceLoss_func = nn.BCEWithLogitsLoss()
#     def build_discriminator(self):
#         self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel
#     # def build_discriminator_ins(self):
#     #     self.D_ins = FCDiscriminator_proposal(self.backbone._out_feature_channels[self.dis_type]).to(self.device)
#     @classmethod
#     def from_config(cls, cfg):
#         backbone = build_backbone(cfg)
#         return {
#             "backbone": backbone,
#             "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
#             "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
#             "input_format": cfg.INPUT.FORMAT,
#             "vis_period": cfg.VIS_PERIOD,
#             "pixel_mean": cfg.MODEL.PIXEL_MEAN,
#             "pixel_std": cfg.MODEL.PIXEL_STD,
#             "dis_type": cfg.SEMISUPNET.DIS_TYPE,
#             # "dis_loss_ratio": cfg.xxx,
#         }

#     def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
#         """Generate 2D gaussian kernel.

#         Args:
#             radius (int): Radius of gaussian kernel.
#             sigma (int): Sigma of gaussian function. Default: 1.
#             dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
#             device (str): Device of gaussian tensor. Default: 'cpu'.

#         Returns:
#             h (Tensor): Gaussian kernel with a
#                 ``(2 * radius + 1) * (2 * radius + 1)`` shape.
#         """
#         x = torch.arange(
#             -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
#         y = torch.arange(
#             -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

#         # h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()
#         h = (-(x * x + y * y) / (5 * sigma * sigma)).exp()
#         # h[h < torch.finfo(h.dtype).eps * h.max()] = 0
#         return h

#     def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         Normalize, pad and batch the input images.
#         """
#         images = [x["image"].to(self.device) for x in batched_inputs]
#         images = [(x - self.pixel_mean) / self.pixel_std for x in images]
#         images = ImageList.from_tensors(images, self.backbone.size_divisibility)

#         images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
#         images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
#         images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

#         return images, images_t

#     @staticmethod
#     def _postprocess(instances, batched_inputs, image_sizes):
#         """
#         Rescale the output instances to the target size.
#         """
#         # note: private function; subject to changes
#         processed_results = []
#         for results_per_image, input_per_image, image_size in zip(
#             instances, batched_inputs, image_sizes
#         ):
#             # height = image_size[0]
#             # width = image_size[1]
#             height = input_per_image.get("height", image_size[0])
#             width = input_per_image.get("width", image_size[1])
#             r = detector_postprocess(results_per_image, height, width)
#             processed_results.append({"instances": r})
#         return processed_results

#     def inference_demo(self, batched_inputs, detected_instances=None, do_postprocess=True):
#         """
#         Run inference on the given inputs.

#         Args:
#             batched_inputs (tensor): same as in :meth:`forward`
#             detected_instances (None or list[Instances]): if not None, it
#                 contains an `Instances` object per image. The `Instances`
#                 object contains "pred_boxes" and "pred_classes" which are
#                 known boxes in the image.
#                 The inference will then skip the detection of bounding boxes,
#                 and only predict other per-ROI outputs.
#             do_postprocess (bool): whether to apply post-processing on the outputs.

#         Returns:
#             When do_postprocess=True, same as in :meth:`forward`.
#             Otherwise, a list[Instances] containing raw network outputs.
#         """
#         assert not self.training

#         images = self.preprocess_image([batched_inputs])
#         features = self.backbone(images.tensor)

#         if detected_instances is None:
#             if self.proposal_generator:
#                 proposals, _ = self.proposal_generator(images, features, None)
#             else:
#                 assert "proposals" in batched_inputs[0]
#                 proposals = [x["proposals"].to(self.device) for x in batched_inputs]

#             results, _ = self.roi_heads(images, features, proposals, None)
#         else:
#             detected_instances = [x.to(self.device) for x in detected_instances]
#             results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

#         if do_postprocess:
#             return self._postprocess(results, [batched_inputs], images.image_sizes)
#         # else:
#         return results

#     def inference_demo_wbf(self, batched_inputs, detected_instances=None, do_postprocess=True):
#         """
#         Run inference on the given inputs.

#         Args:
#             batched_inputs (tensor): same as in :meth:`forward`
#             detected_instances (None or list[Instances]): if not None, it
#                 contains an `Instances` object per image. The `Instances`
#                 object contains "pred_boxes" and "pred_classes" which are
#                 known boxes in the image.
#                 The inference will then skip the detection of bounding boxes,
#                 and only predict other per-ROI outputs.
#             do_postprocess (bool): whether to apply post-processing on the outputs.

#         Returns:
#             When do_postprocess=True, same as in :meth:`forward`.
#             Otherwise, a list[Instances] containing raw network outputs.
#         """
#         assert not self.training

#         images = self.preprocess_image(batched_inputs)
#         features = self.backbone(images.tensor)

#         if detected_instances is None:
#             if self.proposal_generator:
#                 proposals, _ = self.proposal_generator(images, features, None)  #num 1000 predictions
#             else:
#                 assert "proposals" in batched_inputs[0]
#                 proposals = [x["proposals"].to(self.device) for x in batched_inputs]

#             results, _ = self.roi_heads(images, features, proposals, None)
#         else:
#             detected_instances = [x.to(self.device) for x in detected_instances]
#             results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

#         if do_postprocess:
#             return self._postprocess(results, batched_inputs, images.image_sizes)
#         # else:
#         return results

#     def forward(
#         self, batched_inputs,gt_proposals='', branch="supervised", given_proposals=None, val_mode=False
#     ):
#         """
#         Args:
#             batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
#                 Each item in the list contains the inputs for one image.
#                 For now, each item in the list is a dict that contains:

#                 * image: Tensor, image in (C, H, W) format.
#                 * instances (optional): groundtruth :class:`Instances`
#                 * proposals (optional): :class:`Instances`, precomputed proposals.

#                 Other information that's included in the original dicts, such as:

#                 * "height", "width" (int): the output resolution of the model, used in inference.
#                   See :meth:`postprocess` for details.

#         Returns:
#             list[dict]:
#                 Each dict is the output for one input image.
#                 The dict contains one key "instances" whose value is a :class:`Instances`.
#                 The :class:`Instances` object has the following keys:
#                 "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
#         """
#         if self.D_img == None:
#             self.build_discriminator()
#         # if self.D_ins == None:
#         #     self.build_discriminator_ins()
#         # if self.D_proposal == None:
#         #     self.build_discriminator_proposal()
#         if (not self.training) and (not val_mode):  # only conduct when testing mode
#             return self.inference(batched_inputs)
#             #todo:only use when wanna visible results
#             # return self.inference_demo(batched_inputs)
#             # return self.inference_demo_wbf(batched_inputs)
#         source_label = 0
#         target_label = 1
#         if branch == "domain":
#             # self.D_img.train()
#             # source_label = 0
#             # target_label = 1
#             # images = self.preprocess_image(batched_inputs)
#             images_s, images_t = self.preprocess_image_train(batched_inputs)
#             if "instances" in batched_inputs[0]:
#                 gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
#             else:
#                 gt_instances = None
#             if "instances_unlabeled" in batched_inputs[0]:
#                 fake_instances = [x["instances_unlabeled"].to(self.device) for x in batched_inputs]
#             else:
#                 fake_instances = None
#             features = self.backbone(images_s.tensor)

#             # import pdb
#             # pdb.set_trace()
# #todo:原来的东西的样子
#             features_s = grad_reverse(features[self.dis_type])
#             D_img_out_s = self.D_img(features_s)
#             loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))
# #改过的
#             # features_ins_s = self.roi_heads.box_pooler([features[self.dis_type]], [x.gt_boxes for x in gt_instances])
#             # D_ins_out_s = self.D_ins(grad_reverse(features_ins_s)).permute(1,2,3,0)
#             # # D_ins_out_s = torch.mul(D_ins_out_s,gaussian2D(3, 7 / 6, device=self.device))
#             # loss_D_ins_s = F.binary_cross_entropy_with_logits(D_ins_out_s,
#             #                                                   torch.FloatTensor(D_ins_out_s.data.size()).fill_(
#             #                                                       source_label).to(self.device))

#             # features_proposal_s = [features[self.dis_type]]is_quantized = {bool} False
#             # # features_proposal_s = grad_reverse(features_proposal_s)
#             # features_proposal_s = self.roi_heads.box_pooler(features_proposal_s, [x.gt_boxes for x in gt_instances])
#             # features_proposal_s = grad_reverse(features_proposal_s)
#             # D_proposal_s = self.D_proposal(features_proposal_s)
#             # #
#             # loss_D_proposal_s = F.binary_cross_entropy_with_logits(D_proposal_s,
#             #                                                        torch.FloatTensor(D_proposal_s.data.size()).fill_(
#             #                                                            source_label).to(self.device))


# #dis tgt feature w/o grl
#             features_target = self.backbone(images_t.tensor)

#             # features_t = grad_reverse(features_target[self.dis_type])
#             # #todo:看看加grl会怎么样，会很差，2000+无了
#             # #不加grl了
#             D_img_out_t = self.D_img(grad_reverse(features_target[self.dis_type]))
#             #
#             # # features_t = grad_reverse(features_t['p2'])
#             # D_img_out_t = self.D_img(features_t)
#             loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))
# #TODO:ins_compare
# #todo：
#             # features_ins_t = self.roi_heads.box_pooler([features_target[self.dis_type]], [x.gt_boxes for x in fake_instances])
#             # D_ins_out_t = self.D_ins(features_ins_t)
#             # if len(fake_instances[0]) != 0 and len(fake_instances[1]) != 0:
#         #     features_ins_t = self.roi_heads.box_pooler([features[self.dis_type]], [x.pred_boxes for x in fake_instances])
#         # # D_ins_out_t = self.D_ins(grad_reverse(features_ins_t))
#         #     D_ins_out_t = self.D_ins(features_ins_t).permute(1,2,3,0)
#         #
#         #     loss_D_ins_t = F.binary_cross_entropy_with_logits(D_ins_out_t,
#         #                                                   torch.FloatTensor(D_ins_out_t.data.size()).fill_(
#         #                                                       target_label).to(self.device),weight=torch.cat([x.scores for x in fake_instances],dim=0))


#             # features_proposal_t = [features_target[self.dis_type]]
#             # # features_proposal_s = grad_reverse(features_proposal_s)
#             # features_proposal_t = self.roi_heads.box_pooler(features_proposal_t, [x.gt_boxes for x in gt_instances])
#             # features_proposal_t = grad_reverse(features_proposal_t)
#             # D_proposal_t = self.D_proposal(features_proposal_t)
#             # #
#             # loss_D_proposal_t = F.binary_cross_entropy_with_logits(D_proposal_t,
#             #                                                        torch.FloatTensor(D_proposal_t.data.size()).fill_(
#             #                                                            target_label).to(self.device))

#             # import pdb
#             # pdb.set_trace()

#             losses = {}
#             losses["loss_D_img_s"] = loss_D_img_s
#             # losses["loss_D_ins_s"] = loss_D_ins_s
#             # losses["loss_D_proposal_s"] = loss_D_proposal_s
#             losses["loss_D_img_t"] = loss_D_img_t
#             # losses["loss_D_ins_t"] = loss_D_ins_t
#             # losses["loss_D_proposal_t"] = loss_D_proposal_t
#             return losses, [], [], None

#         # self.D_img.eval()
#         images = self.preprocess_image(batched_inputs)

#         if "instances" in batched_inputs[0]:
#             gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
#         else:
#             gt_instances = None

#         features = self.backbone(images.tensor)

#         # TODO: remove the usage of if else here. This needs to be re-organized
#         if branch == "supervised":
#             features_s = grad_reverse(features[self.dis_type])
#             # features_s = features[self.dis_type]
#             D_img_out_s = self.D_img(features_s)
#             loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))
#             # ins_num = [len(ins) for ins in gt_instances] #count num to affect weight
#             # with torch.no_grad():
#             #     loss_weight = torch.exp(loss_D_img_s-torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label))

#             # features_ins_s = self.roi_heads.box_pooler([features_s], [x.gt_boxes for x in gt_instances])
#             # D_ins_out_s = self.D_ins(features_ins_s)
#             # loss_D_ins_s = F.binary_cross_entropy_with_logits(D_ins_out_s, torch.FloatTensor(D_ins_out_s.data.size()).fill_(source_label).to(self.device))

#             # print(D_ins_out_s.mean(),'src')

#             # features_proposal_s = [features[self.dis_type]]
#             # # features_proposal_s = grad_reverse(features_proposal_s)
#             # features_proposal_s = self.roi_heads.box_pooler(features_proposal_s,[x.gt_boxes for x in gt_instances])
#             # features_proposal_s = grad_reverse(features_proposal_s)
#             # D_proposal_s = self.D_proposal(features_proposal_s)
#             # #
#             # loss_D_proposal_s = F.binary_cross_entropy_with_logits(D_proposal_s,
#             #                                                        torch.FloatTensor(D_proposal_s.data.size()).fill_(
#             #                                                            source_label).to(self.device))

#             # Region proposal network
#             proposals_rpn, proposal_losses = self.proposal_generator(
#                 images, features, gt_instances
#             )

#             # roi_head lower branch
#             _, detector_losses = self.roi_heads(
#                 images,
#                 features,
#                 proposals_rpn,
#                 compute_loss=True,
#                 targets=gt_instances,
#                 branch=branch,
#             )

#             # visualization
#             if self.vis_period > 0:
#                 storage = get_event_storage()
#                 if storage.iter % self.vis_period == 0:
#                     self.visualize_training(batched_inputs, proposals_rpn, branch)

#             losses = {}
#             losses.update(detector_losses)
#             losses.update(proposal_losses)
#             # losses["loss_D_img_s"] = (loss_D_img_s)*0.001
#             losses["loss_D_img_s"] = (loss_D_img_s)*0.01
#             # losses["loss_D_ins_s"] = loss_D_ins_s * 0.001
#             # losses["loss_D_proposal_s"] = (loss_D_proposal_s)*0.001

#             return losses, [], [], None
#         elif branch == "supervised_compare":
#             # features_t = grad_reverse(features[self.dis_type])
#             # x = features[self.dis_type]

#             D_img_out_t = self.D_img(grad_reverse(features[self.dis_type]))
#             # print(D_img_out_t.mean(),'tgt')
#             loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t,
#                                                               torch.FloatTensor(D_img_out_t.data.size()).fill_(
#                                                                   target_label).to(self.device))
#             # features_proposal_t = [features[self.dis_type]]
#             # # features_proposal_s = grad_reverse(features_proposal_s)
#             # features_proposal_t = self.roi_heads.box_pooler(features_proposal_t, [x.gt_boxes for x in gt_instances])
#             # features_proposal_t = grad_reverse(features_proposal_t)
#             # D_proposal_t = self.D_proposal(features_proposal_t)
#             # #
#             # loss_D_proposal_t = F.binary_cross_entropy_with_logits(D_proposal_t,
#             #                                                        torch.FloatTensor(D_proposal_t.data.size()).fill_(
#             #                                                            target_label).to(self.device))

#             # Region proposal network
#             proposals_rpn, proposal_losses = self.proposal_generator(
#                 images, features, gt_instances
#             )

#             # roi_head lower branch
#             _, detector_losses = self.roi_heads(
#                 images,
#                 features,
#                 proposals_rpn,
#                 compute_loss=True,
#                 targets=gt_instances,
#                 branch=branch,
#             )

#             # visualization
#             if self.vis_period > 0:
#                 storage = get_event_storage()
#                 if storage.iter % self.vis_period == 0:
#                     self.visualize_training(batched_inputs, proposals_rpn, branch)

#             losses = {}
#             losses.update(detector_losses)
#             losses.update(proposal_losses)
#             losses["loss_D_img_t"] = loss_D_img_t * 0.01
#             # losses["loss_D_proposal_t"] = loss_D_proposal_t * 0.001
#             return losses, [], [], None

#         elif branch == "supervised_target":

#             # features_t = grad_reverse(features_t[self.dis_type])
#             # D_img_out_t = self.D_img(features_t)
#             # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))


#             # Region proposal network
#             proposals_rpn, proposal_losses = self.proposal_generator(
#                 images, features, gt_instances
#             )

#             # roi_head lower branch
#             _, detector_losses = self.roi_heads(
#                 images,
#                 features,
#                 proposals_rpn,
#                 compute_loss=True,
#                 targets=gt_instances,
#                 branch=branch,
#             )

#             # visualization
#             if self.vis_period > 0:
#                 storage = get_event_storage()
#                 if storage.iter % self.vis_period == 0:
#                     self.visualize_training(batched_inputs, proposals_rpn, branch)

#             losses = {}
#             losses.update(detector_losses)
#             losses.update(proposal_losses)
#             # losses["loss_D_img_t"] = loss_D_img_t*0.001
#             # losses["loss_D_img_s"] = loss_D_img_s*0.001
#             return losses, [], [], None
#         elif branch == "compare_data_weak":
#             """
#             unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
#             """
#             # Region proposal network
#             # proposals_rpn, _ = self.proposal_generator(
#             #     images, features, None, compute_loss=False
#             # )
#             # roi_head lower branch (keep this for further production)
#             # notice that we do not use any target in ROI head to do inference!

#             proposals_roih, ROI_predictions = self.roi_heads(
#                 images,
#                 features,
#                 # proposals_rpn,
#                 gt_proposals,
#                 targets=None,
#                 compute_loss=False,
#                 branch=branch,
#             )

#             # if self.vis_period > 0:
#             #     storage = get_event_storage()
#             #     if storage.iter % self.vis_period == 0:
#             #         self.visualize_training(batched_inputs, proposals_rpn, branch)

#             return proposals_roih, ROI_predictions

#         elif branch == "unsup_data_weak":
#             """
#             unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
#             """
#             # Region proposal network
#             proposals_rpn, _ = self.proposal_generator(
#                 images, features, None, compute_loss=False
#             )

#             # roi_head lower branch (keep this for further production)
#             # notice that we do not use any target in ROI head to do inference!
#             proposals_roih, ROI_predictions = self.roi_heads(
#                 images,
#                 features,
#                 proposals_rpn,
#                 targets=None,
#                 compute_loss=False,
#                 branch=branch,
#             )

#             # if self.vis_period > 0:
#             #     storage = get_event_storage()
#             #     if storage.iter % self.vis_period == 0:
#             #         self.visualize_training(batched_inputs, proposals_rpn, branch)

#             return {}, proposals_rpn, proposals_roih, ROI_predictions
#         elif branch == "unsup_data_strong":
#             raise NotImplementedError()
#         elif branch == "val_loss":
#             raise NotImplementedError()

#     def visualize_training(self, batched_inputs, proposals, branch=""):
#         """
#         This function different from the original one:
#         - it adds "branch" to the `vis_name`.

#         A function used to visualize images and proposals. It shows ground truth
#         bounding boxes on the original image and up to 20 predicted object
#         proposals on the original image. Users can implement different
#         visualization functions for different models.

#         Args:
#             batched_inputs (list): a list that contains input to the model.
#             proposals (list): a list that contains predicted proposals. Both
#                 batched_inputs and proposals should have the same length.
#         """
#         from detectron2.utils.visualizer import Visualizer

#         storage = get_event_storage()
#         max_vis_prop = 20

#         for input, prop in zip(batched_inputs, proposals):
#             img = input["image"]
#             img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
#             v_gt = Visualizer(img, None)
#             v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
#             anno_img = v_gt.get_image()
#             box_size = min(len(prop.proposal_boxes), max_vis_prop)
#             v_pred = Visualizer(img, None)
#             v_pred = v_pred.overlay_instances(
#                 boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
#             )
#             prop_img = v_pred.get_image()
#             vis_img = np.concatenate((anno_img, prop_img), axis=1)
#             vis_img = vis_img.transpose(2, 0, 1)
#             vis_name = (
#                 "Left: GT bounding boxes "
#                 + branch
#                 + ";  Right: Predicted proposals "
#                 + branch
#             )
#             storage.put_image(vis_name, vis_img)
#             break  # only visualize one image in a batch



# @META_ARCH_REGISTRY.register()
# class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
#     def forward(
#         self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
#     ):
#         if (not self.training) and (not val_mode):
#             return self.inference(batched_inputs)

#         images = self.preprocess_image(batched_inputs)

#         if "instances" in batched_inputs[0]:
#             gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
#         else:
#             gt_instances = None

#         features = self.backbone(images.tensor)

#         if branch == "supervised":
#             # Region proposal network
#             proposals_rpn, proposal_losses = self.proposal_generator(
#                 images, features, gt_instances
#             )

#             # # roi_head lower branch
#             _, detector_losses = self.roi_heads(
#                 images, features, proposals_rpn, gt_instances, branch=branch
#             )

#             losses = {}
#             losses.update(detector_losses)
#             losses.update(proposal_losses)
#             return losses, [], [], None

#         elif branch == "unsup_data_weak":
#             # Region proposal network
#             proposals_rpn, _ = self.proposal_generator(
#                 images, features, None, compute_loss=False
#             )

#             # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
#             proposals_roih, ROI_predictions = self.roi_heads(
#                 images,
#                 features,
#                 proposals_rpn,
#                 targets=None,
#                 compute_loss=False,
#                 branch=branch,
#             )

#             return {}, proposals_rpn, proposals_roih, ROI_predictions

#         elif branch == "val_loss":

#             # Region proposal network
#             proposals_rpn, proposal_losses = self.proposal_generator(
#                 images, features, gt_instances, compute_val_loss=True
#             )

#             # roi_head lower branch
#             _, detector_losses = self.roi_heads(
#                 images,
#                 features,
#                 proposals_rpn,
#                 gt_instances,
#                 branch=branch,
#                 compute_val_loss=True,
#             )

#             losses = {}
#             losses.update(detector_losses)
#             losses.update(proposal_losses)
#             return losses, [], [], None


