""" The Code is under Tencent Youtu Public Rule
"""

from __future__ import print_function
import argparse
import logging
import math
import os
from re import escape
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from functools import partial
from .base_trainer import Trainer

loss_dict = {"cross_entropy": F.cross_entropy}

def build(cfg):
    loss_cfg = deepcopy(cfg)
    loss_type = loss_cfg.pop("type")
    return partial(loss_dict[loss_type], **loss_cfg)

class SoftSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SoftSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, max_probs, labels=None, mask=None, reduction="mean", select_matrix=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None and select_matrix is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            max_probs = max_probs.contiguous().view(-1, 1)
            score_mask = torch.matmul(max_probs, max_probs.T)
            mask = mask.mul(score_mask) * select_matrix

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            #max_probs = max_probs.reshape((batch_size,1))
            max_probs = max_probs.contiguous().view(-1, 1)
            score_mask = torch.matmul(max_probs,max_probs.T)
            mask = mask.mul(score_mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        if reduction == "mean":
            loss = loss.mean()

        return loss

class CoMatchCCSSL(Trainer):
    """ Comatch CCSSL trainer based on CoMatch
    """
    def __init__(self, batch_size, num_class,device, **kwargs):
        super().__init__()
        # prepare self params
        self.device = device
        self.batch_size = batch_size
        self.temperature = 0.2
        self.T = 0.2
        self.num_classes = num_class
        self.mu = 7
        self.low_dim = 64
        self.init_params()
        # da setup use prob_list because the same name in official code
        self.da_len = 32
        self.prob_list = []
        
        #build loss function for supervised branch and contrastive branch
        self.loss_x = build(dict(type='cross_entropy', reduction='mean'))
        self.loss_contrast = SoftSupConLoss(temperature=self.temperature)
        self.init_memory_smoothed_data()

    # initialize memory smoothed data
    def init_memory_smoothed_data(self):
        self.queue_batch =5
        self.queue_size = self.queue_batch * (self.mu + 1) * self.batch_size
        self.queue_feats = torch.zeros(self.queue_size, self.low_dim).to(self.device)
        self.queue_probs = torch.zeros(self.queue_size, self.num_classes).to(self.device)
        self.queue_ptr = 0

    # init params
    def init_params(self):
        self.temperature = self.T
        self.alpha = 0.9
        self.threshold = 0.95
        self.contrast_threshold = 0.8
        self.lambda_c = 5
        self.lambda_u = 1
        self.contrast_with_thresh = 0.9
        self.lambda_supcon = 0.1

    def get_task_specific_info(self,task_specific_info):
        try:
            self.queue_feats = task_specific_info['queue_feats']
            self.queue_probs = task_specific_info['queue_probs']
            self.queue_ptr = task_specific_info['queue_ptr']
        except KeyError:
            pass
    
    def make_inputs(self,inputs_x,inputs_u):
        inputs_x = inputs_x[0]
        inputs_u_w, inputs_u_s0, inputs_u_s1 = inputs_u
        batch_size = inputs_x.shape[0]
        batch_size_u = inputs_u_w.shape[0]
        inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s0, inputs_u_s1],
                           dim=0).to(self.device)
        return batch_size,batch_size_u,inputs

    # Distribution Alignment mentioned in paper
    def DA(self,logits_u_w):
        # pseudo label with weak aug
        probs = torch.softmax(logits_u_w, dim=1)
        self.prob_list.append(probs.mean(0))
        if len(self.prob_list) > self.da_len:
            self.prob_list.pop(0)
        prob_avg = torch.stack(self.prob_list, dim=0).mean(0)
        probs = probs / prob_avg
        probs = probs / probs.sum(dim=1, keepdim=True)

        probs_orig = probs.clone()
        return probs,probs_orig
    
    def memory_smoothing(self,feats_u_w):
        A = torch.exp(
            torch.mm(feats_u_w, self.queue_feats.t()) /
            self.temperature)
        A = A / A.sum(1, keepdim=True)
        probs = self.alpha * probs + (1 - self.alpha) * torch.mm(
            A, self.queue_probs)
        return probs

    def get_lbs_and_masks(self,probs):
        scores, lbs_u_guess = torch.max(probs, dim=1)
        mask = scores.ge(self.threshold).float()
        return lbs_u_guess,mask

    def update_mmbank(self,
                     feats_u_w,
                     feats_x,
                     targets_x,
                     probs_orig,
                     batch_size,
                     batch_size_u):
        feats_w = torch.cat([feats_u_w, feats_x], dim=0)
        onehot = torch.zeros(batch_size,
                                self.num_classes).to(self.device).scatter(
                                    1, targets_x.view(-1, 1), 1)
        probs_w = torch.cat([probs_orig, onehot], dim=0)

        n = batch_size + batch_size_u
        self.queue_feats[self.queue_ptr:self.queue_ptr + n, :] = feats_w
        self.queue_probs[self.queue_ptr:self.queue_ptr + n, :] = probs_w
        self.queue_ptr = (self.queue_ptr + n) % self.queue_size

    def pseudo_lb_graph(self,probs):
        Q = torch.mm(probs, probs.t())
        Q.fill_diagonal_(1)
        pos_mask = (Q >= self.contrast_threshold).float()

        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)
        return Q
    
    def contrast_left_out(self,max_probs):
        contrast_mask = max_probs.ge(self.contrast_with_thresh).float()
        contrast_mask2 = torch.clone(contrast_mask)
        contrast_mask2[contrast_mask == 0] = -1
        select_elements = torch.eq(
            contrast_mask2.reshape([-1, 1]), contrast_mask.reshape([-1, 1]).T).float()
        select_elements += torch.eye(contrast_mask.shape[0]).to(self.device)
        select_elements[select_elements > 1] = 1
        select_matrix = torch.ones(
            contrast_mask.shape[0]).to(self.device) * select_elements
        return select_matrix

    def compute_loss(self,
                     data_x,
                     data_u,
                     model,
                     optimizer,
                     epoch,
                     iter,
                     ema_model=None,
                     task_specific_info=None,
                     **kwargs):

        self.get_task_specific_info(task_specific_info)

        inputs_x, targets_x = data_x
        inputs_u, targets_u = data_u

        #prepare inputs
        batch_size,batch_size_u,inputs = self.make_inputs(inputs_x,inputs_u)

        targets_u,targets_x = targets_u.to(self.device),targets_x.to(self.device)

        # inference logits and features 
        logits, features,code = model(inputs)
        logits_x = logits[:batch_size]
        feats_x = features[:batch_size]
        logits_u_w, logits_u_s0, _ = torch.split(logits[batch_size:], batch_size_u)
        feats_u_w, feats_u_s0, feats_u_s1 = torch.split(features[batch_size:], batch_size_u)

        # supervision loss
        # loss_x = self.loss_x(logits_x, targets_x)

        # other losses
        with torch.no_grad():
            logits_u_w,feats_x,feats_u_w= logits_u_w.detach(),feats_x.detach(),feats_u_w.detach()
            probs,probs_orig = self.DA(logits_u_w)

            # memory-smoothing using feature similairty
            if epoch > 0 or iter > self.queue_batch:
                probs = self.memory_smoothing(feats_u_w)

            # get pseudo label and mask
            # note here the label is soft, hard label is for acc calculation
            lbs_u_guess,mask = self.get_lbs_and_masks(probs)

            # update memory bank
            self.update_mmbank(feats_u_w,
                               feats_x,
                               targets_x,
                               probs_orig,
                               batch_size,
                               batch_size_u)

        # embedding similarity
        sim = torch.exp(torch.mm(feats_u_s0, feats_u_s1.t())/self.temperature)
        sim_probs = sim / sim.sum(1, keepdim=True)

        # pseudo-label graph with self-loop for contrustive similarity
        Q =self.pseudo_lb_graph(probs)

        # contrastive loss
        loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1)

        # unsupervised classification loss, cross entropy but author self implemented
        #loss_u = -torch.sum((F.log_softmax(logits_u_s0, dim=1) * probs), dim=1) * mask
        loss_contrast = loss_contrast.mean()
        # for supervised contrastive
        #features = torch.cat([feats_u_s0.unsqueeze(1), feats_u_s1.unsqueeze(1)], dim=1)
        # max_probs, p_targets_u = torch.max(probs, dim=-1)
        # if p_targets_u.shape[0] != 0:
        #     select_matrix = None
        #     Lcontrast = self.loss_contrast(features, max_probs, p_targets_u)
        # else:
        #     Lcontrast = sum(features.view(-1, 1)) * 0
        #loss = loss_x + self.lambda_u * loss_u + self.lambda_c * loss_contrast + self.lambda_supcon * Lcontrast

        # # calculate pseudo label acc
        # right_labels = (lbs_u_guess == targets_u).float() * mask
        # pseudo_label_acc = right_labels.sum() / max(mask.sum(), 1.0)

        # # modify task_specific_info in place
        # task_specific_info['queue_feats'] = self.queue_feats
        # task_specific_info['queue_probs'] = self.queue_probs
        # task_specific_info['queue_ptr'] = self.queue_ptr

        # # output loss
        # loss_dict = {
        #     "loss": loss,
        #     "loss_x": loss_x,
        #     "loss_u": loss_u,
        #     "loss_c": loss_contrast,
        #     "Lcontrast": Lcontrast,
        #     "mask_prob": mask.mean(),
        #     "pseudo_acc": pseudo_label_acc,
        # }
        return self.lambda_c * loss_contrast
