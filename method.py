
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from loguru import logger
import torch.nn.functional as F
import model.network as network
import model.loss as loss
import itertools
from model.linearaverage import LinearAverage
from ccssl.comatch_ccssl import CoMatchCCSSL
from ccssl.soft_supconloss import *
from model.model_loader import load_model
from evaluate import mean_average_precision
from model.labelmodel import *
from torch.nn import Parameter
from torch.autograd import Variable
from utils import *
from model.gss import *
import random
from PIL import ImageFilter
# from apex import amp
torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

transform1 = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.ToTensor(), 
    ])

class BaseClassificationLoss(nn.Module):
    def __init__(self):
        super(BaseClassificationLoss, self).__init__()
        self.losses = {}

    def forward(self, logits, code_logits, labels, onehot=True):
        raise NotImplementedError

def get_imbalance_mask(sigmoid_logits, labels, nclass, threshold=0.7, imbalance_scale=-1):
    if imbalance_scale == -1:
        imbalance_scale = 1 / nclass

    mask = torch.ones_like(sigmoid_logits) * imbalance_scale

    # wan to activate the output
    mask[labels == 1] = 1

    # if predicted wrong, and not the same as labels, minimize it
    correct = (sigmoid_logits >= threshold) == (labels == 1)
    mask[~correct] = 1

    multiclass_acc = correct.float().mean()

    # the rest maintain "imbalance_scale"
    return mask, multiclass_acc

class OrthoHashLoss(BaseClassificationLoss):
    def __init__(self,
                 ce=1,
                 s=8,
                 m=0.2,
                 m_type='cos',  # cos/arc
                 multiclass=False,
                 quan=0,
                 quan_type='cs',
                 multiclass_loss='label_smoothing',
                 **kwargs):
        super(OrthoHashLoss, self).__init__()
        self.ce = ce
        self.s = s
        self.m = m
        self.m_type = m_type
        self.multiclass = multiclass

        self.quan = quan
        self.quan_type = quan_type
        self.multiclass_loss = multiclass_loss
        assert multiclass_loss in ['bce', 'imbalance', 'label_smoothing']

    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits

    def forward(self, logits, code_logits, labels, onehot=True):
        if self.multiclass:
            if not onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()

            margin_logits = self.compute_margin_logits(logits, labels)

            if self.multiclass_loss in ['bce', 'imbalance']:
                loss_ce = F.binary_cross_entropy_with_logits(margin_logits, labels, reduction='none')
                if self.multiclass_loss == 'imbalance':
                    imbalance_mask, multiclass_acc = get_imbalance_mask(torch.sigmoid(margin_logits), labels,
                                                                        labels.size(1))
                    loss_ce = loss_ce * imbalance_mask
                    loss_ce = loss_ce.sum() / (imbalance_mask.sum() + 1e-7)
                    self.losses['multiclass_acc'] = multiclass_acc
                else:
                    loss_ce = loss_ce.mean()
            elif self.multiclass_loss in ['label_smoothing']:
                log_logits = F.log_softmax(margin_logits, dim=1)
                labels_scaled = labels / labels.sum(dim=1, keepdim=True)
                loss_ce = - (labels_scaled * log_logits).sum(dim=1)
                loss_ce = loss_ce.mean()
            else:
                raise NotImplementedError(f'unknown method: {self.multiclass_loss}')
        else:
            if onehot:
                labels = labels.argmax(1)
            margin_logits = self.compute_margin_logits(logits, labels)
            loss_ce = F.cross_entropy(margin_logits, labels)
            loss_ce_batch = F.cross_entropy(margin_logits, labels, reduction='none')


        if self.quan != 0:
            if self.quan_type == 'cs':
                quantization = (1. - F.cosine_similarity(code_logits, code_logits.detach().sign(), dim=1))
            elif self.quan_type == 'l1':
                quantization = torch.abs(code_logits - code_logits.detach().sign())
            else:  # l2
                quantization = torch.pow(code_logits - code_logits.detach().sign(), 2)
            quantization_batch = quantization
            quantization = quantization.mean()
        else:
            quantization_batch = torch.zeros_like(loss_ce_batch)
            quantization = torch.tensor(0.).to(logits.device)

        self.losses['ce'] = loss_ce
        self.losses['quan'] = quantization
        loss = self.ce * loss_ce + self.quan * quantization
        loss_batch = self.ce * loss_ce_batch + self.quan * quantization_batch
        return loss, loss_batch


def entropy(p):
    p = F.softmax(p)
    return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))

def hinge(input, margin=0.2):
    return torch.clamp(input, min=margin)

def entropy_margin(p, value, margin=0.2, weight=None):
    p = F.softmax(p)
    return -torch.mean(hinge(torch.abs(-torch.sum(p * torch.log(p+1e-5), 1)-value), margin))


def _da_pseudo_label(logits_u_w):
    prob_list = []
    with torch.no_grad():
        probs = torch.softmax(logits_u_w, dim=1)

        prob_list.append(probs.mean(0))
        if len(prob_list) > 5:
            prob_list.pop(0)
        prob_avg = torch.stack(prob_list, dim=0).mean(0)
        probs = probs / prob_avg
        probs = probs / probs.sum(dim=1, keepdim=True)
        probs = probs.detach()
    return probs

def contrast_left_out(max_probs,contrast_with_thresh,device):
    contrast_mask = max_probs.ge(contrast_with_thresh).float()
    contrast_mask2 = torch.clone(contrast_mask)
    contrast_mask2[contrast_mask == 0] = -1
    select_elements = torch.eq(
        contrast_mask2.reshape([-1, 1]), contrast_mask.reshape([-1, 1]).T).float()
    select_elements += torch.eye(contrast_mask.shape[0]).to(device)
    select_elements[select_elements > 1] = 1
    select_matrix = torch.ones(
        contrast_mask.shape[0]).to(device) * select_elements
    return select_matrix

def train(train_s_dataloader,
          train_t_dataloader,
          query_dataloader,
          retrieval_dataloader,
          code_length,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_class,
          evaluate_interval,
          tag,
          batch_size,
          source_class,
          ):

    model = load_model(arch, code_length,num_class,source_class)
    # logger.info(model)
    model = nn.DataParallel(model,device_ids=[0,1])
    model.to(device)
    ad_net = network.AdversarialNetwork(4096*code_length, 1024)
    ad_net.to(device)
    ad_net = nn.DataParallel(ad_net,device_ids=[0,1])
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    if isinstance(ad_net,torch.nn.DataParallel):
        ad_net = ad_net.module
    parameter_list = model.get_parameters() + ad_net.get_parameters()
    optimizer = optim.SGD(parameter_list, lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion_new = OrthoHashLoss()
    model = nn.DataParallel(model,device_ids=[0,1])
    ad_net = nn.DataParallel(ad_net,device_ids=[0,1])
    ccssl = CoMatchCCSSL(batch_size, num_class,device)
    temp = 0.05

    lemniscate = LinearAverage(4096, 3609, temp, 0.0).to(device)
    GSS_ent = GSS_loss(alpha = 5, beta = 0.09)
    GSS_conf = GSS_loss(alpha = 10, beta = 0.09)
    loss_contrast = SoftSupConLoss(temperature=0.1)

    model.train()
    
    for epoch in range(max_iter):

        for batch_idx, ((data_s, _, target_s, index), (data_t, data_t_aug,target_t, index_t)) in\
             enumerate(zip(train_s_dataloader, train_t_dataloader)):
            start = time.time()
            
            batch_size = data_t.shape[0]
            data_s = data_s.to(device)
            target_s = target_s.to(device)
            data_t = data_t.to(device)
            data_t_aug = data_t_aug.to(device)
            index_t = index_t.to(device)

            optimizer.zero_grad()
            logit_s, f_s, feature_s, code_s = model(data_s)
            logit_t, f_t, feature_t, code_t = model(data_t)
            
            loss_s, _ = criterion_new(logit_s, code_s, target_s)
            
            contrast_with_thresh = 0.8
            probs_u_w = _da_pseudo_label(f_t.detach())
             # pseudo label and scores for u_w
            max_probs, p_targets_u = torch.max(probs_u_w, dim=-1)
            labels = p_targets_u
            feats = torch.cat([code_t.unsqueeze(1), \
                                code_t.unsqueeze(1)], dim=1)
            with torch.no_grad():
                select_matrix = contrast_left_out(max_probs,contrast_with_thresh,device)
            Lcontrast = loss_contrast(feats, max_probs, labels, select_matrix=select_matrix)

            confidence, indice_1 = torch.max(f_t , 1)
            entro = torch.sum(- f_t * torch.log(f_t + 1e-10), dim=1)
            entropy_norm = np.log(f_t.size(1))
            entro /= entropy_norm 
            
            loss_entropy = GSS_ent.gss_loss(entro) 
            loss_confidence = GSS_conf.gss_loss(confidence)
            loss_gss = loss_entropy + loss_confidence
           
           
            total_loss = loss_s + 0.1*loss_gss +0.1*Lcontrast
            optimizer.step()
            end = time.time()
            optimizer.zero_grad()
           

        logger.info('[Epoch:{}/{}][loss:{:.4f}]'.format(epoch+1, max_iter, total_loss.item()))
        # logger.info('[Epoch:{}/{}][loss:{:.4f}][loss_s:{:.4f}][loss_ent:{:.4f}][loss_conf:{:.4f}][loss_contrast:{:.4f}]' \
        #     .format(epoch+1, max_iter, total_loss.item(),\
        #     loss_s.item(),loss_entropy.item(),loss_confidence.item(),Lcontrast.item()))
        # Evaluate
        if (epoch % evaluate_interval == evaluate_interval-1):
            mAP = evaluate(model,
                            query_dataloader,
                            retrieval_dataloader,
                            code_length,
                            device,
                            topk,
                            save = True,
                            )
            logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                epoch+1,
                max_iter,
                mAP,
            ))

    # Evaluate and save 
    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   save=False,
                   )
    # torch.save({'iteration': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #         }, os.path.join('checkpoints', 'resume_{}.t'.format(code_length)))
    logger.info('Training finish, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))


def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, save):
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
    
    # One-hot encode targets

    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
   
    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )

    if save:
        np.save("code/query_code_{}_mAP_{}".format(code_length, mAP), query_code.cpu().detach().numpy())
        np.save("code/retrieval_code_{}_mAP_{}".format(code_length, mAP), retrieval_code.cpu().detach().numpy())
        np.save("code/query_target_{}_mAP_{}".format(code_length, mAP), onehot_query_targets.cpu().detach().numpy())
        np.save("code/retrieval_target_{}_mAP_{}".format(code_length, mAP), onehot_retrieval_targets.cpu().detach().numpy())
    
    model.train()

    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, _,index in dataloader:
            data = data.to(device)
            _,_,_,outputs= model(data)
            code[index, :] = outputs.sign().cpu()

    return code

def print_image(data, name):
    from PIL import Image
    im = Image.fromarray(data)
    im.save('fig/topk/{}.png'.format(name))








    


    
