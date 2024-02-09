"""
Main script for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import models
from my_test import test
import utils
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from my_data_loader import CWRU_data
import random

epochs = 30
source_domain = 'A'
target_domain = 'B'

batch_size = 512

lr = 0.01
save_dir = './experiment'
gamma = 10
theta = 1

C = 4
use_gpu = True
manual_seed = 123
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.backends.cudnn.deterministic = True

# prepare the source data and target data

Source_data = CWRU_data(source=True, target=False, length_signal=400)
source_dataloader = DataLoader(Source_data, batch_size=batch_size, shuffle=True)


Target_data = CWRU_data(source=False, target=True, length_signal=400)
target_dataloader = DataLoader(Target_data, batch_size=batch_size, shuffle=True)


# init models
feature_extractor = models.Extractor()
class_classifier = models.Class_classifier()
domain_classifier = models.Domain_classifier()
domain1_classifier = models.Domain1_classifier()
domain2_classifier = models.Domain2_classifier()
domain3_classifier = models.Domain3_classifier()
domain4_classifier = models.Domain4_classifier()


if use_gpu:
    feature_extractor.cuda()
    class_classifier.cuda()
    domain_classifier.cuda()
    domain1_classifier.cuda()
    domain2_classifier.cuda()
    domain3_classifier.cuda()
    domain4_classifier.cuda()

# init criterions
class_criterion = nn.NLLLoss()
domain_criterion = nn.NLLLoss()

# init optimizer
# optimizer = optim.SGD([{'params': feature_extractor.parameters()},
#                         {'params': class_classifier.parameters()},
#                         {'params': domain_classifier.parameters()}], lr= lr, momentum= 0.9)
optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                        {'params': class_classifier.parameters()},
                        {'params': domain_classifier.parameters()},
                       {'params': domain1_classifier.parameters()},
                       {'params': domain2_classifier.parameters()},
                       {'params': domain3_classifier.parameters()},
                       {'params': domain4_classifier.parameters()}], weight_decay=5e-4, lr= lr, momentum= 0.9)


source_acc_list = []
source_f1_list = []
target_acc_list = []
target_f1_list = []
loss_sum_list = []
loss_class_list = []
loss_domain_list = []
loss_fg_domain_list = []


for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))

    # setup models
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()
    domain1_classifier.train()
    domain2_classifier.train()
    domain3_classifier.train()
    domain4_classifier.train()

    # steps
    start_steps = epoch*len(source_dataloader)
    total_steps = epochs*len(source_dataloader)

    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):

        # setup hyperparameters
        p = float(batch_idx+start_steps)/total_steps
        constant = 2./(1.+np.exp(-gamma*p))-1

        # prepare the data
        input1, label1 = sdata
        input2, label2 = tdata
        size = min((input1.shape[0], input2.shape[0]))
        input1, label1 = input1[0:size, :, :], label1[0:size]
        input2, label2 = input2[0:size, :, :], label2[0:size]
        if use_gpu:
            input1, label1 = Variable(input1.type(torch.FloatTensor).cuda()), Variable(label1.cuda())
            input2, label2 = Variable(input2.type(torch.FloatTensor).cuda()), Variable(label2.cuda())
        else:
            input1, label1 = Variable(input1), Variable(label1)
            input2, label2 = Variable(input2), Variable(label2)

        # setup optimizer
        optimizer = utils.optimizer_scheduler(optimizer, p)
        optimizer.zero_grad()

        # prepare domain labels
        if use_gpu:
            source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor).cuda())
            target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
        else:
            source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor))
            target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))

        # compute the output of source domain and target domain
        src_feature = feature_extractor(input1)
        tgt_feature = feature_extractor(input2)

        # compute the class loss of src_feature
        class_preds = class_classifier(src_feature)
        class_loss = class_criterion(class_preds, label1)


        # compute the domain loss of src_feature and target_feature
        tgt_preds = domain_classifier(tgt_feature, constant)
        src_preds = domain_classifier(src_feature, constant)
        tgt_loss = domain_criterion(tgt_preds, target_labels)
        src_loss = domain_criterion(src_preds, source_labels)
        domain_loss = tgt_loss+src_loss


        tgt_class_preds = class_classifier(tgt_feature)
        tgt_pred = tgt_class_preds.data.max(1, keepdim=True)[1]

        for k in range(C):
            input_source_k_signal = input1[np.where(label1.cpu() == k)[0], :, :]
            if input_source_k_signal.shape[0] > 1:
                src_k = feature_extractor(input_source_k_signal)
                src_k_preds = domain1_classifier(src_k, constant)
                src_k_labels = Variable(torch.zeros((src_k.size()[0])).type(torch.LongTensor).cuda())
                src_k_loss = domain_criterion(src_k_preds, src_k_labels)
            else:
                src_k_loss = 0

            input_target_k_signal = input2[np.where(label2.cpu() == k)[0], :, :]
            if input_target_k_signal.shape[0] > 1:
                tgt_k = feature_extractor(input_target_k_signal)
                tgt_k_preds = domain1_classifier(tgt_k, constant)
                tgt_k_labels = Variable(torch.ones((tgt_k.size()[0])).type(torch.LongTensor).cuda())
                tgt_k_loss = domain_criterion(tgt_k_preds, tgt_k_labels)
            else:
                tgt_k_loss = 0

            err_k_domain = src_k_loss+tgt_k_loss

            if k == 0:
                err_fine_grained = err_k_domain
            else:
                err_fine_grained += err_k_domain

        finegrained_domain_loss = err_fine_grained
        # loss = class_loss + theta*finegrained_domain_loss
        loss = class_loss + theta * domain_loss + theta * finegrained_domain_loss
        # loss = class_loss
        loss.backward()
        optimizer.step()

        # print loss
        # if (batch_idx+1)%10 == 0:
        print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tGlobal Domain Loss: {:.6f}\tFine-grained Domain Loss: {:.6f}'.format(
            batch_idx*len(input2), len(target_dataloader.dataset),
            100.*batch_idx/len(target_dataloader), loss.item(), class_loss.item(),
            domain_loss.item(),finegrained_domain_loss.item()
        ))

    source_acc, source_f1, target_acc, target_f1 = test(feature_extractor, class_classifier, source_dataloader, target_dataloader)
