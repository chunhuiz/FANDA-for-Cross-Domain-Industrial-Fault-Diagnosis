"""
Test the model with target domain
"""
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score

use_gpu = True

def test(feature_extractor, class_classifier, source_dataloader, target_dataloader):
    """
    Test the performance of the model
    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :return: None
    """
    # setup the network
    feature_extractor.eval()
    class_classifier.eval()
    source_correct = 0.0
    target_correct = 0.0

    ground_truth_source = []
    predict_source = []
    for batch_idx, sdata in enumerate(source_dataloader):
        # setup hyperparameters

        input1, label1 = sdata
        if use_gpu:
            input1, label1 = Variable(input1.type(torch.FloatTensor).cuda()), Variable(label1.cuda())
        else:
            input1, label1 = Variable(input1), Variable(label1)

        output1 = class_classifier(feature_extractor(input1))
        pred1 = output1.data.max(1, keepdim = True)[1]
        source_correct += pred1.eq(label1.data.view_as(pred1)).cpu().sum()

        ground_truth_source += list(label1.data.cpu().numpy().squeeze())
        predict_source += list(pred1.data.cpu().numpy().squeeze())
        
    ground_truth_target = []
    predict_target = []
    for batch_idx, tdata in enumerate(target_dataloader):
        # setup hyperparameters

        input2, label2 = tdata
        if use_gpu:
            input2, label2 = Variable(input2.type(torch.FloatTensor).cuda()), Variable(label2.cuda())
        else:
            input2, label2 = Variable(input2), Variable(label2)

        output2 = class_classifier(feature_extractor(input2))
        pred2 = output2.data.max(1, keepdim=True)[1]
        target_correct += pred2.eq(label2.data.view_as(pred2)).cpu().sum()
        #
        # tgt_preds = domain_classifier(feature_extractor(input2), constant)
        # tgt_preds = tgt_preds.data.max(1, keepdim=True)[1]
        # tgt_correct += tgt_preds.eq(tgt_labels.data.view_as(tgt_preds)).cpu().sum()

        ground_truth_target += list(label2.data.cpu().numpy().squeeze())
        predict_target += list(pred2.data.cpu().numpy().squeeze())

    source_acc = accuracy_score(ground_truth_source, predict_source)
    source_f1 = f1_score(ground_truth_source, predict_source, average='weighted')
    
    target_acc = accuracy_score(ground_truth_target, predict_target)
    target_f1 = f1_score(ground_truth_target, predict_target, average='weighted')

    print('\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} ({:.4f}%)'.format(
        source_correct, len(source_dataloader.dataset), 100. * source_acc,
        target_correct, len(target_dataloader.dataset), 100. * target_acc
    ))
    return source_acc, source_f1, target_acc, target_f1