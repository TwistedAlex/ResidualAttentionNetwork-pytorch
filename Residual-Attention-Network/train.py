from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
import argparse
from torchvision import transforms, datasets, models
import os
import cv2
import time
from dataloaders.deepfake_data import DeepfakeLoader
# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
from model.residual_attention_network import ResidualAttentionModel_92 as ResidualAttentionModel
from utils.image import deepfake_preprocess_imagev2
from metrics.metrics import save_roc_curve, save_roc_curve_with_threshold, roc_curve
from sklearn.metrics import accuracy_score, average_precision_score
import logging
import pathlib
import sys
from datetime import datetime


np.set_printoptions(threshold=sys.maxsize)
model_file = 'model_92_sgd.pkl'
classes = ('Neg', 'Pos')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def my_collate(batch):
    orig_imgs, preprocessed_imgs, agumented_imgs, masks, preprocessed_masks, \
    used_masks, labels, datasource, file, background_mask, indices = zip(*batch)
    masks_only = [mask for mask in preprocessed_masks if mask.size > 1]
    res_dict = {'orig_images': orig_imgs,
                'preprocessed_images': preprocessed_imgs,
                'augmented_images': agumented_imgs, 'orig_masks': masks,
                'preprocessed_masks': masks_only,
                'used_masks': used_masks,
                'labels': labels, 'source': datasource, 'filename': file,
                'bg_mask': background_mask, 'idx': indices, 'all_masks': preprocessed_masks, }
    return res_dict


# for test
def test(model, test_loader, logger, writer, epoch, btrain=False, model_file='model_92.pkl',
         device=torch.device('cuda:0'), test_intermediate_output_dir=""):
    # Test
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    y_true, y_pred = [], []
    # y_pred2 = []
    # y_true2, y_pred2 = [], []
    # count = 0
    for sample in test_loader:
        # if count == 2:
        #     break
        label_idx_list = sample['labels']
        filename_list = sample['filename']
        batch = torch.stack(sample['preprocessed_images'], dim=0).squeeze()
        images = batch.to(device)
        labels = torch.Tensor(label_idx_list).to(device).float().squeeze()
        if len(test_intermediate_output_dir) == 0:
            outputs = model(images)
        else:
            outputs = model(images, output_intermediate=True, filename_list=filename_list,
                            output_dir=test_intermediate_output_dir)
        # _, predicted = torch.max(outputs.data, 1)
        # original logic:
        # total += labels.size(0)
        # correct += (predicted == labels).sum()
        # #
        # c = (predicted == labels).squeeze()
        # for i in range(20):
        #     label = labels.data[i]
        #     class_correct[label] += c[i]
        #     class_total[label] += 1
        # Version 1: output size 1
        # print(outputs.sigmoid().flatten().tolist())
        # print(labels.flatten().tolist())
        # y_pred2.extend(outputs.flatten().tolist())
        y_pred.extend(outputs.sigmoid().flatten().tolist())
        y_true.extend(labels.flatten().tolist())
        # Version 2: output size 2
        # y_pred.extend(outputs.sigmoid()[:, 0].flatten().tolist())
        # y_true.extend(labels[:, 0].flatten().tolist())
        # y_pred.extend(outputs.sigmoid()[:, 1].flatten().tolist())
        # y_true.extend(labels[:, 1].flatten().tolist())
        # count += 1
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # y_true2, y_pred2 = np.array(y_true2), np.array(y_pred2)
    # print("y_true")
    # print(y_true)
    # print("y_pred")
    # print(y_pred)
    # print("y_pred2")
    # print(y_pred2)
    # print("y_true2")
    # print(y_true2)
    # print("y_pred2")
    # print(y_pred2)
    # print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    # print('Accuracy of the model on the test images:', float(correct) / total)
    # logger.warning('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    # logger.warning('Accuracy of the model on the test images:' + str(float(correct) / total))
    ap = average_precision_score(y_true, y_pred)
    fpr, tpr, auc, threshold = roc_curve(y_true, y_pred)
    print('AP :', ap)
    logger.warning('AP :' + str(ap))
    print('AUC :', auc)
    logger.warning('AUC :' + str(auc))
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    logger.warning('r_acc :' + str(r_acc))
    logger.warning('f_acc :' + str(f_acc))
    writer.add_scalar('Loss/train/AP',
                      ap,
                      epoch)
    writer.add_scalar('Loss/train/AUC',
                      auc,
                      epoch)
    # exit(1)
    # ap2 = average_precision_score(y_true2, y_pred2)
    # fpr2, tpr2, auc2, threshold2 = roc_curve(y_true2, y_pred2)
    # avgap = (ap2 + ap) / 2
    # avgAuc = (auc2 + auc) / 2
    # print('AP for Pos/fake image:', ap2)
    # logger.warning('AP for Pos/fake image:' + str(ap2))
    # print('AUC for Pos/fake image:', auc2)
    # logger.warning('AUC for Pos/fake image:' + str(auc2))
    # print('Avg AP:', avgap)
    # logger.warning('Avg AP:' + str(avgap))
    # print('Avg AUC:', str(avgAuc))
    # logger.warning('Avg AUC:' + str(avgAuc))

    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))
    #     logger.warning('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))
    return auc  # correct / total


def train(model, device, logger, epoch, train_loader, optimizer, criterion, writer, cfg):
    model.train()
    tims = time.time()
    iter_i = 0
    total_iter_i = 0
    total_losses = 0
    logger.warning('epoch： ' + str(epoch))
    print('epoch： ' + str(epoch))
    for sample in train_loader.datasets['train']:
        logger.warning('    iter： ' + str(iter_i))
        label_idx_list = sample['labels']
        filename_list = sample['filename']
        batch = torch.stack(sample['preprocessed_images'], dim=0).squeeze()
        images = batch.to(device)
        labels = torch.Tensor(label_idx_list).to(device)
        # images = Variable(images.cuda())
        # # print(images.data)
        # labels = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        # print(loss)
        loss.backward()
        optimizer.step()
        # print("hello")
        total_losses += loss.detach().cpu().item()
        writer.add_scalar('Loss/train/cl_loss_per_iter',
                          loss.detach().cpu().item(),
                          total_iter_i)
        if (iter_i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (
                epoch + 1, args.total_epoch, iter_i + 1, len(train_loader), loss.item()))
            logger.warning("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (
                epoch + 1, args.total_epoch, iter_i + 1, len(train_loader), loss.item()))
        iter_i += 1
        total_iter_i += 1
        # if iter_i == 10:
        #     break
    writer.add_scalar('Loss/train/cl_loss_per_epoch',
                      total_losses / (iter_i * args.batchsize),
                      epoch)
    print('the epoch takes time:', time.time() - tims)
    print('evaluate test set:')
    logger.warning('the epoch takes time:' + str(time.time() - tims))
    logger.warning('evaluate test set:')
    if epoch == args.total_epoch - 1:
        acc = test(model, train_loader.datasets['test'], logger, writer, epoch, btrain=True, device=device,
                   test_intermediate_output_dir=cfg['test_intermediate_output_dir'])
    else:
        acc = test(model, train_loader.datasets['test'], logger, writer, epoch, btrain=True, device=device)

    if acc > cfg['acc_best']:
        cfg['acc_best'] = acc
        print("***************************************")
        logger.warning("***************************************")
        print(' epoch: ', str(epoch))
        logger.warning(' epoch: ' + str(epoch))
        print('current best acc,', cfg['acc_best'])
        logger.warning('current best acc,' + str(cfg['acc_best']))
        # torch.save(model.state_dict(), args.output_dir + args.log_name + '/' + str(epoch) + '_' + model_file)
        torch.save({
            'total_steps': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, args.output_dir + args.log_name + '/' + model_file)  # datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        print(' Latest epoch: ', str(epoch))
        logger.warning(' Latest epoch: ' + str(epoch))
        torch.save({
            'total_steps': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, args.output_dir + args.log_name + '/' + "trainLast_" + model_file)
    # Decaying Learning Rate
    if (epoch + 1) / float(args.total_epoch) == 0.3 or (epoch + 1) / float(args.total_epoch) == 0.6 or (
            epoch + 1) / float(args.total_epoch) == 0.9:
        cfg['lr'] /= 10
        print('reset learning rate to:', cfg['lr'])
        logger.warning('reset learning rate to:' + str(cfg['lr']))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg['lr']
            print(param_group['lr'])
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)


parser = argparse.ArgumentParser(description='PyTorch GAIN Training')
parser.add_argument('--batchsize', type=int, default=32, help='batch size')
parser.add_argument('--input_dir', help='path to the input idr', type=str)
parser.add_argument('--batch_pos_dist', type=float, help='positive relative amount in a batch', default=0.5)
parser.add_argument('--total_epochs', type=int, default=50, help='total number of epoch to train')
parser.add_argument('--nepoch', type=int, default=510, help='number of iterations per epoch')
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--masks_to_use', type=float, default=0.2,
                    help='the relative number of masks to use in ex-supevision training')
parser.add_argument('--output_dir', help='path to the outputdir', type=str, default="logs/")
parser.add_argument('--log_name', type=str, help='identifying name for storing tensorboard logs')
parser.add_argument('--writer_file_load', type=str, default='',
                    help='a full path including the name of the writer_file to load from, empty otherwise')
parser.add_argument('--checkpoint_file_path_load', type=str, default='',
                    help='a full path including the name of the checkpoint_file to load from, empty otherwise')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--test', '-t', action='store_true', help='test mode')
# Image Preprocessing


def main(args):
    if len(args.writer_file_load) > 1:
        writer = SummaryWriter(args.output_dir + args.writer_file_load)
    else:
        writer = SummaryWriter(args.output_dir + args.log_name + '/' + 'logs_' +
                               datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    device = torch.device('cuda:' + str(args.deviceID))
    batch_size = args.batchsize
    epoch_size = args.nepoch

    intermediate_output_dir = args.output_dir + args.log_name + '/' + "intermediate_output" + '/'
    test_intermediate_output_dir = args.output_dir + args.log_name + '/' + "test_intermediate_output" + '/'
    pathlib.Path(intermediate_output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_intermediate_output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output_dir + args.log_name + '/').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,
                        filename=args.output_dir + args.log_name + '/' + "std.log",
                        format='%(asctime)s %(message)s')
    logger = logging.getLogger('PIL')
    logger.setLevel(logging.WARNING)
    # when image is rgb, totensor do the division 255
    # Deepfake Dataset


    deepfake_loader = DeepfakeLoader(args.input_dir, [1 - args.batch_pos_dist, args.batch_pos_dist],
                                     batch_size=batch_size, steps_per_epoch=epoch_size,
                                     masks_to_use=args.masks_to_use, mean=mean, std=std,
                                     transform=deepfake_preprocess_imagev2,
                                     collate_fn=my_collate)

    model = ResidualAttentionModel().to(device)
    i = 0
    num_train_samples = 0
    am_i = 0
    ex_i = 0
    total_i = 0
    IOU_i = 0
    acc_best = 0
    cfg = {'categories': classes, 'i': i, 'num_train_samples': num_train_samples,
           'am_i': am_i, 'ex_i': ex_i, 'total_i': total_i,
           'IOU_i': IOU_i, 'acc_best': acc_best, 'test_intermediate_output_dir': test_intermediate_output_dir,
           'lr': args.lr}
    lr = args.lr
    criterion = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    is_train = not args.test
    is_pretrain = False
    acc_best = 0
    total_epoch = args.total_epochs
    init_epoch = 0

    if is_train is True:
        if len(args.checkpoint_file_path_load) > 0:
            checkpoint = torch.load(args.checkpoint_file_path_load, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            init_epoch = checkpoint['epoch'] + 1
            # model.load_state_dict((torch.load(model_file)))
        # Training
        for epoch in range(init_epoch, total_epoch):
            train(model, device, logger, epoch, deepfake_loader, optimizer, criterion, writer, cfg)
        # Save the Model
        # torch.save(model.state_dict(), args.output_dir + args.log_name + '/last_model_92_sgd.pkl')
        torch.save({
            'total_steps': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, args.output_dir + args.log_name + '/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +
           '_last_model_92_sgd.pkl')
    else:
        test(model, deepfake_loader.datasets['test'], logger, writer, 0, btrain=False, device=device)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
