import os
import argparse
import shutil
import time

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import numpy as np
import dataset.Dataset as Data
import dataset.transforms as joint_transforms
from model.DeepLab import SFNet
from evaluation import evaluate

# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
parser = argparse.ArgumentParser(description='Pytorch WeatherNet Training')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    help='mini-batch size(default:5)')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--power', default=0.9, type=float,
                    help='lr power (default: 0.9)')
parser.add_argument('--print-freq', default=20, type=int,
                    help='print frequency(default: 10)')
parser.add_argument('--num-class', default=2, type=int,
                    help='number of class(default: 5)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='wheter to use standard augmentation(default: True)')
parser.add_argument('--data-root', default='/data/caoyong/train_data/', type=str,
                    help='path to data')
parser.add_argument('--val-root', default='/data/caoyong/test_data/', type=str,
                    help='path to data')
parser.add_argument('--resume', default='', type=str,
                    help='path to latset checkpoint(default: None')
parser.add_argument('--name', default='WeatherNet', type=str,
                    help='name of experiment')
parser.set_defaults(augment=True)

best_record = {'epoch': 0, 'val_loss': 0.0, 'acc': 0.0, 'miou': 0.0}


def main():
    global args, best_record
    args = parser.parse_args()

    if args.augment:
        transform_train = joint_transforms.Compose([
            joint_transforms.RandomCrop(256),
            joint_transforms.Normalize(),
            joint_transforms.ToTensor(),
            ])
    else:
        transform_train = None

    dataset_train = Data.WData(args.data_root, transform_train)
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=16)

    dataset_val = Data.WData(args.val_root, transform_train)
    dataloader_val = data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=None, num_workers=16)

    model = SFNet(input_channels=37, dilations=[2, 4, 8], num_class=2)

    # multi gpu
    model = torch.nn.DataParallel(model)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
    ))

    model = model.cuda()
    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda()
    optimizer = torch.optim.SGD([{'params': get_1x_lr_params(model)},
                                {'params': get_10x_lr_params(model), 'lr': 10 * args.learning_rate}],
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(dataloader_train, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc, mean_iou, val_loss = validate(dataloader_val, model, criterion, epoch)

        is_best = mean_iou > best_record['miou']
        if is_best:
            best_record['epoch'] = epoch
            best_record['val_loss'] = val_loss.avg
            best_record['acc'] = acc
            best_record['miou'] = mean_iou
        save_checkpoint({
            'epoch': epoch + 1,
            'val_loss': val_loss.avg,
            'accuracy': acc,
            'miou': mean_iou,
            'model': model,
        }, is_best)

        print('------------------------------------------------------------------------------------------------------')
        print('[epoch: %d], [val_loss: %5f], [acc: %.5f], [miou: %.5f]' %(
            epoch, val_loss.avg, acc, mean_iou))
        print('best record: [epoch: {epoch}], [val_loss: {val_loss:.5f}], [acc: {acc:.5f}], [miou: {miou:.5f}]'.format(**best_record))
        print('------------------------------------------------------------------------------------------------------')


def train(dataloader_train, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(dataloader_train):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target.long())

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() -end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(dataloader_train),
                   batch_time=batch_time, loss=losses
            ))


def validate(dataloader_val, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    target_list = []
    pred_list = []

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(dataloader_val):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target.long(), volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        for j in range(target.shape[0]):
            target_list.append(target.cpu()[j].numpy())
            pred_list.append(np.argmax(output[j].cpu().data.numpy(), axis=0))

        if i % args.print_freq == 0:

            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(dataloader_val), batch_time=batch_time, loss=losses))

    acc, mean_iou = evaluate(target_list, pred_list, args.num_class, 'result/{}.csv'.format(epoch))
    return acc, mean_iou, losses


def adjust_learning_rate(optimizer, epoch):
    lr = args.learning_rate*((1-float(epoch)/args.epochs)**args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def split_dataset(label_list):
    label = [i_id.strip() for i_id in open(label_list)]
    train_label = [name for i, name in enumerate(label) if i % 10 != 0]
    val_label = [name for i, name in enumerate(label) if i % 10 == 0]
    return train_label, val_label


def save_checkpoint(model, is_best, filename='checkpoing.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(model, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % args.name + 'model_best.pth.tar')


def get_1x_lr_params(model):
    b = []

    b.append(model.module.deeplab.tem_module.layer1)
    b.append(model.module.deeplab.tem_module.layer2)
    b.append(model.module.deeplab.tem_module.layer3)
    b.append(model.module.deeplab.tem_module.layer4)
    b.append(model.module.deeplab.hum_module.layer1)
    b.append(model.module.deeplab.hum_module.layer2)
    b.append(model.module.deeplab.hum_module.layer3)
    b.append(model.module.deeplab.hum_module.layer4)
    b.append(model.module.deeplab.x_wind_module.layer1)
    b.append(model.module.deeplab.x_wind_module.layer2)
    b.append(model.module.deeplab.x_wind_module.layer3)
    b.append(model.module.deeplab.x_wind_module.layer4)
    b.append(model.module.deeplab.y_wind_module.layer1)
    b.append(model.module.deeplab.y_wind_module.layer2)
    b.append(model.module.deeplab.y_wind_module.layer3)
    b.append(model.module.deeplab.y_wind_module.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            for k in j.parameters():
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    b = []

    b.append(model.module.deeplab.tem_module.layer5)
    b.append(model.module.deeplab.hum_module.layer5)
    b.append(model.module.deeplab.x_wind_module.layer5)
    b.append(model.module.deeplab.y_wind_module.layer5)
    b.append(model.module.deeplab.aspp)
    b.append(model.module.deeplab.multiply)
    b.append(model.module.classifier)

    for i in range(len(b)):
        for j in b[i].modules():
            for k in j.parameters():
                if k.requires_grad:
                    yield k


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
