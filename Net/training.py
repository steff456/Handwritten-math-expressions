# -*- coding: utf-8 -*-

"""
Handwritten Net PyTorch implementation.
"""

# Standard lib imports
import os
import time
import glob
import argparse
import os.path as osp
from urllib.parse import urlparse

# PyTorch imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Normalize, Compose

# Local imports
from utils import AverageMeter
from math_dataset import MathDataset
from model_factory import create_model
from utils.misc_utils import VisdomWrapper

# Other imports
from tqdm import tqdm
from PIL import Image

"""
python -u -m torch.distributed.launch --nproc_per_node=2 training.py --*args
--world-size 2 --dist-backend nccl
"""


parser = argparse.ArgumentParser(
    description='Mathematical symbols Net training routine')

# Dataloading-related settings
parser.add_argument('--data', type=str, default='.',
                    help='path to train data folder')
parser.add_argument('--save-folder', default='weights/',
                    help='location to save checkpoint models')
parser.add_argument('--snapshot', default='weights/texture_weights.pth',
                    help='path to weight snapshot file')
parser.add_argument('--split', default='train', type=str,
                    help='name of the dataset split used to train')
parser.add_argument('--val', default='test', type=str,
                    help='name of the dataset split used to validate')
parser.add_argument('--eval', default='test', type=str,
                    help='name of the dataset split used to evaluate')
parser.add_argument('--backend', default='vgg16', type=str,
                    help='base architecture to train')
# Realizar la evaluación antes del entrenamiento
parser.add_argument('--eval-first', default=False, action='store_true',
                    help='evaluate model weights before training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# Training procedure settings
parser.add_argument('--no-cuda', action='store_true',
                    help='Do not use cuda to train model')
# Cada cuanto quiero imprimir
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
# Cada cuanto quiero en el archivo
parser.add_argument('--backup-iters', type=int, default=1000,
                    help='iteration interval to perform state backups')
parser.add_argument('--batch-size', default=50, type=int,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum factor value')
parser.add_argument('--patience', default=2, type=int,
                    help='patience epochs for LR decreasing')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--start-epoch', type=int, default=1,
                    help='epoch number to resume')
parser.add_argument('--optim-snapshot', type=str,
                    default='weights/texture_optim.pth',
                    help='path to optimizer state snapshot')

# Distributed settings
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--local_rank', default=0, type=int,
                    help='distributed node rank number identification')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--pin-memory', action='store_true', default=False,
                    help='enable DataLoader CUDA memory pinning')

# Other settings
parser.add_argument('--visdom', type=str, default=None,
                    help='visdom URL endpoint')
parser.add_argument('--env', type=str, default='DMN-train',
                    help='visdom environment name')

args = parser.parse_args()

args_dict = vars(args)
print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                 for arg in args_dict]))
print('\n')

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.local_rank)

args.distributed = args.world_size > 1

args.rank = 0
args.nodes = 1

if args.distributed:
    print('Starting distribution node')
    dist.init_process_group(args.dist_backend, init_method='env://')
    print('Done!')

    args.rank = dist.get_rank()
    args.nodes = dist.get_world_size()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

transform = Compose([
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

trainset = MathDataset(args.data, args.split, transform=transform)

if args.distributed:
    train_sampler = DistributedSampler(trainset)
else:
    train_sampler = None

train_loader = DataLoader(trainset, batch_size=args.batch_size,
                          shuffle=(train_sampler is None),
                          sampler=train_sampler,
                          pin_memory=args.pin_memory,
                          num_workers=args.workers)

start_epoch = args.start_epoch

if args.val:
    valset = MathDataset(args.data, args.val, transform=transform)

    val_sampler = None
    if args.distributed:
        val_sampler = DistributedSampler(valset)

    val_loader = DataLoader(refer_val, batch_size=args.batch_size,
                            pin_memory=args.pin_memory,
                            num_workers=args.workers,
                            sampler=val_sampler)

if not osp.exists(args.save_folder):
    os.makedirs(args.save_folder)

net = create_model(args.backend,
                   num_classes=len(trainset.labels),
                   pretrained=True, **kwargs)

if osp.exists(args.snapshot):
    net.load_state_dict(torch.load(args.snapshot))

if args.distributed:
    if args.cuda:
        net = net.cuda()
    net = parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank], output_device=args.local_rank)

if args.cuda:
    net = net.cuda()

if args.visdom is not None and args.rank == 0:
    visdom_url = urlparse(args.visdom)

    port = 80
    if visdom_url.port is not None:
        port = visdom_url.port

    print('Initializing Visdom frontend at: {0}:{1}'.format(
          args.visdom, port))
    vis = VisdomWrapper(server=visdom_url.geturl(), port=port, env=args.env)

    vis.init_line_plot('train_plt', xlabel='Iteration', ylabel='Loss',
                       title='Current Model Loss Value', legend=['Loss'])
    vis.init_line_plot('train_epoch_plt', xlabel='Epoch', ylabel='Loss',
                       title='Current Model Epoch Loss Value', legend=['Loss'])
    vis.init_line_plot('train_acc_plt', xlabel='Epoch', ylabel='Acc',
                       title='Current Model Accuracy Value', legend=['Acc'])

    if args.val is not None:
        vis.init_line_plot('val_plt', xlabel='Epoch', ylabel='Loss',
                           title='Current Model Validation Loss Value',
                           legend=['Loss'])
        vis.init_line_plot('val_acc_plt', xlabel='Epoch', ylabel='Acc',
                           title='Current Model Validatos Accuracy Value',
                           legend=['Acc'])

criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

scheduler = ReduceLROnPlateau(
    optimizer, patience=args.patience)


def train(epoch):
    net.train()
    total_loss = AverageMeter()
    epoch_loss_stats = AverageMeter()
    start_time = time.time()
    if args.distributed:
        train_sampler.set_epoch(epoch)

    bar = tqdm(enumerate(train_loader))
    for batch_idx, (inputs, labels) in bar:
        inputs = inputs.requires_grad_()
        labels = labels.requires_grad_()
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item(), inputs.size(0))
        epoch_loss_stats.update(loss.item(), inputs.size(0))

        if args.visdom is not None and args.rank == 0:
            cur_iter = batch_idx + (epoch - 1) * len(train_loader)
            vis.plot_line('train_plt',
                          X=torch.ones((1,)).cpu() * cur_iter,
                          Y=loss.data.cpu(),
                          update='append')

        if batch_idx % args.backup_iters == 0:
            filename = 'math_{0}_snapshot.pth'.format(args.split)
            filename = osp.join(args.save_folder, filename)
            state_dict = net.state_dict()
            torch.save(state_dict, filename)

            optim_filename = 'math_{0}_optim.pth'.format(args.split)
            optim_filename = osp.join(args.save_folder, optim_filename)
            state_dict = optimizer.state_dict()
            torch.save(state_dict, optim_filename)

        if batch_idx % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            bar.set_description('{:2d}/{:2d} [{:5d}] ({:5d}/{:5d}) '
                                '| ms/batch {:.6f} |'
                                ' loss {:.6f} | lr {:.7f}'.format(
                                    args.rank, args.nodes - 1,
                                    epoch, batch_idx, len(train_loader),
                                    elapsed_time * 1000, total_loss.avg,
                                    optimizer.param_groups[0]['lr']))
            total_loss.reset()

        start_time = time.time()

    epoch_total_loss = epoch_loss_stats.avg

    if args.visdom is not None and args.rank == 0:
        vis.plot_line('train_epoch_plt',
                      X=torch.ones((1, )).cpu() * epoch,
                      Y=torch.ones((1, )).cpu() * epoch_total_loss,
                      update='append')
    return epoch_total_loss


def val(epoch, loader):
    net.eval()
    if args.distributed:
        val_sampler.set_epoch(epoch)
    acc_meter = AverageMeter()
    epoch_loss_stats = AverageMeter()
    start_time = time.time()

    bar = tqdm(enumerate(loader))
    for batch_idx, (inputs, labels) in bar:
        with torch.no_grad():
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            epoch_loss_stats.update(loss.item(), inputs.size(0))
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum() / inputs.size(0)
            acc_meter.update(correct, inputs.size(0))

    return epoch_loss_stats.avg, acc_meter.avg


if __name__ == '__main__':
    print('Train begins...')
    best_val_acc = None
    if args.eval_first:
        loader = val_loader if args.val is not None else train_loader
        val(start_epoch, loader)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(epoch)
            val_loss, train_acc = val(epoch, train_loader)
            val_acc = train_acc
            if args.visdom is not None and args.rank == 0:
                vis.plot_line('train_acc_plt',
                              X=torch.ones((1,)).cpu() * epoch,
                              Y=torch.ones((1,)).cpu() * train_acc,
                              update='append')
            if args.val is not None:
                val_loss, val_acc = val(epoch, val_loader)
                if args.visdom is not None and args.rank == 0:
                    vis.plot_line('val_acc_plt',
                                  X=torch.ones((1,)).cpu() * epoch,
                                  Y=torch.ones((1,)).cpu() * val_acc,
                                  update='append')
                    vis.plot_line('val_plt',
                                  X=torch.ones((1,)).cpu() * epoch,
                                  Y=torch.ones((1,)).cpu() * val_loss,
                                  update='append')
            scheduler.step(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '
                  '| epoch loss {:.6f} | val loss {:.6f} | acc {:.6f} |'
                  '| val acc {:.6f} |'.format(
                      epoch, time.time() - epoch_start_time, train_loss,
                      val_loss, train_acc, val_acc))
            print('-' * 89)
            if best_val_acc is None or val_acc > best_val_acc:
                best_val_acc = val_acc
                filename = osp.join(args.save_folder, 'math_best_weights.pth')
                torch.save(net.state_dict(), filename)
        # test()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    finally:
        filename = osp.join(args.save_folder, 'math_best_weights.pth')
        if osp.exists(filename):
            net.load_state_dict(torch.load(filename))
        # test()
