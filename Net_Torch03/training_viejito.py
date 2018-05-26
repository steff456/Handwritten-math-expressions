# -*- coding: utf-8 -*-

"""
Texture Net PyTorch implementation.
"""

# Standard lib imports
import os
import time
import glob
import argparse
import os.path as osp
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
# PyTorch imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local imports
from utils import AverageMeter
from utils.misc_utils import VisdomWrapper
from math_dataset import MathDataset

# Other imports
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser(
    description='Texture Net training routine')

# Dataloading-related settings
parser.add_argument('--data', type=str, default='.',
                    help='path to train data folder')
parser.add_argument('--save-folder', default='weights/',
                    help='location to save checkpoint models')
parser.add_argument('--snapshot', default='weights/texture_weights.pth',
                    help='path to weight snapshot file')
parser.add_argument('--predictions', default='predictions.txt',
                    help='file location to write predictions on')
parser.add_argument('--num-workers', default=2, type=int,
                    help='number of workers used in dataloading')
parser.add_argument('--split', default='train', type=str,
                    help='name of the dataset split used to train')
parser.add_argument('--val', default='test', type=str,
                    help='name of the dataset split used to validate')
parser.add_argument('--eval', default='test', type=str,
                    help='name of the dataset split used to evaluate')
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

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

trainset = MathDataset(args.data, args.split, transform=transform)
print(len(trainset.labels))

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, pin_memory=False,
    num_workers=args.workers)

start_epoch = args.start_epoch

if args.val:
    valset = MathDataset(args.data, args.val, transform=transform,
                         num_images={'testSymbols': None})

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, pin_memory=False,
        num_workers=args.workers)

if not osp.exists(args.save_folder):
    os.makedirs(args.save_folder)

# classes = ('bark1', 'bark2', 'bark3', 'wood1', 'wood2', 'wood1', 'water',
#            'granite', 'marbel', 'floor1', 'floor2', 'pebbles', 'wall',
#            'brick1', 'brick2', 'glass1', 'glass2', 'carpet1', 'carpet2',
#            'upholstery', 'wallpaper', 'fur', 'knit', 'corduroy', 'plaid')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 96, 3)
        self.conv4 = nn.Conv2d(96, 128, 3)
        self.lin1 = nn.Linear(8192, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 102)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.selu(self.conv4(x))
        # print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        # print(x.size())
        return F.log_softmax(x, dim=1)

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
        super().load_state_dict(state)


net = Net()

if osp.exists(args.snapshot):
    net.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    net = net.cuda()

if args.visdom is not None:
    visdom_url = urlparse(args.visdom)

    port = 80
    if visdom_url.port is not None:
        port = visdom_url.port

    print('Initializing Visdom frontend at: {0}:{1}'.format(
          args.visdom, port))
    vis = VisdomWrapper(server=visdom_url.geturl(), port=port,
                        env=args.env, use_incoming_socket=False)

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

    bar = tqdm(enumerate(train_loader))
    for batch_idx, (inputs, labels) in bar:
        inputs = Variable(inputs)
        labels = Variable(labels)
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss.update(loss.data[0], inputs.size(0))
        epoch_loss_stats.update(loss.data[0], inputs.size(0))

        if args.visdom is not None:
            cur_iter = batch_idx + (epoch - 1) * len(train_loader)
            vis.plot_line('train_plt',
                          X=torch.ones((1,)).cpu() * cur_iter,
                          Y=loss.data.cpu(),
                          update='append')

        if batch_idx % args.backup_iters == 0:
            filename = 'texture_{0}_snapshot.pth'.format(args.split)
            filename = osp.join(args.save_folder, filename)
            state_dict = net.state_dict()
            torch.save(state_dict, filename)

            optim_filename = 'texture_{0}_optim.pth'.format(args.split)
            optim_filename = osp.join(args.save_folder, optim_filename)
            state_dict = optimizer.state_dict()
            torch.save(state_dict, optim_filename)

        if batch_idx % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            bar.set_description('[{:5d}] ({:5d}/{:5d}) | ms/batch {:.6f} |'
                                ' loss {:.6f} | lr {:.7f}'.format(
                                    epoch, batch_idx, len(train_loader),
                                    elapsed_time * 1000, total_loss.avg,
                                    optimizer.param_groups[0]['lr']))
            total_loss.reset()

        start_time = time.time()

    epoch_total_loss = epoch_loss_stats.avg

    if args.visdom is not None:
        vis.plot_line('train_epoch_plt',
                      X=torch.ones((1, )).cpu() * epoch,
                      Y=torch.ones((1, )).cpu() * epoch_total_loss,
                      update='append')
    return epoch_total_loss


def val(epoch, loader):
    net.eval()
    acc_meter = AverageMeter()
    epoch_loss_stats = AverageMeter()
    start_time = time.time()

    bar = tqdm(enumerate(loader))
    for batch_idx, (inputs, labels) in bar:
        inputs = Variable(inputs, volatile=True)
        labels = Variable(labels, volatile=True)
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        epoch_loss_stats.update(loss.data[0], inputs.size(0))
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels.data).sum() / inputs.size(0)
        acc_meter.update(correct, inputs.size(0))

    return epoch_loss_stats.avg, acc_meter.avg


def test():
    print("Writing predictions")
    net.eval()
    labels = {}
    files = glob.glob(osp.join(args.data, "{0}_128/label_00/*.jpg".format(args.eval)))
    for file in tqdm(files):
        _id, _ = osp.splitext(osp.basename(file))
        img = Image.open(file)
        # img = transforms.ToTensor()(img)
        img = transform(img)
        if img.size(0) == 1:
            img = torch.stack([img] * 3, dim=1).squeeze()
        img = Variable(img, volatile=True).unsqueeze(0).cuda()
        output = net(img)
        output = output.cpu()
        _, predicted = output.max(1, keepdim=True)
        labels[_id] = predicted[0]

    with open(args.predictions, 'w') as f:
        f.write('\n'.join(['{0},{1}'.format(k, labels[k].data[0]) for k in labels]))


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
            if args.visdom is not None:
                vis.plot_line('train_acc_plt',
                              X=torch.ones((1,)).cpu() * epoch,
                              Y=torch.ones((1,)).cpu() * train_acc,
                              update='append')
            if args.val is not None:
                val_loss, val_acc = val(epoch, val_loader)
                if args.visdom is not None:
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
                filename = osp.join(args.save_folder, 'textures_best_weights.pth')
                torch.save(net.state_dict(), filename)
        test()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    finally:
        filename = osp.join(args.save_folder, 'textures_best_weights.pth')
        if osp.exists(filename):
            net.load_state_dict(torch.load(filename))
        test()
