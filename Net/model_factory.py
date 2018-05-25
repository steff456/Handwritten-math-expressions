from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

import re
import math
import torch.nn as nn
import torch.nn.functional as F
from dpn_model import dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107
import torchvision.models.resnet as resnet
from torchvision.models.resnet import (
    resnet18, resnet34, resnet50, resnet101, resnet152)
import torchvision.models.densenet as densenet
from torchvision.models.densenet import (
    densenet121, densenet169, densenet161, densenet201)
import torchvision.models.inception as inception
from torchvision.models.inception import inception_v3
import torchvision.models.vgg as vgg
from torchvision.models.vgg import vgg16 as vgg16_builder
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from PIL import Image

VGG16_URL = 'https://download.pytorch.org/models/vgg16-397923af.pth'
MODULE_REGEX = r'([a-zA-Z]+)((_.*)|(\d+))'
REGEX = re.compile(MODULE_REGEX)


class VGG16(nn.Module):
    def __init__(self, *args, num_classes=1000, pretrained=None, **kwargs):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 256, 3)
        self.conv8 = nn.Conv2d(256, 512, 3)
        self.conv9 = nn.Conv2d(512, 512, 3)
        self.conv10 = nn.Conv2d(512, 512, 3)
        self.fc1 = nn.Linear(512*5*5, num_classes)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        # y = x
        # x = F.max_pool2d(x, 2)
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        # z = x
        # x = F.max_pool2d(x, 2)
        x = F.selu(self.conv5(x))
        x = F.selu(self.conv6(x))
        x = F.selu(self.conv7(x))
        # w = x
        # x = F.max_pool2d(x, 2)
        x = F.selu(self.conv8(x))
        x = F.selu(self.conv9(x))
        x = F.selu(self.conv10(x))
        # x = F.upsample(x, size=(y.size(-2), y.size(-1)), mode='bilinear')
        # z = F.upsample(z, size=(y.size(-2), y.size(-1)), mode='bilinear')
        # w = F.upsample(w, size=(y.size(-2), y.size(-1)), mode='bilinear')
        # x = torch.cat([x, y, z, w], dim=1)
        x = F.selu(self.convL(x))
        x = x.view(-1, 512*5*5)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

    # def load_state_dict(self, new_state):
    #     state = self.state_dict()
    #     for layer in state:
    #         if layer in new_state:
    #             if state[layer].size() == new_state[layer].size():
    #                 state[layer] = new_state[layer]
    #     super().load_state_dict(state)


def vgg16(*args, num_classes=1000, **kwargs):
    model = VGG16(*args, num_classes=num_classes, **kwargs)
    return model


def create_model(model_name, num_classes=1000, pretrained=False, **kwargs):
    if model_name in globals():
        net_definition = globals()[model_name]
        act_pretrained = pretrained
        if not model_name.startswith('dpn'):
            pretrained = False
        model = net_definition(
            num_classes=num_classes, pretrained=pretrained, **kwargs)
        if act_pretrained and not pretrained:
            re_match = REGEX.match(model_name)
            module = globals()[re_match.group(1)]
            URL = module.model_urls[model_name]
            state_dict = model.state_dict()
            pretrained_state = model_zoo.load_url(URL)
            for layer_name in pretrained_state:
                if layer_name in state_dict:
                    if (state_dict[layer_name].size() ==
                            pretrained_state[layer_name].size()):
                        state_dict[layer_name] = pretrained_state[layer_name]
            model.load_state_dict(state_dict)
    else:
        assert False, "Unknown model architecture (%s)" % model_name
    return model


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


DEFAULT_CROP_PCT = 0.875


def get_transforms_eval(model_name, img_size=224, crop_pct=None):
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    if 'dpn' in model_name:
        if crop_pct is None:
            # Use default 87.5% crop for model's native img_size
            # but use 100% crop for larger than native as it
            # improves test time results across all models.
            if img_size == 224:
                scale_size = int(math.floor(img_size / DEFAULT_CROP_PCT))
            else:
                scale_size = img_size
        else:
            scale_size = int(math.floor(img_size / crop_pct))
        normalize = transforms.Normalize(
            mean=[124 / 255, 117 / 255, 104 / 255],
            std=[1 / (.0167 * 255)] * 3)
    elif 'inception' in model_name:
        scale_size = int(math.floor(img_size / crop_pct))
        normalize = LeNormalize()
    else:
        scale_size = int(math.floor(img_size / crop_pct))
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Scale(scale_size, Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize])
