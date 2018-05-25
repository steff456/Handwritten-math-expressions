from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

import re
import math
import torch.nn as nn
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


def vgg16(*args, num_classes=1000, **kwargs):
    pretrained = False
    if 'pretrained' in kwargs:
        pretrained = kwargs['pretrained']
        kwargs['pretrained'] = False
    base_vgg = vgg16_builder(*args, **kwargs)
    classifier = nn.Sequential(
        nn.Linear(102400, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes)
    )
    base_vgg = list(base_vgg.children())[:-1]
    base_vgg += [classifier]
    model = nn.Sequential(*base_vgg)
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
