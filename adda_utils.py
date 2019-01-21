import os
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import refs

def make_cuda(tensor):
      if torch.cuda.is_available:
            tensor = tensor.cuda()
      return tensor

def make_variable(tensor, volatile=False):
      """ turn tensors into variables. 
      This is deprecated in newer pytorch versions"""
      tensor = make_cuda(tensor)
      return torch.autograd.Variable(tensor, volatile=volatile)



############################# LENET TEST #####################################

def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loader(name, train=True):
    """Get data loader by name."""
    if name == "MNIST":
        return get_mnist(train)
    elif name == "USPS":
        return get_usps(train)


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        #net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net