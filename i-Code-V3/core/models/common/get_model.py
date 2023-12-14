from email.policy import strict
import torch
import torchvision.models
import os.path as osp
import copy
# from core.common.logger import print_log 
from .utils import \
    get_total_param, get_total_param_sum, \
    get_unit

def singleton(class_):  # 为了确保一个类只被实例化一次，class_ :要实例化的类
    instances = {}      # 一个字典，用于存储已经创建的类的实例
    # 当第一次调用getinstance时，它会创建一个新的class_的实例并将其存储在instances字典中。
    # 之后再次调用getinstance时，它会返回已经存储在instances中的同一个实例。
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

def preprocess_model_args(args):   # 对传入的参数进行预处理，特别是与模型相关的参数
    # If args has layer_units, get the corresponding
    #     units.
    # If args get backbone, get the backbone model.
    args = copy.deepcopy(args)   # 深拷贝，以确保原始参数不会被修改
    if 'layer_units' in args:    # args中有'layer_units'这个键
        layer_units = [
            get_unit()(i) for i in args.layer_units  # 使用get_unit()函数为每个单位创建一个新的实例
        ]
        args.layer_units = layer_units
    if 'backbone' in args:
        args.backbone = get_model()(args.backbone)
    return args

@singleton   # 装饰器，表示get_model是一个单例类，这意味着在整个程序运行过程中，这个类只能有一个实例。
class get_model(object):  # 当创建一个get_model的实例时，该方法会被调用。它初始化两个字典：model和version
    def __init__(self):
        self.model = {}    # 存储模型
        self.version = {}  # 存储版本

    def register(self, model, name, version='x'):  # 用于注册模型。它接受模型、名称和版本作为参数，并将它们存储在model和version字典中
        self.model[name] = model
        self.version[name] = version

    def __call__(self, cfg, verbose=True):  # 使得类的实例可以像函数一样被调用
    # cfg是一个配置对象，可能包含模型的各种配置信息；verbose是一个布尔值，用于控制是否输出详细信息
        """
        Construct model based on the config. 
        """
        t = cfg.type

        # the register is in each file
        # 据类型t的值，动态地导入不同的模块或类，灵活地支持多种不同的模型类型
        if t.find('audioldm')==0:
            from ..latent_diffusion.vae import audioldm
        elif t.find('autoencoderkl')==0:
            from ..latent_diffusion.vae import autokl
        elif t.find('optimus')==0:
            from ..latent_diffusion.vae import optimus
            
        elif t.find('clip')==0:
            from ..encoders import clip
        elif t.find('clap')==0:
            from ..encoders import clap   
            
        elif t.find('sd')==0:
            from .. import sd
        elif t.find('codi')==0:
            from .. import codi
        elif t.find('openai_unet')==0:
            from ..latent_diffusion import diffusion_unet
        
        # 预处理传入的参数，然后使用这些参数来构造模型
        args = preprocess_model_args(cfg.args)
        net = self.model[t](**args)

        return net

    def get_version(self, name):
        return self.version[name]

def register(name, version='x'): # 返回一个装饰器
    def wrapper(class_):
        get_model().register(class_, name, version)  # 这个装饰器接受一个类作为参数，并注册这个类
        return class_
    return wrapper
