from email.policy import strict
import torch
import torchvision.models
import os.path as osp
import copy
# from core.common.logger import print_log 
from .utils import \
    get_total_param, get_total_param_sum, \
    get_unit

def singleton(class_):  # Ϊ��ȷ��һ����ֻ��ʵ����һ�Σ�class_ :Ҫʵ��������
    instances = {}      # һ���ֵ䣬���ڴ洢�Ѿ����������ʵ��
    # ����һ�ε���getinstanceʱ�����ᴴ��һ���µ�class_��ʵ��������洢��instances�ֵ��С�
    # ֮���ٴε���getinstanceʱ�����᷵���Ѿ��洢��instances�е�ͬһ��ʵ����
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

def preprocess_model_args(args):   # �Դ���Ĳ�������Ԥ�����ر�����ģ����صĲ���
    # If args has layer_units, get the corresponding
    #     units.
    # If args get backbone, get the backbone model.
    args = copy.deepcopy(args)   # �������ȷ��ԭʼ�������ᱻ�޸�
    if 'layer_units' in args:    # args����'layer_units'�����
        layer_units = [
            get_unit()(i) for i in args.layer_units  # ʹ��get_unit()����Ϊÿ����λ����һ���µ�ʵ��
        ]
        args.layer_units = layer_units
    if 'backbone' in args:
        args.backbone = get_model()(args.backbone)
    return args

@singleton   # װ��������ʾget_model��һ�������࣬����ζ���������������й����У������ֻ����һ��ʵ����
class get_model(object):  # ������һ��get_model��ʵ��ʱ���÷����ᱻ���á�����ʼ�������ֵ䣺model��version
    def __init__(self):
        self.model = {}    # �洢ģ��
        self.version = {}  # �洢�汾

    def register(self, model, name, version='x'):  # ����ע��ģ�͡�������ģ�͡����ƺͰ汾��Ϊ�������������Ǵ洢��model��version�ֵ���
        self.model[name] = model
        self.version[name] = version

    def __call__(self, cfg, verbose=True):  # ʹ�����ʵ����������һ��������
    # cfg��һ�����ö��󣬿��ܰ���ģ�͵ĸ���������Ϣ��verbose��һ������ֵ�����ڿ����Ƿ������ϸ��Ϣ
        """
        Construct model based on the config. 
        """
        t = cfg.type

        # the register is in each file
        # ������t��ֵ����̬�ص��벻ͬ��ģ����࣬����֧�ֶ��ֲ�ͬ��ģ������
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
        
        # Ԥ������Ĳ�����Ȼ��ʹ����Щ����������ģ��
        args = preprocess_model_args(cfg.args)
        net = self.model[t](**args)

        return net

    def get_version(self, name):
        return self.version[name]

def register(name, version='x'): # ����һ��װ����
    def wrapper(class_):
        get_model().register(class_, name, version)  # ���װ��������һ������Ϊ��������ע�������
        return class_
    return wrapper
