import torch
import torch.optim as optim
import numpy as np
import itertools

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

class get_optimizer(object):  # ���ݸ�����������Ϣ��̬�ع����Ż���
    def __init__(self):
        self.optimizer = {}   # optimizer���ڴ洢ע����Ż������ͼ����Ӧ���࣬register���ڽ��Ż������ͺͶ�Ӧ�������ע��
        self.register(optim.SGD, 'sgd')  # ��ʼ��SGD�Ż���
        self.register(optim.Adam, 'adam')
        self.register(optim.AdamW, 'adamw')

    def register(self, optim, name):  # ���Ż���������Ӧ���Ż������Ƶ�optimizer�ֵ���
        self.optimizer[name] = optim

    def __call__(self, net, cfg): # ����ģ��net��������Ϣcfg
        if cfg is None:
            return None
        t = cfg.type
        # �������ģ����DataParallel��DistributedDataParallel��ʵ��(�ֲ�ʽ����ʽ������ģ��)�����ȡ���ڲ�ģ��(ͨ��module����)
        if isinstance(net, (torch.nn.DataParallel,                           
                            torch.nn.parallel.DistributedDataParallel)):
            netm = net.module
        else:  #����ֱ��ʹ��ԭʼ������ģ��
            netm = net
        pg = getattr(netm, 'parameter_group', None)  # ��ȡ����ģ�͵Ĳ����飨ͨ�����parameter_group����)

        if pg is not None:
            params = []
            for group_name, module_or_para in pg.items():  # �����������еļ�ֵ��
            # group_name�Ǽ�����ʾ����������ƣ�module_or_para��ֵ����ʾģ������
                if not isinstance(module_or_para, list): # ���module_or_para����һ���б�˵����������һ��������ģ������
                    module_or_para = [module_or_para]    # ����ת��Ϊһ�������б�

                # �������һ��ģ�飨������torch.nn.Module��ʵ����,������Ϊһ�������б�.����,���ȡ�����
                grouped_params = [mi.parameters() if isinstance(mi, torch.nn.Module) else [mi] for mi in module_or_para]
                grouped_params = itertools.chain(*grouped_params) # ����Щ������ϳ�һ������
                # �洢��һ���ֵ��У����м��ǲ���������ƣ�ֵ�ǲ����б�
                pg_dict = {'params':grouped_params, 'name':group_name}
                params.append(pg_dict)
        else:
            params = net.parameters()  # ��������ڲ����飬��ֱ�ӻ�ȡ����ģ�͵Ĳ���
        # �����Ż�������t��optimizer�ֵ��л�ȡ��Ӧ���Ż����࣬��ʹ�û�ȡ�Ĳ����б��������Ϣ�е��������������Ż���
        return self.optimizer[t](params, lr=0, **cfg.args)  # �ֵ���ԣ��ٴ��������ʼ��
