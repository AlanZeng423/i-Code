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

class get_optimizer(object):  # 根据给定的配置信息动态地构造优化器
    def __init__(self):
        self.optimizer = {}   # optimizer用于存储注册的优化器类型及其对应的类，register用于将优化器类型和对应的类进行注册
        self.register(optim.SGD, 'sgd')  # 初始化SGD优化器
        self.register(optim.Adam, 'adam')
        self.register(optim.AdamW, 'adamw')

    def register(self, optim, name):  # 将优化器类存入对应的优化器名称的optimizer字典中
        self.optimizer[name] = optim

    def __call__(self, net, cfg): # 网络模型net和配置信息cfg
        if cfg is None:
            return None
        t = cfg.type
        # 如果网络模型是DataParallel或DistributedDataParallel的实例(分布式或并行式的网络模型)，则获取其内部模块(通过module属性)
        if isinstance(net, (torch.nn.DataParallel,                           
                            torch.nn.parallel.DistributedDataParallel)):
            netm = net.module
        else:  #否则直接使用原始的网络模型
            netm = net
        pg = getattr(netm, 'parameter_group', None)  # 获取网络模型的参数组（通过检查parameter_group属性)

        if pg is not None:
            params = []
            for group_name, module_or_para in pg.items():  # 遍历参数组中的键值对
            # group_name是键，表示参数组的名称；module_or_para是值，表示模块或参数
                if not isinstance(module_or_para, list): # 如果module_or_para不是一个列表，说明它可能是一个单独的模块或参数
                    module_or_para = [module_or_para]    # 将其转换为一个参数列表

                # 如果不是一个模块（即不是torch.nn.Module的实例）,将其作为一个参数列表.否则,则获取其参数
                grouped_params = [mi.parameters() if isinstance(mi, torch.nn.Module) else [mi] for mi in module_or_para]
                grouped_params = itertools.chain(*grouped_params) # 将这些参数组合成一个链表
                # 存储在一个字典中，其中键是参数组的名称，值是参数列表
                pg_dict = {'params':grouped_params, 'name':group_name}
                params.append(pg_dict)
        else:
            params = net.parameters()  # 如果不存在参数组，则直接获取网络模型的参数
        # 根据优化器类型t从optimizer字典中获取对应的优化器类，并使用获取的参数列表和配置信息中的其他参数构造优化器
        return self.optimizer[t](params, lr=0, **cfg.args)  # 字典配对，再传入参数初始化
