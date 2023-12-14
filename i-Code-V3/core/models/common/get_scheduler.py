import torch
import torch.optim as optim
import numpy as np
import copy
from ... import sync
from ...cfg_holder import cfg_unique_holder as cfguh

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_scheduler(object):
    def __init__(self):
        self.lr_scheduler = {}   # 在类的实例中初始化一个空的字典，用于存储学习率调度器。

    def register(self, lrsf, name):  # 将学习率调度器（lrsf）注册到lr_scheduler字典中，并使用给定的名称（name）
        self.lr_scheduler[name] = lrsf

    def __call__(self, cfg):
        if cfg is None:
            return None
        if isinstance(cfg, list):  # 若cfg是列表
            schedulers = []
            for ci in cfg:
                t = ci.type
                # 从lr_scheduler字典中获取相应的学习率调度器，并使用传入的参数（**ci.args）进行初始化
                # 将所有的调度器存储在schedulers列表中
                schedulers.append(      
                    self.lr_scheduler[t](**ci.args)) 
            if len(schedulers) == 0:
                raise ValueError
            else:
                return compose_scheduler(schedulers)  # 返回由这些调度器组成的组合调度器
        # cfg不是列表，则直接使用该类型获取相应的学习率调度器，并传入参数初始化
        t = cfg.type  
        return self.lr_scheduler[t](**cfg.args)  
        

def register(name):
    def wrapper(class_):
        get_scheduler().register(class_, name)
        return class_
    return wrapper

class template_scheduler(object):
    def __init__(self, step):
        self.step = step

    def __getitem__(self, idx): # 保留了一个模板接口，但不允许直接通过索引访问
        raise ValueError

    def set_lr(self, optim, new_lr, pg_lrscale=None):
        # 三个参数：优化器对象（optim），新的学习率（new_lr），可选的参数组学习率缩放比例字典（pg_lrscale）
        """
        Set Each parameter_groups in optim with new_lr
        New_lr can be find according to the idx.
        pg_lrscale tells how to scale each pg.
        """
        # new_lr = self.__getitem__(idx)
        pg_lrscale = copy.deepcopy(pg_lrscale)
        for pg in optim.param_groups:
            # 检查参数组pg是否存在pg_lrscale
            if pg_lrscale is None:
                pg['lr'] = new_lr  # 直接将新的学习率设置为该参数组的学习率
            else:
                # 若存在，使用该参数组的名称从pg_lrscale中获取相应的缩放比例，并据此设置该参数组的学习率
                pg['lr'] = new_lr * pg_lrscale.pop(pg['name'])
        # 方法检查pg_lrscale是否为空或者是否已经没有更多的键值对，若不是则抛出错误
        assert (pg_lrscale is None) or (len(pg_lrscale)==0), \
            "pg_lrscale doesn't match pg"

@register('constant')
class constant_scheduler(template_scheduler):
    def __init__(self, lr, step):
        super().__init__(step)  # 使用父函数
        self.lr = lr

    def __getitem__(self, idx):  # idx是要获取值的索引
        if idx >= self.step:
            raise ValueError
        return self.lr

# 以下两个类都是为了在训练深度学习模型时根据指定的步数逐步调整学习率
# 'poly'类型的调度器按照幂次进行衰减，而'linear'类型的调度器则线性地进行衰减
@register('poly')
class poly_scheduler(template_scheduler):
    def __init__(self, start_lr, end_lr, power, step):
        super().__init__(step)     # 步数
        self.start_lr = start_lr   # 起始学习率
        self.end_lr = end_lr       # 结束学习率
        self.power = power         # 幂次

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        a, b = self.start_lr, self.end_lr
        p, n = self.power, self.step
        return b + (a-b)*((1-idx/n)**p)   # 计算并返回给定索引位置的学习率

@register('linear')
class linear_scheduler(template_scheduler):
    def __init__(self, start_lr, end_lr, step):
        super().__init__(step)
        self.start_lr = start_lr
        self.end_lr = end_lr

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        a, b, n = self.start_lr, self.end_lr, self.step
        return b + (a-b)*(1-idx/n)

# constant_scheduler类是一个多阶段学习率调度器，它根据给定的里程碑逐步调整学习率。
# compose_scheduler类是一个组合调度器，它允许将多个调度器组合在一起，并根据每个调度器的步数里程碑来调整学习率。
@register('multistage')
class constant_scheduler(template_scheduler):
    def __init__(self, start_lr, milestones, gamma, step):  # milestones:里程碑，gamma:衰减因子
        super().__init__(step)
        self.start_lr = start_lr
        m = [0] + milestones + [step] # 创建一个列表，其中包含步数、里程碑和开始步数
        lr_iter = start_lr  # 初始化学习率迭代器为起始学习率
        self.lr = []        # 初始化一个空列表，用于存储每个阶段的学习率
        # m[0:-1]：从 m 列表的第一个元素开始，直到倒数第二个元素
        # zip 函数会将两个列表的对应元素打包成一个元组
        # 将每个元组的第一个元素赋值给 ms，第二个元素赋值给 me
        for ms, me in zip(m[0:-1], m[1:]): 
            for _ in range(ms, me):  # 遍历每个里程碑之间的步数
                self.lr.append(lr_iter)  
            lr_iter *= gamma            # 更新学习率迭代器

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        return self.lr[idx]

class compose_scheduler(template_scheduler):
    def __init__(self, schedulers):
        self.schedulers = schedulers  # 存储传入的调度器列表
        self.step = [si.step for si in schedulers]  # 创建一个步数列表，其中包含每个调度器的步数
        self.step_milestone = []    #  初始化一个空列表，用于存储累计步数里程碑
        acc = 0   # 初始化一个累加器为0
        for i in self.step:
            acc += i    # 累加步数
            self.step_milestone.append(acc)
        self.step = sum(self.step)

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        ms = self.step_milestone
        # zip 函数将这两个列表中的元素配对，然后 enumerate 函数为每一对元素提供索引
        for idx, (mi, mj) in enumerate(zip(ms[:-1], ms[1:])):
            if mi <= idx < mj:
                return self.schedulers[idx-mi] # 使用相对索引
        raise ValueError

####################
# lambda schedular #
####################

class LambdaWarmUpCosineScheduler(template_scheduler):
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, 
                 base_lr,
                 warm_up_steps, 
                 lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
    # base_lr：基础学习率。warm_up_steps：预热步数，在这个阶段，学习率从0逐渐增加到最大值。
    # lr_min, lr_max：学习率的上下限。lr_start：学习率开始的值。
    # max_decay_steps：学习率衰减的最大步数。verbosity_interval：用于控制日志输出的间隔
        cfgt = cfguh().cfg.train  # 从某个配置工具cfguh()中获取训练配置
        bs = cfgt.batch_size      # 从配置中提取批量大小（batch size）
        if 'gradacc_every' not in cfgt:
            print('Warning, gradacc_every is not found in xml, use 1 as default.')
        # 使用get方法尝试从配置中获取'gradacc_every'的值。如果该键不存在，则返回默认值1，并将其存储在acc变量中
        acc = cfgt.get('gradacc_every', 1)
        self.lr_multi = base_lr * bs * acc   # 计算学习率乘数。基础学习率（base_lr）乘以批量大小（bs）再乘以梯度累积间隔（acc）
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.   # 初始化上一个学习率为0
        self.verbosity_interval = verbosity_interval

    def schedule(self, n):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:  # n是当前步数
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start # 使用线性插值计算学习率
            self.last_lr = lr
            return lr
        else:  # 否则，使用余弦衰减的方式计算学习率
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __getitem__(self, idx):
        return self.schedule(idx) * self.lr_multi

# 支持重复的迭代，并且这个迭代次数是可配置的
class LambdaWarmUpCosineScheduler2(template_scheduler):
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """
    def __init__(self, 
                 base_lr,
                 warm_up_steps, 
                 f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
    # f_min, f_max, f_start: 频率的最小值、最大值和开始值列表。
    # base_lr: 基础学习率。cycle_lengths: 每个循环的长度列表
        cfgt = cfguh().cfg.train
        # bs = cfgt.batch_size
        # if 'gradacc_every' not in cfgt:
        #     print('Warning, gradacc_every is not found in xml, use 1 as default.')
        # acc = cfgt.get('gradacc_every', 1)
        # self.lr_multi = base_lr * bs * acc
        self.lr_multi = base_lr
        # 确保所有传入的列表参数长度相同：
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths)) # 第一个元素是0，计算循环的累积长度
        self.last_f = 0.
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0   # 记录n所在的区间
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n):
        cycle = self.find_in_interval(n) # 确定当前的步数 n 落在哪个累积循环区间内
        n = n - self.cum_cycles[cycle]   # 从当前的步数 n 中减去当前循环区间的累积步数，得到在当前循环区间内的相对步数
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f 
        else:  # 余弦退火算法
            # 时间比例 t
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0) 
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                    1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __getitem__(self, idx):
        return self.schedule(idx) * self.lr_multi

# 用于根据当前的步数 n 来计算并返回适当的学习率
@register('stable_diffusion_linear')
class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    def schedule(self, n):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: 
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                      f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f