import numpy as np
import torchvision.utils as vutils
import torch
import torch.nn.functional as F
import random
import torch.distributed as dist
from matplotlib import cm
import matplotlib.pyplot as plt


# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
# 装饰器函数make_recursive_func接受一个函数作为参数，并返回一个新的函数wrapper.
# 这个新函数可以递归地处理列表、元组和字典类型的变量，对于其他类型的变量则直接调用原始函数进行处理.
# 这个装饰器可以用于将一个普通的函数转换为可以递归处理复杂数据结构的函数.
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


# tensor2float()函数将传递给它的vars从PyTorch张量（Tensor）类型转换为Python浮点数（float）类型.
@make_recursive_func
def tensor2float(vars):
    # 如果输入vars本身就是Python浮点数，则直接返回该浮点数
    if isinstance(vars, float):
        return vars
    # 如果输入是PyTorch张量，则使用.data.item()方法提取出张量中的一个标量值，并返回该标量值。
    elif isinstance(vars, torch.Tensor):
        # vars是一个PyTorch张量（Tensor）对象,vars.data是该张量的底层数据（underlying data）部分,
        # .item()是一个PyTorch张量对象的方法,它可以将张量中的一个标量值提取出来,返回一个Python标量（scalar）.
        # vars.data.item()的作用是将张量vars中的一个标量值提取出来,并返回一个Python标量.
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


# tensor2numpy()函数将传递给它的vars从PyTorch张量（Tensor）类型转换为NumPy数组（numpy.ndarray）类型.
@make_recursive_func
def tensor2numpy(vars):
    # 如果输入参数本身就是NumPy数组，则直接返回该数组；
    if isinstance(vars, np.ndarray):
        return vars
    # 如果输入参数是PyTorch张量，则对其执行一系列操作，将其转换为NumPy数组，然后返回该数组.
    elif isinstance(vars, torch.Tensor):
        # 在转换PyTorch张量时，先使用.detach()方法分离张量中的梯度（gradient），然后使用.cpu()方法将其移到CPU上，
        # 最后使用.numpy()方法将其转换为NumPy数组类型。
        # .copy()方法用于创建该数组的副本，以避免可能的内存共享和潜在的数据不一致性问题。
        return vars.detach().cpu().numpy().copy()
    else:
        # 如果传递给该函数的参数不是NumPy数组或PyTorch张量，则会抛出一个NotImplementedError异常.
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):  # 将输入的变量vars放在GPU上
    if isinstance(vars, torch.Tensor):  # 如果输入是Tensor，则将其放在GPU上并返回该Tensor
        return vars.cuda()
    elif isinstance(vars, str):  # 如果输入是字符串，则直接返回该字符串
        return vars
    else:  # 如果输入的不是这两种类型，函数会抛出一个错误
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


# save_scalars()将标量数据写入TensorBoard日志
def save_scalars(logger, mode, scalar_dict, global_step):
    # tensor2float函数将scalar_dict中的所有PyTorch张量转换为Python浮点数
    scalar_dict = tensor2float(scalar_dict)
    # 遍历字典中的每个键值对
    for key, value in scalar_dict.items():
        # 如果值不是一个列表或元组,则将键和值传递给TensorBoard日志记录器的add_scalar方法,以将标量数据写入日志.
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        # 如果值是一个列表或元组,则遍历其中的每个元素,并使用模式、键、索引分别构造一个新的名称,
        # 然后将元素值传递给add_scalar方法,以将所有标量数据写入日志.
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


# save_images函数用于将图像数据写入TensorBoard日志.
# 该函数接受四个参数: logger是一个TensorBoard日志记录器对象; mode是一个字符串,表示要记录的模式（例如，训练模式、验证模式等）;
# images_dict是一个字典,其中包含需要记录的图像数据;global_step是一个整数,表示当前步数.
def save_images(logger, mode, images_dict, global_step):
    # 首先调用tensor2numpy函数将images_dict中的所有PyTorch张量转换为NumPy数组
    images_dict = tensor2numpy(images_dict)

    # preprocess函数用于将每个图像数据预处理成适合写入TensorBoard日志的格式.
    def preprocess(name, img):
        # 首先检查输入图像的形状是否符合要求,如果不符合,则抛出一个NotImplementedError异常,并提供一个包含图像名称和形状的错误信息.
        if not (len(img.shape) == 3 or len(img.shape) == 4):  # (B, H, W) 或 (B, C, H, W) 或 (B, N, H, W)
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:  # (B, H, W)
            # np.newaxis是NumPy库中的一个特殊常量,表示在当前维度插入一个新的维度,从而增加数组的维数
            # 将img数组从形状(B, H, W)转换为形状(B, 1, H, W)
            img = img[:, np.newaxis, :, :]
        # torch.from_numpy()函数将NumPy数组转换为PyTorch张量
        # 切片操作img[:1]是为了取B中的第一个, (B, C, H, W)==>(1, C, H, W)/(B, N, H, W)==>(1, N, H, W)或者(B, 1, H, W)==>(1, 1, H, W)
        img = torch.from_numpy(img[:1])
        # vutils.make_grid作用是将若干幅图像拼成一幅图像,输入tensor的形状应该是(B,C,H,W)
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


# DictAverageMeter类用于计算字典中每个键的平均值
class DictAverageMeter(object):
    # 在初始化方法中，self.data保存了一个空字典，用于存储每个键的值，并将self.count初始化为0，用于统计输入数据的数量。
    def __init__(self):
        self.data = {}
        self.count = 0

    # 在update方法中，new_input是一个字典类型的输入数据，用于更新self.data中每个键的值。
    # 如果输入数据中某个键的值不是浮点数类型，将会抛出一个NotImplementedError异常。
    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                # 将输入数据的值赋值给键对应的值
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                # 累加每个键的值
                self.data[k] += v

    # 计算平均值
    def mean(self):
        # items()函数用于返回一个包含字典中所有键值对的元组列表
        return {k: v / self.count for k, v in self.data.items()}


# 该函数的主要作用是包装指定的度量函数,以使其能够逐个计算每个图像的度量，并将所有图像的度量平均
# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        # 该函数使用一个循环逐个计算BATCH中每个图像的度量,并将结果存储在一个列表中. 最后,将结果列表中的所有度量平均,并返回平均值.
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())


# 与compute_metrics_for_each_image的区别：本函数接收额外的参数thres,并使用每张图像阈值来计算度量值。
# a wrapper to compute metrics for each image individually
def compute_batch_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, thres):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], thres[idx])
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_batch_metrics_for_each_image
def Batch_Thres_metrics(depth_est, depth_gt, mask, thres):
    # assert isinstance(thres, (int, float))
    # 检查掩码是否全为False,即整个图像都是无效区域.
    # torch.logical_not()是PyTorch中的逐元素逻辑非运算符,用于计算张量中每个元素的逻辑非值.它将True变为False,将 False变为True.
    # torch.all()是PyTorch中的张量逻辑运算函数,用于判断张量中的所有元素是否都满足某个条件.如果所有元素都满足条件,则返回True,否则返回False.
    if torch.all(torch.logical_not(mask)):
        # 如果整个图像都是无效区域,则返回张量0.0
        error = torch.tensor(0.0, device=depth_est.device)
    else:
        # 根据掩码选择有效区域的估计深度图和真实深度图
        depth_est, depth_gt = depth_est[mask], depth_gt[mask]
        # 计算每个像素的绝对误差
        errors = torch.abs(depth_est - depth_gt)
        # 根据误差是否大于阈值计算掩码
        err_mask = errors > thres
        # 计算所有误差大于阈值的像素占总像素数的比例，作为整个图像的度量值
        error = torch.mean(err_mask.float())
    return error


# # AbsDepthError_metrics()函数用于计算深度估计和深度真值之间的绝对误差
# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask):
    # 判断输入的掩码中是否所有像素都无效
    if torch.all(torch.logical_not(mask)):
        # 如果所有像素都无效，则返回误差为 0.0 的张量
        error = torch.tensor(0.0, device=depth_est.device)
    else:  # 如果掩码中存在有效像素
        # 从深度估计和深度真值张量中仅选择掩码对应位置为True的像素
        depth_est, depth_gt = depth_est[mask], depth_gt[mask]
        # 计算深度估计和深度真值之间的绝对误差,然后计算这些绝对误差的平均值
        error = torch.mean((depth_est - depth_gt).abs())
    return error


# 在使用分布式训练时,对所有进程进行同步(barrier)操作
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    # 判断当前环境是否支持分布式训练（通过调用dist.is_available()函数判断）
    if not dist.is_available():
        return
    # 判断当前进程组是否已经初始化（通过调用dist.is_initialized()函数判断）
    if not dist.is_initialized():
        return
    # 获取当前进程组的进程数（通过调用dist.get_world_size()函数）
    world_size = dist.get_world_size()
    # 如果进程数为1，则说明只有一个进程在运行，不需要同步，直接返回。
    if world_size == 1:
        return
    # 调用dist.barrier()函数进行同步操作，即等待所有进程都到达同步点后再继续执行后面的代码。
    dist.barrier()


# 函数的返回值代表了当前进程组的进程数。如果只有一个进程在运行，返回值为1。
def get_world_size():
    # 首先判断当前环境是否支持分布式训练（通过调用dist.is_available()函数判断），如果不支持，则返回1。
    if not dist.is_available():
        return 1
    # 如果支持分布式训练，则判断当前进程组是否已经初始化（通过调用dist.is_initialized()函数判断），如果没有初始化，则返回1。
    if not dist.is_initialized():
        return 1
    # 如果已经初始化，则调用dist.get_world_size()函数获取进程组的进程数，并返回该值。
    return dist.get_world_size()


def reduce_scalar_outputs(scalar_outputs):
    # 首先获取当前进程组的进程数
    world_size = get_world_size()
    # 如果进程数小于2,则直接返回输入的scalar_outputs字典
    if world_size < 2:
        return scalar_outputs
    with torch.no_grad():
        names = []
        scalars = []
        # 如果进程数大于等于2,将scalar_outputs字典中的key和value分别存储到names和scalars列表中
        for k in sorted(scalar_outputs.keys()):
            names.append(k)
            scalars.append(scalar_outputs[k])
        # 将scalars列表中的元素在第0个维度上堆叠为一个张量scalars
        scalars = torch.stack(scalars, dim=0)
        # dist.reduce(tensor, dst, op=ReduceOp.SUM, group=None)是PyTorch中分布式通信接口dist提供的一个函数,
        # 用于对指定的张量在所有进程中进行全局约简(reduce)操作,并将结果发送到指定的进程dst.
        # 参数tensor是要进行全局约简操作的张量,dst指定了目标进程的rank,op指定了全局约简的操作.
        # ReduceOp.SUM是一种分布式数据约简（reduction）操作，在分布式环境下可以在多个进程/节点中对张量进行累加.
        # 在分布式训练中，通常会在每个进程/节点上计算损失函数的值，然后使用 ReduceOp.SUM 将这些值累加起来，得到全局的损失函数值，
        # 用于计算梯度更新。这种方法可以使得每个进程/节点在执行训练时只需计算部分数据，从而加速训练过程，同时又可以获得全局的损失函数值。
        # 通过调用dist.reduce()函数对所有进程中的scalars进行全局归约,将结果存储在rank=0的进程中.
        dist.reduce(scalars, dst=0)
        # 在rank=0的进程中，将全局归约结果除以进程数，并将结果存储到一个新的字典中返回。
        # 由于只有rank=0的进程获得了全局归约结果，因此在其它进程中函数将返回空字典。
        # 函数的输入参数是一个字典scalar_outputs，包含了多个标量输出值，输出也是一个字典，包含了归约后的标量输出值。
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by world_size in this case
            scalars /= world_size
        reduced_scalars = {k: v for k, v in zip(names, scalars)}

    return reduced_scalars


def set_random_seed(seed):
    random.seed(seed)  # Python内置库random设置相应的随机数种子
    np.random.seed(seed)  # numpy设置相应的随机数种子
    torch.manual_seed(seed)  # PyTorch的torch.manual_seed设置CPU的随机数种子
    torch.cuda.manual_seed_all(seed)  # PyTorch的torch.cuda.manual_seed_all设置GPU的随机数种子


# metric from MVSNet_pl
def abs_error_pl(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    return (depth_pred - depth_gt).abs()


# metric from MVSNet_pl
def acc_threshold_pl(depth_pred, depth_gt, mask, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    errors = abs_error_pl(depth_pred, depth_gt, mask)
    acc_mask = errors < threshold
    return acc_mask.float().mean()


@make_nograd_func
@compute_metrics_for_each_image
def ICVNet_accu(pred_label, gt_label, mask):
    # pred_label: B, H, W
    # gt_label: B, H, W
    # mask: B, H, W

    # 如果掩码全为False，说明没有有效像素，准确率为0。
    if torch.all(torch.logical_not(mask)):
        accu = torch.tensor(0.0, device=pred_label.device)
    else:
        # pred_label[mask]和gt_label[mask]是通过掩码提取出来的有效像素部分,其形状为(N,),其中N是有效像素的数量。
        # torch.eq比较两个张量的元素是否相等,返回一个布尔值的张量,float()将布尔值的张量转化为浮点数的张量。
        # 然后使用torch.mean计算平均值，即为准确率。
        accu = torch.mean(torch.eq(pred_label[mask], gt_label[mask]).float())
    return accu


# 计算每张图片在给定mask下概率图像素的均值
# NOTE: please do not use this to build up training loss
@make_nograd_func
def Prob_mean(prob_map, mask):
    batch_size = prob_map.shape[0]
    results = []
    # compute result one by one
    # 通过循环计算每张图片在给定掩码下概率图像素的均值，将其加入到结果列表中
    for idx in range(batch_size):
        ret = torch.mean(prob_map[idx][mask[idx]])
        results.append(ret)
    # 将结果列表的张量堆叠成一个新张量，并计算所有元素的均值作为输出
    return torch.stack(results).mean()


# 对输入的图像进行颜色映射，返回彩色图像的Tensor
@make_nograd_func
def mapping_color(img, vmin, vmax, cmap="rainbow"):
    batch_size = img.shape[0]  # B, H, W
    results = []
    # compute result one by one
    # 遍历每个图像进行颜色映射
    for idx in range(batch_size):
        # 对于每张图像，将其转换为numpy数组
        np_img = img[idx].cpu().numpy()
        # 如果vmin和vmax不为None,根据vmin和vmax进行归一化
        if vmin is not None and vmax is not None:
            np_img = plt.Normalize(vmin=vmin[idx].item(), vmax=vmax[idx].item())(np_img)
        # 调用matplotlib.cm中的相应颜色映射函数,将numpy数组转换为RGB颜色图像。
        # getattr(cm, cmap)的作用是获取cmap所代表的颜色映射函数对象
        mapped_img = getattr(cm, cmap)(np_img)
        results.append(mapped_img[:, :, 0:4])
    # 最后将所有彩色图像的Tensor拼接起来,并将通道维度（RGB）放到第二维,即返回大小为[B, 3, H, W]的Tensor。
    results = torch.tensor(np.stack(results), device=img.device)
    results = results.permute(0, 3, 1, 2)
    return results


# 将一个列表L划分为n个近似相等大小的子列表。函数返回一个包含n个子列表的列表，其中每个子列表的长度大致相等。
# 如果n不能整除L的长度，那么前几个子列表可能会比后面子列表稍长一些。
# 当verbose参数为True时，函数会输出一些有关划分结果的信息。
def chunk_list(L, n=1, verbose=False):
    '''
    Partition list L into n chunks.
    
    Returns a list of n lists/chunks, where each chunk is 
    of nearly equal size.
        >>> L = 'a b c d'.split(" ")
        ['a', 'b', 'c', 'd']
        >>> chunk(L, 2)
        [['a', 'b'], ['c', 'd']]
        >>> chunk(L, 3)
        [['a', 'b'], ['c'], ['d']]
        >>> chunk(L, 4)
        [['a'], ['b'], ['c'], ['d']]
        >>> chunk(L, 5)
        [['a'], ['b'], ['c'], ['d'], []]
    '''
    # 获取列表L的总长度
    total = len(L)
    if n > 0:
        # 计算每个子列表的大致长度size
        size = total // n
        # 计算多出来的元素数量rest
        rest = total % n
        ranges = []
        if verbose:
            msg = "{} items to be split into {} chunks of size {} with {} extra"
            print(msg.format(total, n, size, rest))
        if not size:  # size == 0 表示将列表L分成的块数n大于列表长度total
            # 在 [[x] for x in L] 的基础上再添加 n - total 个空子列表
            return [[x] for x in L] + [[] for i in range(n - total)]

        if rest:
            index = [x for x in range(0, total, size)]
            # [index[i] + i for i in range(rest + 1)] 表示将原列表index的前rest + 1项分别加上它们的下标i，生成一个新的列表，
            # 例如当rest = 2时，对于原列表index = [0, 3, 6, 9, 12]，生成的新列表为[0, 4, 8]。

            # index[rest+1:][:n-rest]表示两次切片,从index列表的第rest+1个元素开始一直到列表末尾的所有元素,再切片取前n-rest个元素
            # 例如，假设index = [0, 5, 10, 15, 20]，n=4，size=4，total=16，rest=0，则index[rest+1:][:n-rest]就是[5, 10, 15]，
            # 即切分后前三个子列表的起始位置。接着，[x + rest for x in index[rest+1:][:n-rest]]表示将这n-rest个索引值都加上rest，
            # 例如上面的例子就变成了[5, 10, 15]加上rest=1变成[6, 11, 16]。
            extra = [index[i] + i for i in range(rest + 1)] + [x + rest for x in index[rest+1:][:n-rest]]
            ranges = [(extra[i], extra[i+1]) for i in range(len(extra) - 1)]
        else:
            index = [x for x in range(0, total+1, size)]
            ranges = [(index[i], index[i+1]) for i in range(len(index) - 1)]
        return [L[i:j] for i, j in ranges]
