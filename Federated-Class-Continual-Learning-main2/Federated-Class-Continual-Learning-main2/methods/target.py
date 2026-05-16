import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
from Loss.losses import SupConLoss
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb
from abc import ABC
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from kornia import augmentation
import time, os, math
import torch.nn.init as init
from PIL import Image
from utils.empirical_feature_matrix import EmpiricalFeatureMatrix
from utils.toolkit import combine_data

# 根据指定的数据集配置训练参数
dataset = "cifar10"

# 如果数据集为cifar100，配置相应的训练参数
if dataset == "cifar10":
    # 合成批次大小
    synthesis_batch_size = 256
    # 样本批次大小
    sample_batch_size = 256
    # GAN生成器的步数
    g_steps = 50
    # 是否使用MAML算法
    is_maml = 1
    # Knowledge Distillation步数
    kd_steps = 400
    # 预热周期数
    warmup = 100
    # GAN生成器学习率
    lr_g = 0.002
    # 输入噪声学习率
    lr_z = 0.01
    # One-hot编码比例
    oh = 1
    # 温度参数
    T = 20.0
    # 活动性正则化系数
    act = 0.0
    # 对抗性训练强度
    adv = 1.0
    # 是否使用硬样本损失
    hard_loss_weight = 0.1
    # 样本原型对其损失权重
    proto_loss_weight = 0
    # 批量归一化缩放因子
    bn = 10.0
    # 是否重置l0层
    reset_l0 = 1
    # 是否重置批量归一化层
    reset_bn = 0
    # 批量归一化动量
    bn_mmt = 0.9
    # 合成轮数
    syn_round = 50
    # 软标签平滑参数
    tau = 1
    # 数据归一化参数
    data_normalize = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

    # 数据增强策略
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(**dict(data_normalize)),  # 归一化
    ])
else:  # 如果数据集不是cifar100，使用另一组配置
    # 合成批次大小（与cifar100相同）
    synthesis_batch_size = 256
    # 样本批次大小（与cifar100相同）
    sample_batch_size = 256
    # GAN生成器的步数
    g_steps = 100
    # 不使用MAML算法
    is_maml = 1
    # Knowledge Distillation步数（与cifar100相同）
    kd_steps = 400
    # 预热周期数（与cifar100相同）
    warmup = 100
    # GAN生成器学习率（降低）
    lr_g = 0.0002
    # 输入噪声学习率（与cifar100相同）
    lr_z = 0.01
    # One-hot编码比例（降低）
    oh = 0.1
    # 温度参数（降低）
    T = 5
    # 活动性正则化系数（与cifar100相同）
    act = 0.0
    # 对抗性训练强度（与cifar100相同）
    adv = 1.0
    # 批量归一化缩放因子（降低）
    bn = 10
    # 是否使用硬样本损失
    hard_loss_weight = 0.1
    # 样本原型对其损失权重
    proto_loss_weight = 0
    # 是否重置l0层（与cifar100相同）
    reset_l0 = 0
    # 是否重置批量归一化层（与cifar100相同）
    reset_bn = 0
    # 批量归一化动量（与cifar100相同）
    bn_mmt = 0.9
    # 合成轮数（增加）
    syn_round = 55
    # 软标签平滑参数（与cifar100相同）
    tau = 1
    # 数据归一化参数
    data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 数据增强策略
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(**dict(data_normalize)),  # 归一化
    ])


def normalize(tensor, mean, std, reverse=False):
    """
    对给定的张量进行标准化或反标准化处理。

    参数:
    - tensor: 需要标准化或反标准化的张量。
    - mean: 平均值列表，用于标准化或反标准化。
    - std: 标准差列表，用于标准化或反标准化。
    - reverse: 是否执行反标准化，默认为False，如果为True，则执行反标准化。

    返回值:
    - 返回已经标准化或反标准化的张量。
    """
    if reverse:
        # 反标准化时，计算相反的平均值和标准差
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        # 直接使用给定的平均值和标准差
        _mean = mean
        _std = std

    # 将平均值和标准差转换为与输入张量相同的设备和数据类型上的张量
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    # 应用标准化或反标准化公式
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    """
    一个用于数据标准化和反标准化的类。

    参数:
    - mean: 平均值列表，用于标准化或反标准化。
    - std: 标准差列表，用于标准化或反标准化。
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        """
        对输入的张量执行标准化或反标准化处理。

        参数:
        - x: 需要处理的张量。
        - reverse: 是否执行反标准化，默认为False，如果为True，则执行反标准化。

        返回值:
        - 返回已经处理过的张量。
        """
        return normalize(x, self.mean, self.std, reverse=reverse)


normalizer = Normalizer(**dict(data_normalize))


# 创建一个Normalizer实例，用于后续的数据标准化或反标准化操作


def _collect_all_images(nums, root, postfix=['png', 'jpg', 'jpeg', 'JPEG'], is_train=False):
    """
    收集指定根目录下所有指定后缀名的图片文件。

    参数:
    - nums: 收集的图片数量，如果为None，则收集所有图片。
    - root: 根目录路径。
    - postfix: 图片文件的后缀名列表，默认包括['png', 'jpg', 'jpeg', 'JPEG']。

    返回值:
    - images: 包含所有找到的图片文件路径的列表。
    """
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]  # 如果后缀名是以字符串形式传入，将其转换为列表格式。

    # 遍历根目录及其子目录下的所有文件
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            # 如果指定了收集的图片数量
            if nums is not None:
                files.sort()  # 对文件名进行排序
                # 下面的代码被注释掉，用于不同的数据选择策略
                # random.shuffle(files)  # 随机打乱文件顺序
                if is_train:
                    files = files[10*256:10*256+nums]# 抛弃早期数据
                    # files = files[:nums]
                else:
                    # files = files[:nums]  # 只选择前nums个文件
                    files = files[-nums:]  # 选择最后nums个文件
            # 筛选出符合后缀名要求的文件，并加入到结果列表中
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join(dirpath, f))
    return images


class DataIter(object):
    """
    数据迭代器类，用于对数据加载器进行迭代。

    参数:
    - dataloader: 数据加载器对象。
    """

    def __init__(self, dataloader):
        """
        初始化函数。

        参数:
        - dataloader: 数据加载器对象。
        """
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)  # 创建迭代器

    def next(self):
        """
        获取下一个数据批次。如果当前迭代器到达末尾，则重新初始化迭代器并获取下一个批次。

        返回值:
        - data: 下一个数据批次。
        """
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)  # 重新初始化迭代器
            data = next(self._iter)
        return data


class UnlabeledImageDataset(torch.utils.data.Dataset):
    """
    一个用于处理未标注图像数据集的类。
    参数:
    - root: 图像数据集的根目录路径。
    - transform: 对图像进行预处理的变换（可选）。
    - nums: 指定加载图像的数量（可选），若为None，则加载所有图像。
    """

    def __init__(self, root, transform=None, nums=None):
        # 初始化根目录路径，收集所有图像路径，并可选地应用变换
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(nums, self.root)  # 收集根目录下所有图像路径
        self.transform = transform

    def __getitem__(self, idx):
        # 根据索引加载图像，应用变换后返回
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        # 返回图像数量
        return len(self.images)

    def __repr__(self):
        # 返回数据集的字符串表示
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
            self.root, len(self), self.transform)


class LabeledImageDataset(torch.utils.data.Dataset):
    """
    一个用于处理标注图像数据集的类。
    参数:
    - root: 图像数据集的根目录路径。
    - labels: 与图像对应的标签列表。
    - transform: 对图像进行预处理的变换（可选）。
    - nums: 指定加载图像的数量（可选），若为None，则加载所有图像。
    """

    def __init__(self, root, labels, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(nums, self.root, is_train=True)
        self.labels = labels
        self.transform = transform
        if nums is not None:
            self.labels = [self.labels[i] for i in range(10*256,10*256+nums)]

    def __getitem__(self, idx):
        # 根据索引加载图像和对应的标签，应用变换后返回
        img = Image.open(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # 返回数据集中的图像数量
        return len(self.images)

    def __repr__(self):
        # 返回数据集的字符串表示
        return 'Labeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
            self.root, len(self), self.transform)


def pack_images(images, col=None, channel_last=False, padding=1):
    """
    将图片列表打包成一个大图片。

    参数:
    - images: 图片数据，可以是一个numpy数组，或者是包含多个图片数据的列表或元组。
    - col: 打包后图片在水平方向上的数量，默认为None，如果设置，则会根据图片数量自动计算行数。
    - channel_last: 指示图片通道位置是否在最后，默认为False，即通道在最前面。
    - padding: 图片间以及图片边缘的填充大小，默认为1。

    返回值:
    - pack: 打包后的图片数据，是一个numpy数组。
    """
    # 将输入的图片数据转换为numpy数组，并确保其维度为4（批量大小，通道，高度，宽度）
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # 将通道位置从最后调整到最前

    # 确保输入图片数据的维度正确，并且是numpy数组
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    # 获取图片的批量大小、通道数、高度和宽度
    N, C, H, W = images.shape
    # 如果未指定列数，则根据图片数量计算出列数，并据此计算行数
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    # 初始化打包后的图片数据数组
    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    # 遍历每张图片，将其放置到打包后的图片数据中
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def reptile_grad(src, tar):
    """
    实现Reptile算法的梯度计算。

    参数:
    - src: 源模型，其参数需要与目标模型进行梯度同步。
    - tar: 目标模型，其参数用于计算梯度。
    """
    # 遍历源模型和目标模型的参数，计算参数差异的梯度，并加到源模型参数的梯度上
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            p.grad = Variable(torch.zeros(p.size())).to(device)
        p.grad.data.add_(p.data - tar_p.data, alpha=67)  # 计算参数差异的梯度


def fomaml_grad(src, tar):
    """
    实现FOMAML算法的梯度计算。

    参数:
    - src: 源模型，其参数需要与目标模型进行梯度同步。
    - tar: 目标模型，其参数用于计算梯度。
    """
    # 遍历源模型和目标模型的参数，直接将目标模型参数的梯度加到源模型参数的梯度上
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            p.grad = Variable(torch.zeros(p.size())).to(device)
        p.grad.data.add_(tar_p.grad.data)  # 将目标模型的梯度直接加到源模型上


def reset_l0_fun(model):
    """
    重置模型中特定层的权重和偏置。

    参数:
    - model: 要重置的模型，期望是PyTorch的模型实例。

    该函数遍历模型的所有模块，如果模块的名称是"l1.0"或"conv_blocks.0"，则将该模块的权重初始化为正态分布，
    均值为0，标准差为0.02，并将偏置初始化为0。
    """
    for n, m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    """
    保存一批图像到指定路径。

    参数:
    - imgs: 图像数据，可以是PyTorch张量。
    - output: 输出路径，可以是单个图像的路径，如果是批处理，则为包含所有图像的文件名模板。
    - col: 如果指定，将图像以网格形式保存，这个参数指定了每行的图像数量。
    - size: 保存图像的大小。如果是一个数值，则按照长边或短边缩放图像使其满足该大小；如果是一个列表或元组，则直接使用该大小进行缩放。
    - pack: 是否将图像打包成一个大图像再保存，默认为True。

    返回值:
    无。
    """
    if isinstance(imgs, torch.Tensor):
        # 将图像数据从张量转换为numpy数组，并调整到[0, 1]范围，转换为uint8类型。
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    # 确保输出目录存在。
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        # 打包图像为一个大图像。
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            # 根据指定的size缩放图像。
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        # 逐个保存图像。
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + '-%d.png' % (idx))


class DeepInversionHook():
    '''
    实现前向钩子，以追踪特征统计信息并计算其上的损失。
    将计算均值和方差，并使用L2作为损失函数。

    参数:
    - module: 要注册钩子的模块。
    - mmt_rate: 迁移率，用于更新模块的运行均值和方差。
    '''

    def __init__(self, module, mmt_rate):
        # 注册前向钩子函数
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        """
        计算深度反转的特征分布正则化。
        输入:
        - module: 当前模块。
        - input: 输入到当前模块的张量。
        - output: 当前模块的输出张量。

        更新模块的运行均值和方差，以匹配特定的分布。
        """
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])  # 计算输入特征的均值
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)  # 计算输入特征的方差

        # 根据迁移率计算特征的调整距离
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        """
        更新迁移的均值和方差。
        """
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            # 根据迁移率更新均值和方差
            self.mmt = (self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                        self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data)

    def remove(self):
        """
        移除注册的钩子。
        """
        self.hook.remove()


class ImagePool(object):
    """
    图像池类，用于管理和存储图像。

    参数:
    root: 图像存储根目录的路径。
    """

    def __init__(self, root, label_root):
        """
        初始化图像池。

        参数:
        root: 图像存储的根目录路径。
        """
        # 将传入的根目录路径转换为绝对路径，并确保目录存在
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)  # 创建根目录，如果已存在则不重复创建
        self._idx = 0  # 初始化图像索引

        self.label_root = os.path.abspath(label_root)
        self.labels_path = os.path.join(self.label_root, "labels.pt")  # 定义标签存储路径
        if not os.path.exists(self.label_root):
            os.makedirs(self.label_root)
        self.labels = []  # 初始化标签列表

    def add(self, imgs, targets=None):
        """
        向图像池中添加图像。

        参数:
        imgs: 添加的图像数据，可以是单个图像或图像数据列表。
        targets: 与图像对应的标签或目标数据，默认为None。
        """
        # 保存图像批处理，每个图像以当前索引命名
        save_image_batch(imgs, os.path.join(self.root, "%d.png" % (self._idx)), pack=False)
        self._idx += 1  # 更新图像索引

        if targets is not None:
            self.labels.extend(targets.tolist())  # 保存标签
            torch.save(torch.tensor(self.labels), self.labels_path)  # 将标签保存到文件

    def get_dataset(self, nums=None, transform=None, labeled=True):
        """
        获取一个数据集，可以是标记的或未标记的图像数据集。

        参数:
        - nums: 一个可选的整数列表，指定要加载的图像的索引。如果为None，则加载所有图像。
        - transform: 一个可选的转换函数，用于对图像数据进行预处理。
        - labeled: 一个布尔值，指示返回的数据集是否带有标签。默认为True，表示返回带标签的数据集。

        返回值:
        - 返回一个UnlabeledImageDataset实例，该实例可以根据参数配置进行图像数据的加载和转换。
        """
        if labeled:
            # 确保在请求标签时已有标签数据被保存
            assert os.path.exists(self.labels_path), "Labels data not found."
            labels = torch.load(self.labels_path)
            return LabeledImageDataset(self.root, labels, transform=transform, nums=nums)
        else:
            return UnlabeledImageDataset(self.root, transform=transform, nums=nums)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)  # 存储模型参数
        self.init_size = img_size // 4  # 初始化输出图像大小
        # 第一层线性层，将输入噪声映射到一个更高维的空间
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        # 卷积块序列，用于将线性层的输出转换为图像
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.l1(z)  # 通过线性层处理输入噪声
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)  # 调整输出形状以适应卷积块输入
        img = self.conv_blocks(out)  # 通过卷积块序列生成图像
        return img

    def clone(self):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())  # 加载当前模型的状态
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return clone.to(device)  # 返回放置于GPU上的克隆模型

    def sample(self, size, device):
        # sample z
        z = torch.randn(size, self.z_dim).to(device)
        X = self.forward(z)
        return X


def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    """
    计算两个分布之间的KL散度。

    参数:
    logits - 输入的未归一化logits张量，形状为(batch_size, num_classes)。
    targets - 目标分布张量，形状与logits相同。
    T - 温度参数，用于调整分布的尖锐程度，默认值为1.0。
    reduction - 汇总方法，'batchmean'表示对批次求平均，'none'表示不进行汇总，默认为'batchmean'。

    返回值:
    KL散度的汇总值，其形状取决于reduction参数。
    """
    # 对logits应用softmax函数并除以温度T，得到q分布
    q = F.log_softmax(logits / T, dim=1)
    # 对targets应用softmax函数并除以温度T，得到p分布
    p = F.softmax(targets / T, dim=1)
    # 计算q分布和p分布之间的KL散度，并根据reduction参数进行汇总
    return F.kl_div(q, p, reduction=reduction) * (T * T)


class KLDiv(nn.Module):
    """
    KL散度计算模块，用于在模型中作为层进行KL散度的计算。

    参数:
    T - 温度参数，用于调整分布的尖锐程度，默认值为1.0。
    reduction - 汇总方法，'batchmean'表示对批次求平均，'none'表示不进行汇总，默认为'batchmean'。
    """

    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        前向传播逻辑，计算并返回输入logits和目标targets之间的KL散度。

        参数:
        logits - 输入的未归一化logits张量，形状为(batch_size, num_classes)。
        targets - 目标分布张量，形状与logits相同。

        返回值:
        KL散度的汇总值，其形状取决于构造函数中指定的reduction参数。
        """
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


class GlobalSynthesizer(ABC):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,
                 init_dataset=None, iterations=100, lr_g=0.1,
                 synthesis_batch_size=128, sample_batch_size=128,
                 adv=0.0, bn=1, oh=1,
                 save_dir='run/fast', save_label_dir='run/fast', transform=None, autocast=None, use_fp16=False,
                 normalizer=None, distributed=False, lr_z=0.01,
                 warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0,
                 is_maml=1, args=None, global_prototypes=None,
                 hard_loss_weight=hard_loss_weight, proto_loss_weight=proto_loss_weight):
        super(GlobalSynthesizer, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 初始化各种模型和参数
        self.teacher = teacher
        self.student = student
        self.save_dir = save_dir
        self.save_label_dir = save_label_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.ismaml = is_maml
        self.args = args
        self.mse_loss = nn.MSELoss(reduction="none").to(device)  # 均方误差损失函数，设置为在CPU上计算
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.normalizer = normalizer
        self.di_var_scale = 0.001
        # 创建图像池和设置生成器
        self.data_pool = ImagePool(root=self.save_dir, label_root=save_label_dir)
        self.transform = transform
        self.generator = generator.to(device).train()
        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None
        self.ie_loss = 1
        self.w_ie = 1
        self.is_bn_loss = 1
        self.global_prototypes = global_prototypes
        self.hard_loss_weight = hard_loss_weight
        self.proto_loss_weight = proto_loss_weight
        # 根据是否使用MAML，设置不同的优化器
        if self.ismaml:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g * self.iterations,
                                                   betas=[0.5, 0.999])
        else:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g * self.iterations,
                                                   betas=[0.5, 0.999])

        # 设置数据增强和批量归一化转移
        self.aug = transforms.Compose([
            augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
            augmentation.RandomHorizontalFlip(),
            normalizer,
        ])

        self.bn_mmt = bn_mmt
        self.hooks = []
        # 为教师模型中的所有批量归一化层添加深度反转钩子
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

    def synthesize(self, targets=None):
        """
        生成合成图像的过程。

        参数:
        - targets: 目标类别Tensor，如果提供，则优化生成该类别的图像；如果为None，则随机选择类别。

        返回值:
        - 无
        """
        self.ep += 1  # 增加当前episode数
        self.student.eval()  # 设置学生模型为评估模式
        self.teacher.eval()  # 设置教师模型为评估模式
        best_cost = 1e6  # 初始化最佳成本（损失）为一个较大的值
        # 在特定episode重置生成器的L0层
        if (self.ep == 120 + self.ep_start) and self.reset_l0:
            reset_l0_fun(self.generator)

        best_inputs = None

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).to(device)  # 初始化噪声张量
        z.requires_grad = True

        # 如果没有提供targets，则随机生成
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0]  # 为了更好的可视化，对目标进行排序
        targets = targets.to(device)  # 将目标移动到GPU上

        fast_generator = self.generator.clone()  # 克隆生成器用于快速优化
        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])  # 使用Adam优化器优化生成器和噪声z

        for it in range(self.iterations):
            inputs = fast_generator(z).to(device)  # 生成图像
            inputs_aug = self.aug(inputs).to(device)  # 对生成的图像进行增强和标准化

            t_out = self.teacher(inputs_aug)["logits"]  # 教师模型预测
            loss_oh = F.cross_entropy(t_out, targets)  # 计算交叉熵损失

            # 计算margin 
            # 对每个样本计算真实类别与其他类别的logit差值
            true_logits = t_out[torch.arange(len(targets)), targets]  # [batch_size]
            other_logits = t_out.clone()  # [batch_size, num_classes]
            # 将真实类别的logit设置为很小的值，这样在min操作时会被忽略
            other_logits[torch.arange(len(targets)), targets] = float('-inf')
            # 计算margin: 真实类别与最接近的错误类别的logit差值
            margins = true_logits.unsqueeze(1) - other_logits  # [batch_size, num_classes]
            delta_margin = margins.min(dim=1)[0]  # [batch_size]
            
            # 计算hard样本loss 
            epsilon_hb = 0.1  # margin阈值
            # 最小化(margin + epsilon_hb)，使样本更接近决策边界
            loss_hard = torch.mean(torch.maximum(torch.zeros_like(delta_margin), 
                                            delta_margin + epsilon_hb))

            # if targets is None:
            #     targets = torch.argmax(t_out, dim=-1)
            #     targets = targets.cuda()
            if self.is_bn_loss:
                loss_bn = sum([h.r_feature for h in self.hooks])  # 计算批归一化损失
            else:
                loss_bn = 0
            
            # 通过教师模型提取特征
            features = self.teacher.extract_vector(inputs_aug)
            # 初始化损失
            proto_loss = 0.0
            # 获取当前批次中所有样本的目标类别
            unique_targets = torch.unique(targets)
            # 批量计算所有样本与其对应原型的距离
            for target_class in unique_targets:
                # 获取当前类别的掩码
                class_mask = (targets == target_class)
                
                # 如果该类别有对应的原型
                if target_class.item() in self.global_prototypes:
                    # 获取当前类别的原型
                    prototype = self.global_prototypes[target_class.item()].to(device)
                    
                    # 选择属于当前类别的所有样本的特征
                    class_features = features[class_mask]
                    
                    # 计算该类别所有样本与其原型的L2距离
                    alpha_c = 1.0  # 类别权重
                    dist = alpha_c * torch.norm(class_features - prototype.unsqueeze(0), p=2, dim=1) ** 2
                    # 累加损失
                    proto_loss += dist.sum()
            proto_loss = proto_loss / self.synthesis_batch_size
            

            loss = self.bn * loss_bn + self.oh * loss_oh + self.hard_loss_weight * loss_hard + self.proto_loss_weight * proto_loss
            if it == 0:
                print(f"loss_bn: {loss_bn}, loss_oh: {loss_oh}, loss_hard: {loss_hard}, proto_loss: {proto_loss}")
            if self.ie_loss:
                outputs = self.teacher(inputs_aug)["logits"]
                outputs_new = outputs[:, :self.num_classes]
                softmax_o_T = F.softmax(outputs_new, dim=1).mean(dim=0)
                ie_loss = (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(self.num_classes)).sum()) * self.w_ie
                loss += ie_loss

            with torch.no_grad():
                # 更新最佳成本和最佳输入
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data.to(device)

            optimizer.zero_grad()
            loss.backward()

            # 如果是MAML算法，则在特定迭代次数进行元梯度更新
            if self.ismaml:
                if it == 0: self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations - 1): self.meta_optimizer.step()

            optimizer.step()  # 更新参数

        # 更新批归一化层的移动平均值
        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        # 如果不是MAML算法，则执行REPTILE的元梯度更新
        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator)
            self.meta_optimizer.step()

        self.student.train()  # 将学生模型设置回训练模式
        self.prev_z = (z, targets)
        end = time.time()

        self.data_pool.add(best_inputs, targets)  # 将生成的最优图像添加到数据池中
        
    
def weight_init(m):
    '''
    对模型中的各种层进行权重初始化。

    参数:
        m: 模型中的模块，可以是卷积层、批量归一化层、线性层、LSTM等。

    用法:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        # 对1D卷积层的权重使用正态分布初始化
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        # 对2D卷积层的权重使用Xavier正态分布初始化
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        # 对3D卷积层的权重使用Xavier正态分布初始化
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        # 对1D转置卷积层的权重使用正态分布初始化
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        # 对2D转置卷积层的权重使用Xavier正态分布初始化
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        # 对3D转置卷积层的权重使用Xavier正态分布初始化
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        # 对1D批量归一化层的权重使用正态分布初始化，并将偏置初始化为0
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # 对2D批量归一化层的权重使用正态分布初始化，并将偏置初始化为0
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        # 对3D批量归一化层的权重使用正态分布初始化，并将偏置初始化为0
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        # 对线性层的权重使用Xavier正态分布初始化，对偏置使用正态分布初始化
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        # 对LSTM层的参数使用正交初始化
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        # 对LSTM单元的参数使用正交初始化
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        # 对GRU层的参数使用正交初始化
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        # 对GRU单元的参数使用正交初始化
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def refine_as_not_true(logits, targets, num_classes):
    """
    对给定的logits进行精炼，将每个目标类别的logits替换为其非真实类别（not true）的logits。

    参数:
    logits - 原始logits张量，形状为(batch_size, num_classes)，表示每个样本对每个类别的隶属度概率。
    targets - 目标类别张量，形状为(batch_size,)，表示每个样本的真实类别。
    num_classes - 类别的总数。

    返回:
    精炼后的logits张量，形状为(batch_size, num_classes-1)，其中每个样本的真实类别logits被移除。
    """
    # 生成非真实类别位置的张量
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nt_positions = torch.arange(0, num_classes).to(device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    # 过滤掉目标类别，保留非目标类别
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    # 重新调整形状以便于gather操作
    nt_positions = nt_positions.view(-1, num_classes - 1)

    # 根据非真实类别位置收集logits
    logits = torch.gather(logits, 1, nt_positions)

    return logits


class TARGET(BaseLearner):
    def __init__(self, args):
        """
        初始化IncrementalModel。

        参数:
        - args: 传递给模型的参数，具体结构和内容依据实际需求而定。
        """
        super().__init__(args)
        self.generation = None
        self.kd_criterion = nn.MSELoss(reduction="none")
        self._network = IncrementalNet(args, False)
        self.is_loss_kd = 1
        self.is_feature_kd = 1
        self.is_con = 1
        self.previous_efm = None
        self.efc_lamb = 10
        self.damping = 0.1
        self.exp_name = args["exp_name"]
        # 添加全局原型存储
        self.global_prototypes = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def after_task(self):
        """
        在每个任务完成后执行的操作，主要用于更新模型的状态。
        """
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        self.save_checkpoint(self.exp_name)

    def kd_train(self, student, teacher, criterion, optimizer):
        """
        进行知识蒸馏训练。

        参数:
        - student: 学生模型，进行训练的模型。
        - teacher: 教师模型，提供指导的模型。
        - criterion: 损失函数，用于计算损失。
        - optimizer: 优化器，用于更新学生模型的参数。

        返回值:
        - 无
        """
        student.train()  # 将学生模型设置为训练模式
        teacher.eval()  # 将教师模型设置为评估模式
        loader = self.get_all_syn_data()  # 获取所有合成数据的加载器
        data_iter = DataIter(loader)  # 创建数据迭代器
        for i in range(kd_steps):  # 迭代一定次数进行知识蒸馏
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            images = data_iter.next().to(device)  # 加载下一组图像数据
            with torch.no_grad():  # 禁止计算教师模型的梯度
                t_out = teacher(images)["logits"]  # 教师模型预测
            s_out = student(images.detach())["logits"]  # 学生模型预测
            loss_s = criterion(s_out, t_out.detach())  # 计算学生模型的损失
            optimizer.zero_grad()  # 清空梯度
            loss_s.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新学生模型的参数

    def data_generation(self):
        # 初始化参数，根据数据集调整图像大小和形状
        img_shape = (3, 32, 32) if self.args["dataset"] == "cifar10" else (3, 64, 64)
        if self.args["dataset"] == "imagenet100": img_shape = (3, 128, 128)  # (3, 224, 224)
        nz = 256
        student = copy.deepcopy(self._network)
        student.apply(weight_init)  # 应用权重初始化函数到学生对象
        # 设置保存路径，并检查是否存在，不存在则创建
        tmp_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        tmp_label_dir = os.path.join(self.save_dir, "task_label_{}".format(self._cur_task))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if not os.path.exists(tmp_label_dir):
            os.makedirs(tmp_label_dir)
        # 创建全局合成器
        synthesizer = GlobalSynthesizer(copy.deepcopy(self._network), student, self.generation,
                                        nz=nz, num_classes=self._total_classes, img_size=img_shape, init_dataset=None,
                                        save_dir=tmp_dir, save_label_dir=tmp_label_dir,
                                        transform=train_transform, normalizer=normalizer,
                                        synthesis_batch_size=synthesis_batch_size,        sample_batch_size=sample_batch_size,
                                        iterations=g_steps, warmup=warmup, lr_g=lr_g, lr_z=lr_z,
                                        adv=adv, bn=bn, oh=oh,
                                        reset_l0=reset_l0, reset_bn=reset_bn,
                                        bn_mmt=bn_mmt, is_maml=is_maml, args=self.args, global_prototypes=self.global_prototypes,
                                        hard_loss_weight=hard_loss_weight, proto_loss_weight=proto_loss_weight)

        # 设置损失函数和优化器
        criterion = KLDiv(T=T)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.2, weight_decay=0.0001,
                                    momentum=0.9)
        # 设置学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=2e-4)

        # 开始生成和训练循环
        for it in range(syn_round):
            synthesizer.synthesize()  # 生成合成数据
            if it >= warmup:
                self.kd_train(student, self._network, criterion, optimizer)  # 进行知识蒸馏训练
                test_acc = self._compute_accuracy(student, self.test_loader)  # 计算并打印学生模型的测试准确率
                print("Task {}, Data Generation, Epoch {}/{} =>  Student test_acc: {:.2f}".format(
                    self._cur_task, it + 1, syn_round, test_acc, ))
                scheduler.step()  # 更新学习率
                # wandb.log({'Distill {}, accuracy'.format(self._cur_task): test_acc})

        print("For task {}, data generation completed! ".format(self._cur_task))

    def get_syn_data_loader(self):
        """
        获取合成数据的数据加载器。

        根据当前任务的参数设置（包括数据集类型、用户数量、任务数量以及本地批次大小），
        计算合成数据的迭代次数和每个迭代的批次大小。然后根据这些设置和当前任务编号，
        创建并返回一个对应合成数据集的数据加载器。

        返回:
            syn_data_loader: 合成数据的数据加载器，使用PyTorch的DataLoader实现。
        """
        # 根据数据集类型设置数据集大小
        if self.args["dataset"] == "cifar100":
            dataset_size = 50000
        elif self.args["dataset"] == "tiny_imagenet":
            dataset_size = 100000
        elif self.args["dataset"] == "cifar10":
            dataset_size = 50000
        elif self.args["dataset"] == "imagenet100":
            dataset_size = 48000
            # 计算迭代次数和每个迭代的批次大小
        iters = math.ceil(dataset_size / (self.args["num_users"] * self.args["tasks"] * self.args["local_bs"]))
        syn_bs = int(self.nums / iters)

        # 构造数据目录路径
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task - 1))
        data_label_dir = os.path.join(self.save_dir, "task_label_{}".format(self._cur_task - 1))
        labels_path = os.path.join(data_label_dir, "labels.pt")  # 定义标签存储路径

        labels = []  # 初始化标签列表
        labels = torch.load(labels_path)

        # 打印配置信息
        print("iters{}, syn_bs:{}, data_dir: {}".format(iters, syn_bs, data_dir))

        # 创建合成数据集
        syn_dataset = LabeledImageDataset(data_dir, labels, transform=train_transform, nums=self.nums)
        # 创建数据加载器
        syn_data_loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=syn_bs, shuffle=True,
            num_workers=4, pin_memory=True, )
        return syn_data_loader

    def get_all_syn_data(self):
        """
        获取当前任务的所有合成数据的加载器。

        该方法不接受任何参数。

        返回:
        - loader: 一个数据加载器，用于批量加载合成数据集中的无标签图像。
        """
        # 构造数据目录路径
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        # 创建一个无标签图像数据集，应用转换并设置样本数量
        syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform, nums=self.nums)
        # 创建一个数据加载器，用于批量加载数据，同时设置批次大小、是否打乱数据以及并发加载的工作者数量
        loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, sampler=None)
        return loader

    def incremental_train(self, data_manager, generator):
        """
        进行增量训练的函数。

        参数:
        - data_manager: 数据管理器，负责获取不同任务的数据集。

        该函数不返回任何值，但会更新模型以适应新的任务。
        """

        # 更新当前任务标识，并计算当前总类别数
        self._cur_task += 1
        self._cur_task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_task_size
        # 根据总类别数更新网络的全连接层
        self._network.update_fc(self._total_classes)
        self.generation = generator
        # 打印当前学习的任务范围
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # 获取当前任务的训练数据集
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        # 获取包含所有类别的测试数据集
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        # 初始化测试数据的加载器
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )
        # 设置随机种子，确保实验可复现
        setup_seed(self.seed)
        # 如果是第一个任务，并且保存目录不存在，则创建保存目录
        if self._cur_task == 0 and (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)
        # 如果不是第一个任务，则获取旧任务的合成数据的加载器
        if self._cur_task != 0:
            self.syn_data_loader = self.get_syn_data_loader()

        # 对所有任务进行联合训练
        self._fl_train(train_dataset, self.test_loader)
        self.train_loader = DataLoader(train_dataset,batch_size=128,shuffle=False,num_workers=4)
        # 计算当前任务的类别原型
        self._compute_prototypes()
        # 计算当前任务的EFM矩阵
        efm_matrix = EmpiricalFeatureMatrix(self.device)
        efm_matrix.compute(self._network, deepcopy(self.train_loader), self._cur_task)
        self.previous_efm = efm_matrix.get()    
        # 如果当前任务不是最后一个任务，则进行数据生成
        if self._cur_task + 1 != self.tasks:
            self.data_generation()
    def _compute_prototypes(self):
        """
        计算当前模型在训练数据上的类别原型
        """
        self._network.eval()  # 设置为评估模式
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 初始化累积原型和样本计数
        prototypes = {i: torch.zeros(self._network.feature_dim, device=device) for i in range(self._known_classes, self._total_classes)}
        class_counts = {i: 0 for i in range(self._known_classes, self._total_classes)}
        
        # 遍历所有训练数据
        with torch.no_grad():  # 避免存储计算图
            for _, images, labels in self.train_loader:
                # 将数据移动到GPU
                images, labels = images.to(device), labels.to(device)
                
                # 提取特征向量
                features = self._network.extract_vector(images)
                
                # 按类别累积特征
                for i in range(self._known_classes, self._total_classes):
                    mask = labels == i
                    if mask.any():
                        prototypes[i] += features[mask].sum(dim=0)
                        class_counts[i] += mask.sum().item()
            
            # 计算每个类别的平均原型
            for i in range(self._known_classes, self._total_classes):
                if class_counts[i] > 0:
                    prototypes[i] = prototypes[i] / class_counts[i]
                    self.global_prototypes[i] = prototypes[i]
        # 清理内存
        torch.cuda.empty_cache()
        self._network.train()  # 恢复训练模式
        
    def efm_loss(self, features, features_old):
        features = features.unsqueeze(1)
        features_old = features_old.unsqueeze(1)
        matrix_reg = self.efc_lamb *  self.previous_efm + self.damping * torch.eye(self.previous_efm.shape[0], device=self.device) 
        efc_loss = torch.mean(torch.bmm(torch.bmm((features - features_old), matrix_reg.expand(features.shape[0], -1, -1)), (features - features_old).permute(0,2,1)))
        return  efc_loss
    
    def _local_update(self, model, train_data_loader):
        """
        在本地进行模型的更新。

        参数:
        - model: 要更新的模型
        - train_data_loader: 训练数据的加载器

        返回:
        - model.state_dict(): 更新后模型的状态字典
        """
        model.train()  # 将模型设置为训练模式
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 初始化对比增量学习损失函数
        criterion = SupConLoss(temperature=self.args['con_temperature'])
        if torch.cuda.is_available():
            criterion = criterion.cuda()
            cudnn.benchmark = True

        # 统计每个类别的样本数量(正则化交叉熵损失)
        class_counts = torch.zeros(self._total_classes).to(device)
        for _, _, labels in train_data_loader:
            for label in labels:
                class_counts[label] += 1
        
        # 计算先验概率
        self.prior_regularization = class_counts / class_counts.sum()
        self.prior_regularization = torch.tensor(self.prior_regularization).cuda()
        self.prior_regularization = torch.log(self.prior_regularization)

        gradall = np.zeros(self._total_classes, dtype = float)
        # 初始化优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

        self.train_transform = transforms.Compose([
            # 图片大小调整
            transforms.Resize(size=(self.args['size'], self.args['size'])),
        
            # 随机水平翻转
            transforms.RandomHorizontalFlip(),
            # 随机裁剪
            transforms.RandomResizedCrop(size=self.args['size'],
                                         scale=(0.1 if self.args['dataset'] == 'tiny_imagenet' else 0.2, 1.)),
            # 随机颜色抖动
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # 标准化处理
            transforms.Normalize(
               mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        # 进行本地轮次的训练
        for iter in range(self.args["local_ep"]):
            # 遍历训练数据集
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                
                # 将数据移动到GPU上
                images, labels = images.to(device), labels.to(device)
                # 进行数据增强 来使用对比学习
                if self.is_con:
                    images_con1 = self.train_transform(images)
                    images_conv = torch.cat([images, images_con1], dim=0)
                    # 将数据移动到GPU上
                    images_conv = images_conv.to(device)
                    # 计算当前批次大小
                    bsz = labels.shape[0]
                    features = model(images_conv)["proj"]
                    # 对比学习损失
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    # 计算当前目标任务的损失，目标类别为从当前任务开始的类别范围。
                    conv_loss = criterion(features, labels, target_labels=list(
                        range(self._cur_task * self._cur_task_size, (self._cur_task + 1) * self._cur_task_size)))
                else:
                    conv_loss = 0
                output = model(images)["logits"]
                # outputs_loss = output
                # outputs_loss += self.prior_regularization
                # 计算损失
                ce_loss = F.cross_entropy(output, labels.long())
                # 清空梯度
                optimizer.zero_grad()
                # 计算总损失
                loss = ce_loss + self.args["current_task_weight"] * conv_loss
                # 反向传播
                loss.backward()

                # 计算每个类的梯度累积
                # grad_norm = []
                # for ele in model.fc.heads:
                #     grad_norm.extend(torch.norm(ele.weight.grad, dim=1).data.cpu().numpy())
                # # 累积梯度 
                # gradall += np.array(grad_norm)+ 1e-8
                # # 计算类平衡比率 α_i^j
                # # gradall 对应公式中的 Φ_i^j,即累积的梯度大小
                # # 计算 min(Φ_i^m)/Φ_i^j
                # self.grad_weight = np.min(gradall) / np.array(gradall)
                # idx_grad = 0
                # for i in range(len(model.fc.heads)):
                #     for j in range(model.fc.heads[i].weight.shape[0]):
                #         model.fc.heads[i].weight.grad[j, :] = model.fc.heads[i].weight.grad[j, :] * self.grad_weight[j + idx_grad] 
                #     for j in range(len(model.fc.heads[i].bias)):
                #         model.fc.heads[i].bias.grad[j] = model.fc.heads[i].bias.grad[j] * self.grad_weight[j + idx_grad] 
                #     idx_grad += model.fc.heads[i].weight.shape[0]
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)

                # 更新参数
                optimizer.step()
        return model.state_dict()
    
    def compute_channel_importance(self, features, old_features):
        """
        计算每个通道的重要性权重。
        
        参数:
        - features: 当前模型的特征图
        - old_features: 旧模型的特征图
        
        返回:
        - channel_weights: 每个通道的重要性权重
        """
        # 计算旧任务数据在每个通道上的L2范数
        channel_norms = torch.norm(old_features, p=2, dim=(0, 2, 3))  # [C]
        # 归一化得到通道权重
        channel_weights = channel_norms / torch.sum(channel_norms)
        return channel_weights

    def channel_distillation_loss(self, student_features, teacher_features):
        """
        计算通道级别的特征蒸馏损失
        参数:
        - student_features: 学生模型的特征图列表
        - teacher_features: 教师模型的特征图列表
        
        返回:
        - loss: 加权的特征蒸馏损失
        """
        device = student_features[0].device
        total_loss = torch.tensor(0., device=device)
        
        # 对每一层的特征图进行处理
        for sf, tf in zip(student_features, teacher_features):
            # 确保特征图形状匹配
            if sf.shape != tf.shape:
                print("特征图形状不匹配")
                continue
                
            # 计算通道重要性权重
            channel_weights = self.compute_channel_importance(sf, tf)  # [C]
            
            # 将权重扩展到与特征图相同的维度
            weights = channel_weights.view(1, -1, 1, 1)  # [1, C, 1, 1]
            
            # 计算加权的特征距离
            diff = (sf - tf) * weights
            loss = torch.norm(diff, p=2)
            
            total_loss += loss
            return total_loss
    
    def _local_finetune(self, teacher, model, train_data_loader, task_id, client_id):
        model.train()  # 将模型设置为训练模式
        teacher.eval()  # 将教师模型设置为评估模式
        criterion = SupConLoss(temperature=self.args['con_temperature'])
        if torch.cuda.is_available():
            criterion = criterion.cuda()
            cudnn.benchmark = True
        # 初始化优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.train_transform = transforms.Compose([
            # 图片大小调整
            transforms.Resize(size=(self.args['size'], self.args['size'])),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(),
            # 随机裁剪
            transforms.RandomResizedCrop(size=self.args['size'],
                                         scale=(0.1 if self.args['dataset'] == 'tiny_imagenet' else 0.2, 1.)),
            # 随机颜色抖动
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # 标准化处理
            transforms.Normalize(
               mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 如果不是第一个任务且启用了特征知识蒸馏，预先计算教师模型的特征图
        if self._cur_task > 0 and self.is_feature_kd:
            self._old_network.eval()  # 确保教师模型处于评估模式
            self.teacher_features_dict = {}
            
            with torch.no_grad():
                for batch_idx, ((_, images, labels), (syn_input, _)) in enumerate(zip(train_data_loader, self.syn_data_loader)):
                    syn_input = syn_input.to(device)
                    teacher_features = self._old_network.extract_fmaps(syn_input)
                    # 存储到CPU以节省GPU内存
                    self.teacher_features_dict[batch_idx] = teacher_features

        # 统计每个类别的样本数量(正则化交叉熵损失)
        class_counts = torch.zeros(self._total_classes).to(device)
        for _, _, labels in train_data_loader:
            for label in labels:
                class_counts[label] += 1
        gradall = np.zeros(self._total_classes-self._known_classes, dtype = float)
        # 计算先验概率
        self.prior_regularization = class_counts / class_counts.sum()
        self.prior_regularization = torch.tensor(self.prior_regularization).cuda()
        self.prior_regularization = torch.log(self.prior_regularization)    
        # 迭代进行本地训练
        for it in range(self.args["local_ep"]):
            iter_loader = enumerate(zip((train_data_loader), (self.syn_data_loader)))
            total_local = 0.0  # 初始化本地数据样本总数
            total_syn = 0.0  # 初始化合成数据样本总数
            for batch_idx, ((_, images, labels), (syn_input, _)) in iter_loader:
                # 对本轮图像和生成图像进行增强
                # 移动到GPU
                
                images, labels, syn_input = images.to(device), labels.to(device), syn_input.to(
                    device)
                with torch.no_grad():
                    t_out = teacher(syn_input.detach())["logits"]  # 在教师模型上获取输出
                    total_syn += syn_input.shape[0]  # 更新同步数据样本总数
                    total_local += images.shape[0]  # 更新本地数据样本总数
                # 如果有指定允许的预测标签，截取对应的预测分数
                t_hat = t_out[:, :self._known_classes]
                _, syn_target = torch.max(t_hat, dim=1)  # 获取预测的标签
                syn_target = syn_target.to(device)

                if self.is_con:
                    images_con = self.train_transform(images)
                    syn_input_con1 = self.train_transform(syn_input)
                    images_con, syn_input_con1 = images_con.to(device), syn_input_con1.to(
                        device)
                    images_conv = torch.cat([images, syn_input, images_con, syn_input_con1], dim=0)
                    combined_labels = torch.cat([labels, syn_target], dim=0)
                    images_conv, combined_labels = images_conv.to(device), combined_labels.to(
                        device)
                    # 获取图片和生成图片的批次大小
                    bsz_image = labels.shape[0]
                    bsz_syn = syn_target.shape[0]
                    # 获取相应的特征
                    feat = model(images_conv)["proj"]
                    # 对特征进行归一化
                    f1, f2 = torch.split(feat, [bsz_image + bsz_syn, bsz_image + bsz_syn], dim=0)
                    feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    # 计算新任务特征层的对比损失
                    conv_loss = criterion(feat, combined_labels, target_labels=list(
                        range(self._cur_task * self._cur_task_size, (self._cur_task + 1) * self._cur_task_size)))
                else:
                    conv_loss = 0

                if self.is_loss_kd:
                    s_out = model(syn_input)["logits"]  # 在合成数据上获取模型输出
                    # 计算旧任务的知识蒸馏损失
                    loss_kd = _KD_loss(
                        s_out[:, : self._known_classes],  # 之前任务的logits
                        t_out.detach(),
                        2,
                    )
                else:
                    loss_kd = 0
                
                # 重要性特征蒸馏
                if self.is_feature_kd:
                    # teacher_features = self.teacher_features_dict[batch_idx]
                    # student_features = model.extract_fmaps(syn_input)
                    # # 计算通道级别的特征蒸馏损失
                    # loss_feature_kd = self.channel_distillation_loss(student_features, teacher_features)
                    new_features = model.extract_vector(images)
                    old_features = teacher.extract_vector(images)
                    loss_feature_kd = self.efm_loss(new_features, old_features)
                else:
                    loss_feature_kd = 0

                # 计算新任务的交叉熵损失（梯度重新加权）
                output = model(images)["logits"]  # 获取模型输出
                fake_targets = labels - self._known_classes  # 创建伪目标
                # outputs_loss = output
                # outputs_loss += self.prior_regularization
                loss_ce = F.cross_entropy(output[:, self._known_classes:], fake_targets.long())
                
                # 计算总损失并进行反向传播
                loss = 10 * conv_loss +  loss_ce + self.args[
                    "kd"] * loss_kd +  0.005 * loss_feature_kd
                
                optimizer.zero_grad()
                loss.backward()

#                 grad_cn = []
#                 bias_cn = []
#                 grad_norm_cn = []
#                 # 只遍历最后一个head(当前任务的head)
#                 for ele in model.fc.heads[-1:]: 
#                     temp_norm = torch.autograd.grad(loss_ce, ele.weight, retain_graph=True)[0] 
#                     grad_norm_cn.extend(torch.norm(temp_norm, dim =1).data.cpu().numpy())
#                     grad_cn.append(temp_norm)
#                     bias_cn.append(torch.autograd.grad(loss_ce, ele.bias, retain_graph=True)[0] )
#                 gradall += np.array(grad_norm_cn)
#                 self.grad_weight = []
#                 new_temp = np.array(gradall)
#                 new_temp_weight = np.min(new_temp) / new_temp
#                 self.grad_weight.extend(new_temp_weight.tolist())
#                 self.grad_weight = np.array(self.grad_weight)
#                 self.grad_weight = self.grad_weight
#                 # 只处理最后一个head
#                 last_head = model.fc.heads[-1]
#                 for j in range(last_head.weight.shape[0]):
#                     # 应用梯度权重到权重梯度
#                     grad_temp_sum = self.grad_weight[j] * grad_cn[0][j,:]  
#                     last_head.weight.grad[j, :] = grad_temp_sum
                    
#                     # 应用梯度权重到偏置梯度
#                     grad_temp_sum = self.grad_weight[j] * bias_cn[0][j]
#                     last_head.bias.grad[j] = grad_temp_sum
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
                optimizer.step()
        return model.state_dict(), total_syn, total_local

    def _fl_train(self, train_dataset, test_loader):
        """
        执行联邦学习的训练过程。

        参数:
        - train_dataset: 训练数据集，包含了训练数据的信息和标签。
        - test_loader: 测试数据集加载器，用于在每个通信轮之后评估模型性能。

        无返回值。
        """
        # 将网络移至GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._network.to(device)
        # 根据用户数量和指定的参数，将训练数据集划分成不同的用户组
        user_groups = partition_data(train_dataset.labels, self.args["seed"], beta=self.args["beta"],
                                     n_parties=self.args["num_users"])
        # 创建条，用于显示训练进度
        prog_bar = tqdm(range(self.args["com_round"]))
        for _, com in enumerate(prog_bar):
            # 初始化本地权重列表和本地原型表
            local_weights = []
            client_prototypes = {}  # 用于存储每个客户端的本地原型
            class_counts = {i: 0 for i in range(self._known_classes, self._total_classes)}      # 存储每个类别在各个客户端的样本总数
            # 根据参与联邦学习的用户比例，随机选择一批用户参与当前通信轮的训练
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            # 随机选择一批用户参与当前通信轮的训练
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                # 为每个选中的用户创建数据加载器
                local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                                                batch_size=self.args["local_bs"], shuffle=True, num_workers=4)
                # 根据当前任务，执行本地更新或本地微调
                if self._cur_task == 0:
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader)
                else:
                    w, total_syn, total_local = self._local_finetune(self._old_network, copy.deepcopy(self._network),
                                                                     local_train_loader, self._cur_task, idx)
                    # 在第一个通信轮和第二个任务时，打印用户ID、本地数据集大小和合成数据大小
                    if com == 0 and self._cur_task == 1:
                        print("\t \t client {}, local dataset size:{},  syntheic data size:{}".format(idx, total_local,
                                                                                                      total_syn))

                # 将本地更新的权重添加到本地权重列表中
                local_weights.append(copy.deepcopy(w))
                # 计算本地类原型
#                 if com == self.args["com_round"] - 1:
#                     self._network.load_state_dict(w)
#                     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#                     # 初始化客户端原型字典
#                     if idx not in client_prototypes:
#                         client_prototypes[idx] = {}

#                     # 初始化累积原型和样本计数
#                     local_class_prototypes = {i: torch.zeros(self._network.feature_dim, device=device) for i in range(self._known_classes, self._total_classes)}
#                     local_class_counts = {i: torch.zeros(1, device=device) for i in range(self._known_classes, self._total_classes)}
                    
#                     # 遍历所有批次数据
#                     with torch.no_grad():  # 避免存储计算图    
#                         for batch_idx, (_, images, labels) in enumerate(local_train_loader):
#                             # 将数据移动到GPU上
#                             images, labels = images.to(device), labels.to(device)
#                             features = self._network.extract_vector(images)
                            
#                             # 累积每个类别的特征和计数
#                             for i in range(self._known_classes, self._total_classes):
#                                 mask = labels == i
#                                 if mask.any():
#                                     local_class_prototypes[i] += features[mask].sum(dim=0)
#                                     local_class_counts[i] += mask.sum().item()
#                         # 清理临时变量
#                         del features
#                         torch.cuda.empty_cache()
                        
#                         # 计算每个类别的平均原型
#                         for i in range(self._known_classes, self._total_classes):
#                             if local_class_counts[i] > 0:
#                                 # 计算平均原型
#                                 avg_prototype = local_class_prototypes[i] / local_class_counts[i]
#                                 client_prototypes[idx][i] = (avg_prototype, local_class_counts[i])  # 存储原型和样本数 
#                                 class_counts[i] += local_class_counts[i]

#                         # 计算加权平均的全局原型
#                         for class_idx in range(self._known_classes, self._total_classes):
#                             if class_counts[class_idx] > 0:
#                                 for client_idx, client_data in client_prototypes.items():
#                                     if class_idx in client_data:
#                                         proto, count = client_data[class_idx]
                                        
#                                 global_prototype = torch.zeros(self._network.feature_dim).to(device)
#                                 # 对每个客户端的原型进行加权求和
#                                 for client_idx, client_data in client_prototypes.items():
#                                     if class_idx in client_data:
#                                         proto, count = client_data[class_idx]
#                                         # 权重为该客户端该类的样本数除以该类的总样本数
#                                         weight = count / class_counts[class_idx]
#                                         global_prototype += proto * weight
#                                 self.global_prototypes[class_idx] = global_prototype

                # 清理内存
                del local_train_loader, w
                torch.cuda.empty_cache()

                # 计算全局权重的平均值，并加载到模型中
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            if self._cur_task > 0:
                self._network.weight_align(self._total_classes, self._cur_task_size)
            # 如果是每一轮的通信轮，计算并记录测试准确率
            if com % 1 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                # 更新进度条描述，添加测试准确率信息
                info = ("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc, ))
                prog_bar.set_description(info)
                # 如果启用了Wandb日志记录，记录当前任务的测试准确率
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})


def _KD_loss(pred, soft, T):
    """
    计算知识蒸馏损失函数。

    参数:
    pred: 模型的预测输出，是一个tensor。
    soft: 教师模型的softmax输出，也是一个tensor。
    T: 温度参数，用于调整softmax函数的输出。

    返回值:
    返回计算得到的知识蒸馏损失值，是一个scalar tensor。
    """
    # 对预测结果应用log_softmax，其中T用于温度调整
    pred = torch.log_softmax(pred / T, dim=1)
    # 对教师模型的输出应用softmax，同样使用T进行温度调整
    soft = torch.softmax(soft / T, dim=1)
    # 计算损失，即学生模型和教师模型输出的加权交叉熵的负值
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
