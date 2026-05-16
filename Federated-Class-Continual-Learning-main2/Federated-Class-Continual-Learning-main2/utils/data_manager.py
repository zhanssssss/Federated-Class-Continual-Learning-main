import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, TinyImageNet200
import torch, copy
import os, pdb, random
import numpy as np
import torch.backends.cudnn as cudnn


def setup_seed(seed):
    """
    设置随机种子，以确保在不同设备和运行多次时结果一致。

    参数:
    seed (int): 随机种子的值。用于设置各种随机数生成器的初始状态。

    返回值:
    无
    """
    # 设置PyTorch的CPU和GPU随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 重复设置GPU随机种子，确保与手动设置和所有设备一致
    torch.cuda.manual_seed(seed)
    # 设置NumPy和Python内置随机数生成器的种子
    np.random.seed(seed)
    random.seed(seed)
    # 设置CUDNN的行为为确定性的，进一步确保重复性
    cudnn.deterministic = True


def average_weights(w):
    """
    计算给定权重列表的平均值。

    参数:
    w: 一个包含多个权重字典的列表。每个权重字典都含有不同的键值对，其中可能包括一个名为'num_batches_tracked'的特殊键。

    返回值:
    返回一个平均权重字典，其中每个键对应的值是原始权重列表中相应键值的平均值。
    """
    # 深拷贝第一个权重字典作为计算平均值的初始值
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        # 对每个键对应的值求和
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        # 如果键包含'num_batches_tracked'，则对值进行特殊处理
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            # 对其他键对应的值除以权重列表的长度求平均
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class DatasetSplit(Dataset):
    """
    一个围绕Pytorch Dataset类的抽象Dataset类。

    参数:
    - dataset: 原始的Pytorch Dataset类实例，数据集分割将基于此实例。
    - idxs: 一个字符串列表，包含了数据集中被选择的样本的索引。
    """

    def __init__(self, dataset, idxs):
        # 初始化数据集分割
        self.dataset = dataset  # 原始数据集
        self.idxs = [int(i) for i in idxs]  # 将字符串索引转换为整型

    def __len__(self):
        # 返回数据集分割中样本的数量
        return len(self.idxs)

    def __getitem__(self, item):
        """
        根据索引获取数据集中的样本。

        参数:
        - item: 整数，表示要获取的样本的索引。

        返回值:
        - 一个元组，包含样本的索引、图像数据和标签。
        """
        idx, image, label = self.dataset[self.idxs[item]]  # 从原始数据集中获取指定索引的样本
        return idx, image, label


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(y_train, seed, beta=0.4, n_parties=5):
    """
    根据训练集的标签将数据分割为多个客户端（party）。

    参数:
    - y_train: 一维数组，表示训练集的标签。
    - beta: 浮点数，用于非iid（Non-IID）数据划分的参数，控制数据在派系间的分布。beta值为0时表示iid情况，大于0时表示非iid情况。
    - n_parties: 整数，表示要将数据分割成的派系数量。

    返回值:
    - net_dataidx_map: 字典，键为派系索引，值为该派系包含的数据索引列表。
    """
    setup_seed(seed)
    data_size = y_train.shape[0]
    if beta == 0:  # 对于iid数据的划分
        idxs = np.random.permutation(data_size)  # 随机打乱数据索引
        batch_idxs = np.array_split(idxs, n_parties)  # 将打乱后的索引分割为n_parties个派系
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}  # 构建客户端与数据索引的映射

    elif beta > 0:  # 对于非iid数据的划分
        min_size = 0
        min_require_size = 1
        labels = np.unique(y_train)  # 获取训练集中所有独特的标签
        net_dataidx_map = {}

        while min_size < min_require_size:  # 循环直到每个客户端的数据量满足要求
            idx_batch = [[] for _ in range(n_parties)]  # 初始化每个客户端的数据索引列表

            for k in labels:  # 遍历每个标签
                idx_k = np.where(y_train == k)[0]  # 找到所有属于该标签的数据索引
                np.random.shuffle(idx_k)  # 随机打乱该标签的数据索引
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))  # 生成客户端间数据分布的比例
                # 根据客户端当前已有的数据量，调整比例，确保数据在客户端间的分布
                proportions = np.array([
                    p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()  # 调整比例使其总和为1
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]  # 根据比例分割数据索引
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])  # 更新最小数据量

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])  # 对每个客户端的数据索引再次随机打乱
            net_dataidx_map[j] = idx_batch[j]  # 将客户端与数据索引映射加入结果字典
    # 返回数据索引的客户端划分
    return net_dataidx_map


class DataManager(object):
    """
    数据管理器类，负责初始化和管理数据集。

    参数:
    - dataset_name: 数据集的名称，用于指定加载的数据集。
    - shuffle: 是否对数据集进行洗牌。
    - seed: 随机种子，用于确保数据集的洗牌可复现。
    - init_cls: 初始加载的类别数量。
    - increment: 每次增量加载的类别数量。
    """

    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name  # 数据集名称
        self._setup_data(dataset_name, shuffle, seed)  # 初始化数据集
        # 确保初始加载的类别数量不超过总类别数
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]  # 初始化增量列表
        # 循环直到增量加载的类别总数达到或超过总类别数
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        # 计算并添加剩余的类别数量到增量列表中
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
            self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        """
        根据指定的索引、数据源、模式获取数据集。

        参数:
        - indices: 指定的数据索引列表。
        - source: 数据源，可选值为"train"或"test"。
        - mode: 数据处理模式，可选值为"train"、"flip"或"test"。
        - appendent: 附加的数据集，可选。
        - ret_data: 是否返回原始数据，布尔值。
        - m_rate: 用于数据增强的随机抹除率，可选。

        返回值:
        - 如果ret_data为True，则返回(data, targets, dataset)元组，其中data为数据，targets为标签，dataset为处理后的数据集对象。
        - 如果ret_data为False，则仅返回处理后的数据集对象。
        """
        # 根据source选择数据和标签
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        # 根据mode选择数据处理方式
        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        # 根据indices选取数据和对应的标签
        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        # 如果提供了appendent，将其数据和标签添加到结果中
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        # 将选取的数据和标签合并
        data, targets = np.concatenate(data), np.concatenate(targets)

        # 根据是否需要返回原始数据，构造并返回相应的数据集对象
        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
            self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        """
        设置数据集，包括下载数据、划分训练集和测试集，以及应用变换。

        参数:
        - dataset_name (str): 数据集的名称，用于标识和获取特定的数据集。
        - shuffle (bool): 是否对训练集的目标进行洗牌。如果为True，则训练样本的顺序会被随机打乱。
        - seed (int): 如果shuffle为True，用于随机数生成器的种子值，以确保可复现性。

        无返回值，但会修改实例的多个属性，包括训练数据、测试数据、目标、变换等。
        """

        # 获取数据集实例并下载数据
        idata = _get_idata(dataset_name)
        idata.download_data()

        # 初始化数据和变换
        # 数据加载
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # 变换初始化
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # 设置类别顺序
        # 首先基于训练目标的唯一值创建一个默认顺序
        order = [i for i in range(len(np.unique(self._train_targets)))]
        # 如果需要洗牌，使用给定的种子值进行随机重排序
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            # 如果不洗牌，则使用数据集固有的类别顺序
            order = idata.class_order
        self._class_order = order

        # 映射新的类别索引
        # 这一步将原始类别标签映射到新的索引，以便于处理和表示
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        """
        选择符合条件的数据项。

        该方法从给定的x, y数据中，筛选出y值在指定范围内的数据项，并返回这些数据项的x和y值。

        参数:
        - x: 包含x值的data。
        - y: 包含y值的target，将根据此参数的值进行筛选。
        - low_range: 筛选的下限。
        - high_range: 筛选的上限（不包含）。

        返回值:
        - 一个元组，包含两个元素：筛选后的x值数组和y值数组。
        """
        # 根据条件筛选出符合条件的y值的索引
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        # 使用索引筛选出对应的x和y值
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        """
        根据给定的条件选择样本，并按照指定的保留率m_rate进行随机采样。

        :param x: 输入数据集
        :param y: 目标数据集
        :param low_range: 选择范围的下限
        :param high_range: 选择范围的上限
        :param m_rate: 保留率，决定了采样的比例
        :return: 采样后的输入数据和目标数据
        """
        # 确保m_rate不为None
        assert m_rate is not None
        if m_rate != 0:
            # 根据条件筛选出符合条件的索引
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            # 随机选择部分索引，根据保留率确定数量
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            # 对选择的索引进行排序
            new_idxes = np.sort(new_idxes)
        else:
            # 如果保留率为0，则直接选择所有符合条件的索引
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        """
        计算目标索引在目标数据集中出现的总次数。

        :param index: 目标索引
        :return: 目标索引出现的总次数
        """
        y = self._train_targets
        # 统计目标索引出现的次数
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    """
    一个用于处理图像数据集的虚拟类，支持对图像应用变换。

    参数:
    - images: 图像数据，可以是图像的路径列表或直接的图像数组。
    - labels: 对应图像的标签列表。
    - trsf: 应用于图像的变换函数或方法。
    - use_path: 指定是否将images作为图像路径处理。如果为True，则images应为路径列表。
    """

    def __init__(self, images, labels, trsf, use_path=False):
        # 确保图像和标签数量一致
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        # 返回数据集中图像的数量
        return len(self.images)

    def __getitem__(self, idx):
        """
        根据索引获取图像及其标签，可应用变换。

        参数:
        - idx: 图像的索引。

        返回值:
        - 一个元组，包含图像索引、应用变换后的图像和对应的标签。
        """
        if self.use_path:
            # 如果use_path为True，从路径加载图像并应用变换
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            # 如果use_path为False，将图像数组转换为图像并应用变换
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    """
    将原始类别标签映射到新的索引序列。

    参数:
    y: 原始类别标签的数组。
    order: 定义了新索引序列的列表，按照此顺序映射原始标签。

    返回值:
    映射后类别标签的新索引数组。
    """
    return np.array(list(map(lambda x: order.index(x), y)))
    # 使用lambda函数和map()将y中的每个元素映射到order中对应索引，然后转换为numpy数组


def _get_idata(dataset_name):
    """
    根据指定的数据集名称获取相应的数据集实例。

    参数:
    - dataset_name: 字符串，指定要加载的数据集的名称，必须是小写。

    返回值:
    - 返回一个数据集实例，具体类型取决于传入的dataset_name。

    异常:
    - 如果传入了未知的数据集名称，将抛出NotImplementedError。
    """
    name = dataset_name.lower()  # 将传入的名称转换为小写，确保后续比较的一致性
    if name == "cifar10":
        return iCIFAR10()  # 加载CIFAR-10数据集
    elif name == "cifar100":
        return iCIFAR100()  # 加载CIFAR-100数据集
    elif name == "imagenet1000":
        return iImageNet1000()  # 加载ImageNet1000数据集
    elif name == "imagenet100":
        return iImageNet100()  # 加载ImageNet100数据集
    elif name == "tiny_imagenet":
        return TinyImageNet200()  # 加载TinyImageNet200数据集
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))  # 如果没有匹配到任何已知数据集，抛出异常


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

# def accimage_loader(path):
#     """
#     Ref:
#     https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
#     accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
#     accimage is available on conda-forge.
#     """
#     import accimage

#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


# def default_loader(path):
#     """
#     Ref:
#     https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
#     """
#     from torchvision import get_image_backend

#     if get_image_backend() == "accimage":
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)
