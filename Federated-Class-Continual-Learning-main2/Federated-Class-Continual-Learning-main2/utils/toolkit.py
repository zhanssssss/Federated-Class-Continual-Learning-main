import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    """
    计算模型的参数数量。

    参数:
    - model: 要计算参数数量的模型。
    - trainable (bool): 是否只计算可训练参数的默认值为False，即计算所有参数的数量。

    返回:
    - int: 模型参数的总数。
    """
    if trainable:
        # 如果只计算可训练参数，筛选出需要梯度的参数并求和
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 否则，计算模型的所有参数数量
    return sum(p.numel() for p in model.parameters())



def tensor2numpy(x):
    """
    将Tensor转换为Numpy数组。
    如果Tensor在CUDA设备上，则先转移到CPU再进行转换。

    参数:
    x (torch.Tensor): 输入的Tensor。

    返回:
    numpy.ndarray: 转换后的Numpy数组。
    """
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def target2onehot(targets, n_classes):
    """
    将目标标签转换为one-hot编码形式。

    参数:
    targets (torch.Tensor): 输入的目标标签Tensor。
    n_classes (int): 类别的总数。

    返回:
    torch.Tensor: 转换后的one-hot编码Tensor。
    """
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)  # 创建一个与目标相同设备的全零Tensor
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)  # 根据目标标签设置对应位置为1
    return onehot

def makedirs(path):
    """
    创建目录及其父目录。如果目录已存在，则不进行任何操作。

    参数:
    path (str): 要创建的目录路径。
    """
    if not os.path.exists(path):  # 如果路径不存在，则创建路径
        os.makedirs(path)



def accuracy(y_pred, y_true, nb_old, increment=10):
    """
    计算并返回预测准确率的详细报告。

    参数:
    y_pred: 一维数组，表示模型的预测结果。
    y_true: 一维数组，表示真实标签。
    nb_old: 整数，用于将标签分为"旧"和"新"两类的阈值。
    increment: 整数，默认为10，用于分组计算准确率的步长。

    返回值:
    一个字典，包含总准确率、分组准确率（以类区间为键）、"旧"标签准确率和"新"标签准确率。
    """
    assert len(y_pred) == len(y_true), "Data length error."  # 确保预测结果和真实标签长度一致

    all_acc = {}  # 准确率报告字典
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )  # 计算总体准确率

    # 分组计算准确率
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # 计算"旧"标签的准确率
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # 计算"新"标签的准确率
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


class AverageMeter(object):
    """
    用于计算并存储平均值和当前值的类。
    """

    def __init__(self):
        """
        初始化方法，调用后将所有度量值重置为0。
        """
        self.reset()

    def reset(self):
        """
        重置所有度量值为0。
        """
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数器，记录更新次数

    def update(self, val, n=1):
        """
        更新度量值。

        参数:
        - val: 当前的值。
        - n: 本次更新的权重，默认为1，表示简单更新。

        返回值: 无
        """
        self.val = val  # 更新当前值
        self.sum += val * n  # 更新总和
        self.count += n  # 更新计数器
        self.avg = self.sum / self.count  # 更新平均值
def split_images_labels(imgs):
    """
    分离图像和标签

    参数:
    imgs - 一个列表，包含了图像和其对应的标签，每个元素是一个二元组，其中第一个元素是图像路径，第二个元素是标签。

    返回值:
    images - 图像路径的numpy数组
    labels - 标签的numpy数组
    """
    # 初始化图像和标签的列表
    images = []
    labels = []
    # 遍历输入的图像标签列表，分离图像路径和标签，并分别添加到对应的列表中
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    # 将图像路径和标签的列表转换为numpy数组并返回
    return np.array(images), np.array(labels)


class TwoCropTransform:
    """
    生成同一个图像的两个裁剪。
    参数:
    - transform: 一个转换函数，用于对图像进行转换。
    """
    def __init__(self, train_con_transform):
        """
        初始化TwoCropTransform对象。
        参数:
        - transform: 一个转换函数，用于对图像进行转换。
        """
        self.train_con_transform = train_con_transform
    def __call__(self, x):
        """
        对输入的图像进行转换，生成两个相同的裁剪。
        返回值:
        - 包含两个经过transform转换后的图像的列表。
        """
        return [x, self.train_con_transform(x), self.train_con_transform(x)]

def combine_data(data):
    x, y = [], []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    x, y = torch.cat(x), torch.cat(y)
    return x, y