import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import os


data_dir = os.path.join(os.environ['HOME'],"autodl-fs","datasets","cifar100")


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(data_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(data_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):  # 1300*100 = 13w, 5 tasks, each task 20*1300=2.6w
    use_path = True
    train_trsf = [
        # transforms.RandomResizedCrop(224),
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.CenterCrop(128),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dir = "{}/imagenet100/train/".format(data_dir)
        test_dir = "{}/imagenet100/val/".format(data_dir)

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class TinyImageNet(Dataset):
    """
    TinyImageNet数据集类，用于加载和预处理TinyImageNet数据集。

    参数:
    - root: 数据集根目录的路径。
    - train: 指定是否加载训练集，True为加载训练集，False为加载验证集。
    - transform: 应用于图像的转换，可为None或实现为图像转换的函数。
    """

    def __init__(self, root, train=True, transform=None):
        # 初始化数据集属性
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")  # 训练集目录路径
        self.val_dir = os.path.join(self.root_dir, "val")  # 验证集目录路径

        # 根据train参数创建相应的类索引字典
        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        # 构建数据集
        self._make_dataset(self.Train)

        # 读取类别ID和单词文件
        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()  # 存储有效类别ID的集合

        # 从文件中读取并填充set_nids
        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}  # 类别ID到标签的映射

        # 读取类别ID到标签的映射并填充class_to_label
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        """
        创建训练集的类别索引字典。
        此方法会扫描训练目录，为每个子目录（假定为一个类别）生成一个索引，并统计训练集中图像的总数。
        利用这些信息，它会更新类到索引的映射以及索引到类的映射。

        参数:
        self - 对象自身的引用。

        返回值:
        无
        """
        # 根据Python版本选择合适的目录扫描方法
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        # 对类别名称进行排序
        classes = sorted(classes)

        # 统计训练集中图像的总数
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images  # 更新数据集总长度

        # 创建类别索引到类名的映射以及类名到类别索引的映射
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        """
        返回值:
        - 无。此方法不返回任何值，但会更新实例的 `images` 属性，包含图片路径及其对应的类别标签。
        """
        self.images = []
        # 根据训练或验证集来选择图片根目录
        if Train:
            img_root_dir = self.train_dir
            # 训练集按类别目录组织
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            # 验证集图片目录统一为 "images"
            list_of_dirs = ["images"]

        # 遍历每个目录，收集图片信息
        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            # 跳过非目录项
            if not os.path.isdir(dirs):
                continue

            # 在目录中遍历所有图片文件
            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    # 仅选择以 .JPEG 结尾的文件
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        # 根据集合并选择对应的标签组织数据
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)


class TinyImageNet200(iData):   # 200*500=10w, 5 tasks, each task=40*500=2w
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.CenterCrop(64),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = data_dir
        test_dir = data_dir
        # print()

        train_dset = TinyImageNet(train_dir, train=True)
        test_dset = TinyImageNet(test_dir, train=False)


        self.train_data, self.train_targets = split_images_labels(train_dset.images)
        self.test_data, self.test_targets = split_images_labels(test_dset.images)
