import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from utils.data_manager import DummyDataset, _get_idata
from torchvision import transforms

EPSILON = 1e-8
batch_size = 128


class BaseLearner(object):
    """
    基础学习器类，用于定义和管理迁移学习或类Incremental学习任务中的基本操作和属性。

    参数:
    - args: 一个字典，包含各种配置参数，如学习任务数、随机种子、数据集名称等。

    属性:
    - _cur_task: 当前任务的索引。
    - _known_classes: 目前已知类的数量。
    - _total_classes: 总类数。
    - _network: 当前使用的网络模型。
    - _old_network: 之前任务的网络模型（如果适用）。
    - _data_memory, _targets_memory: 存储数据和目标的内存数组。
    - topk: 评估时考虑的top k结果。
    - wandb: 是否使用Wandb进行日志记录。
    - save_dir: 模型保存的目录。
    - dataset_name: 数据集的名称。
    - nums: 各任务类的数量。
    """

    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5
        self.args = args
        self.each_task = args["increment"]
        self.seed = args["seed"]
        self.tasks = args["tasks"]
        self.wandb = args["wandb"]
        self.save_dir = args["save_dir"]
        self.dataset_name = args["dataset"]
        self.nums = args["nums"]

        # 设置内存相关参数的默认值
        args["memory_size"] = 300
        args["memory_per_class"] = 20
        args["fixed_memory"] = False

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = "0"
        # self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        """
        获取示例存储的大小。

        条件:
        - _data_memory 和 _targets_memory 的长度必须相等。

        返回:
        - _targets_memory 的长度，即存储中示例的数量。
        """
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        """
        获取每个类的样本数量。

        条件:
        - 如果设置了固定内存大小 (_fixed_memory=True)，则返回每个类固定的样本数。
        - 否则，根据总内存大小和总类数计算平均每个类的样本数。

        返回:
        - 每个类的样本数量。
        """
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        """
        获取特征维度。

        条件:
        - 如果网络使用了DataParallel，从module中获取特征维度。
        - 否则，直接从网络中获取特征维度。

        返回:
        - 网络的特征维度。
        """
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def real_build_rehearsal_memory(self):
        """
        构建排练记忆库的实际实现。

        该方法应在子类中被重写，以提供具体的排练记忆库构建逻辑。
        """
        pass

    def combine_dataset(self, pre_dataset, cur_dataset, size):
        # correct
        idx = pre_dataset.idxs
        pre_labels = pre_dataset.dataset.labels[idx]  # label 22, wrong
        pre_data = pre_dataset.dataset.images[idx]

        idx = cur_dataset.idxs
        cur_labels = cur_dataset.dataset.labels[idx]
        cur_data = cur_dataset.dataset.images[idx]

        if size !=0:
            idxs = np.random.choice(range(len(pre_dataset.idxs)), size, replace=False)
            selected_exemplar_data, selected_exemplar_label = pre_data[idxs], pre_labels[idxs]
            
            combined_data = np.concatenate((cur_data, selected_exemplar_data),axis=0)
            combined_label = np.concatenate((cur_labels, selected_exemplar_label),axis=0)
            # combined_label = np.concatenate(combined_label)
            # idata = _get_idata(self.dataset_name)
            # _train_trsf, _common_trsf = idata.train_trsf, idata.common_trsf
            # trsf = transforms.Compose([*_train_trsf, *_common_trsf])      
            # combined_dataset = DummyDataset(combined_data, combined_label, trsf, use_path=False)
        else:
            combined_data = np.concatenate((cur_data, pre_data),axis=0)
            combined_label = np.concatenate((cur_labels, pre_labels),axis=0)
            # combined_data, combined_label = np.vstack((cur_dataset.images, pre_dataset.images)), np.vstack((cur_dataset.labels, pre_dataset.labels))
            # combined_label = np.concatenate(combined_label)
        idata = _get_idata(self.dataset_name)
        _train_trsf, _common_trsf = idata.train_trsf, idata.common_trsf
        trsf = transforms.Compose([*_train_trsf, *_common_trsf])      
        combined_dataset = DummyDataset(combined_data, combined_label, trsf, use_path=False)

        return combined_dataset

    
    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:  # false
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        # 使用 os.path.join 来构建完整的文件路径
        import os
        save_path = os.path.join("models_checkpoints", f"{filename}_{self._cur_task}.pkl")
        
        # 确保保存目录存在
        os.makedirs("models_checkpoints", exist_ok=True)
    
        torch.save(save_dict, save_path)


    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        """
        评估预测结果的准确性。

        参数:
        y_pred: 预测标签的二维数组，shape为(n_samples, n_classes)，其中n_samples为样本数量，n_classes为类别数量。
        y_true: 真实标签的一维数组，shape为(n_samples,)，其中n_samples为样本数量。

        返回值:
        一个字典，包含不同评估指标的结果。包括分组准确度（grouped）、第一位置的准确度（top1）和前k位置的准确度（topk）。
        """
        ret = {}  # 初始化返回结果的字典

        # 计算分组准确度
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, increment=self.each_task)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]  # 将总准确度作为top1

        # 计算前k位置的准确度
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        """
        评估模型任务的性能。

        使用CNN和NME（如果已定义）两种方法对测试集进行评估，计算并返回两种方法的准确性。

        返回:
            tuple: 包含两个元素的元组。第一个元素是CNN评估的准确性，第二个元素是NME评估的准确性（如果NME未定义，则为None）。
        """
        y_pred, y_true = self._eval_cnn(self.test_loader)  # 使用CNN评估模型预测的准确度
        cnn_accy = self._evaluate(y_pred, y_true)  # 计算并存储CNN评估的准确性

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)  # 如果定义了_class_means，使用NME评估模型预测的准确度
            nme_accy = self._evaluate(y_pred, y_true)  # 计算并存储NME评估的准确性
        else:
            nme_accy = None  # 如果没有定义_class_means，则NME准确性为None

        return cnn_accy, nme_accy  # 返回两种方法的评估结果

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        """
        计算模型在给定数据加载器上的准确率。

        参数:
        - model: 训练好的模型，用于进行预测。
        - loader: 数据加载器，包含用于评估的样本。

        返回:
        - 一个字符串，表示模型的准确率，格式为百分比。
        """
        model.eval()  # 将模型设置为评估模式
        correct, total = 0, 0  # 初始化正确预测数和总预测数

        for i, (_, inputs, targets) in enumerate(loader):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择使用CPU或GPU
            inputs = inputs.to(device)  # 将输入数据移动到指定设备
            with torch.no_grad():  # 禁止计算梯度
                outputs = model(inputs)["logits"]  # 获取模型的输出
            # 将outputs列表连接成一个tensor
            predicts = torch.max(outputs, dim=1)[1]  # 从模型输出中提取预测类别
            correct += (predicts.cpu() == targets).sum()  # 在CPU上计算正确预测的数量
            total += len(targets)  # 更新总预测数

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)  # 计算并返回准确率
    
    def _compute_accuracy1(self, model, loader):
        model.eval()
        correct, total = 0, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择使用CPU或GPU
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                if self._cur_task==0:
                    outputs=model(inputs)['new_logits']
                else:
                    logits = model(inputs)
                    outputs = logits["new_logits"]+logits["old_logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)
    
    def _eval_cnn1(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择使用CPU或GPU
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                if self._cur_task==0:
                    outputs=self._network(inputs)['new_logits']
                else:
                    logits = self._network(inputs)
                    outputs = logits["new_logits"]+logits["old_logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true) 
    def _eval_cnn(self, loader):
        """
        评估CNN模型的性能。

        参数:
        - loader: 数据加载器，用于加载测试数据集。

        返回:
        - y_pred: 预测结果的集合，包含了每个样本的前topk个预测类别。
        - y_true: 真实标签的集合，包含了每个样本的真实类别。
        """
        self._network.eval()  # 将网络设置为评估模式
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 根据硬件可用性选择设备
            inputs = inputs.to(device)  # 将输入数据移动到指定设备
            with torch.no_grad():  # 禁止计算梯度
                outputs = self._network(inputs)["logits"]  # 通过网络获取 logits
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # 获取每个样本的前topk个预测类别
            y_pred.append(predicts.cpu().numpy())  # 将预测结果转换为numpy数组并添加到集合中
            y_true.append(targets.cpu().numpy())  # 将真实标签转换为numpy数组并添加到集合中

        return np.concatenate(y_pred), np.concatenate(y_true)  # 将集合中的所有样本的预测结果和真实标签合并为单一数组

    def _eval_nme(self, loader, class_means):
        """
        评估网络的名称错误（Name Error）指标。

        参数:
        - loader: 数据加载器，用于加载和处理数据集。
        - class_means: 每个类的特征向量平均值，用于计算距离。

        返回:
        - predictions: 每个样本的前topk个最有可能的类别索引。
        - y_true: 实际的类别标签。
        """
        self._network.eval()  # 将网络设置为评估模式
        vectors, y_true = self._extract_vectors(loader)  # 提取特征向量和真实标签

        # 标准化特征向量
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        # 计算每个样本到每个类平均向量的平方欧几里得距离
        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], 距离越小，得分越高

        # 返回每个样本的前topk个类别索引
        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        """
        从给定的数据加载器中提取向量和目标。

        :param loader: 一个数据加载器，用于加载样本和对应的标签。
        :return: 一个元组，包含两个元素。第一个元素是所有样本的向量组成的numpy数组，第二个元素是所有样本的目标标签组成的numpy数组。
        """
        self._network.eval()  # 将网络设置为评估模式
        vectors, targets = [], []  # 初始化向量列表和目标列表

        for _, _inputs, _targets in loader:  # 遍历数据加载器中的每个批次
            _targets = _targets.numpy()  # 将目标标签转换为numpy数组
            if isinstance(self._network, nn.DataParallel):  # 如果网络是数据并行的
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(device))  # 在指定设备上提取向量，并转换为numpy数组
                )
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(device))  # 在指定设备上提取向量，并转换为numpy数组
                )

            vectors.append(_vectors)  # 将提取的向量添加到列表中
            targets.append(_targets)  # 将目标标签添加到列表中

        return np.concatenate(vectors), np.concatenate(targets)  # 将向量列表和目标列表合并为单一的numpy数组

    def _reduce_exemplar(self, data_manager, m):
        print("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )   # empty list
        self._class_means = np.zeros((self._total_classes, self.feature_dim)) # shape, (20, 64)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        # for each old class, xx
        for class_idx in range(self._known_classes):  # 0 for the first task
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        print("Constructing exemplars...({} per classes)".format(m))
        # for current task
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )   # return dataset for one class, 500 samples
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)  # get feature maps
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)  # (100, 32, 32, 3)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        print(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means




