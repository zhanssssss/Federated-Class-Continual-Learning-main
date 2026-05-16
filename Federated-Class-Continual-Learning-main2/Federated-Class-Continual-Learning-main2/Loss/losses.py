"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
# 验证输入特征features的维度至少为3（批次大小、视图数、特征维度）。
# 将特征向量展平（如果有必要）以简化处理。
# 处理标签labels或掩码mask以创建一个适用于所有正样本对的掩码。如果提供了labels，则使用标签信息生成掩码；如果直接提供了mask，则直接使用。
# 根据contrast_mode确定锚点特征集合。
# 计算锚点特征和所有特征之间的点积，并除以温度系数以计算对数概率。
# 应用掩码以忽略自对比的情况，并计算正样本对的对数概率的均值。
# 计算最终的对比损失，可能根据target_labels进一步筛选特定类别的损失。
# 根据reduction参数决定损失的缩减方式。
class SupConLoss(nn.Module):
    """
    参数:
    - temperature (float): 对比学习中使用的温度参数，默认为0.07。
    - contrast_mode (str): 对比模式，决定如何选择样本进行对比，'all'表示所有样本都用于对比，默认为'all'。
    - base_temperature (float): 基础温度参数，用于微调温度，默认为0.07。
    """

    def __init__(self, temperature=0.14, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature  # 设置温度参数
        self.contrast_mode = contrast_mode  # 设置对比模式
        self.base_temperature = base_temperature  # 设置基础温度参数

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
        """
        参数:
            features: 形状为[bsz, n_views, ...]的隐藏向量。
            labels: 真实标签，形状为[bsz]。
            mask: 对比掩码，形状为[bsz, bsz]，如果样本j与样本i具有相同类别，则mask_{i,j}=1。
                 可以是不对称的。
        返回值:
            一个损失标量。
        """
        assert target_labels is not None and len(
            target_labels) > 0, "Target labels should be given as a list of integer"
        # 确定计算设备
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        # 检查features的维度是否符合要求
        if len(features.shape) < 3:
            raise ValueError('`features`需要是[bsz, n_views, ...]的形状，'
                             '至少需要3个维度')
        if len(features.shape) > 3:
            # 如果维度大于3，将features压缩为[bsz, n_views, -1]的形状
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # 判断labels和mask是否同时被定义
        if labels is not None and mask is not None:
            raise ValueError('不能同时定义`labels`和`mask`')
        elif labels is None and mask is None:
            # 如果两者都未定义，使用单位矩阵作为掩码
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # 如果只定义了labels，根据labels生成掩码
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('标签数量与特征数量不匹配')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            # 如果只定义了mask，直接将其转换为float类型并移动到相应的设备上
            mask = mask.float().to(device)

        # 初始化对比度计数和对比度特征
        contrast_count = features.shape[1]  # 对比度特征的维度数
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 将特征展开并重新组合以获取对比度特征

        # 根据对比模式选择锚点特征和锚点计数
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # 选择第一个特征作为锚点
            anchor_count = 1  # 锚点计数为1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # 使用所有对比度特征作为锚点
            anchor_count = contrast_count  # 锚点计数为对比度特征的数量
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))  # 如果模式未知，则抛出异常

        # 计算logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)  # 计算锚点和对比度特征的点积，然后除以温度参数
        # 为了数值稳定性，对logits进行规范化处理
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 在每行中找到最大logits值
        logits = anchor_dot_contrast - logits_max.detach()  # 减去最大logits值以规范化

        # 对mask进行重复，并屏蔽自我对比的情况
        mask = mask.repeat(anchor_count, contrast_count)  # 重复mask以匹配logits维度
        # 创建一个屏蔽矩阵，防止锚点与自身对比
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # 应用屏蔽

        # 计算log_prob
        exp_logits = torch.exp(logits) * logits_mask  # 应用屏蔽后，计算每个logits的指数并相加
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 计算log_prob

        """
        计算并返回经过调整的平均对数概率损失。

        参数:
        - mask: 一个二进制掩码，用于指示哪些是正样本。
        - log_prob: 对数概率，来自模型的预测。
        - labels: 输入样本的实际标签。
        - target_labels: 目标标签的集合，用于计算损失。
        - self.temperature: 用于调整损失的温度参数。
        - self.base_temperature: 基础温度参数，用于进一步调整温度。
        - device: 指定计算设备（如"cpu", "cuda"）。
        - reduction: 指定损失的减少方式，可选值为'none'、'mean'。

        返回值:
        - 计算得到的损失值，根据reduction参数进行了相应的减少处理。
        """

        # 计算正样本的对数概率的平均值
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 根据温度参数调整损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        # 为每个目标标签创建当前类别掩码
        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels:
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        # 应用类别掩码到损失上
        loss = curr_class_mask * loss.view(anchor_count, batch_size)

        # 根据指定的减少方式处理损失
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss
