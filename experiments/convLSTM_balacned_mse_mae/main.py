import sys
import os
sys.path.insert(0, '../../')
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path)
import torch
from nowcasting.config import cfg
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from nowcasting.models.model import EF
from torch.optim import lr_scheduler
from nowcasting.models.loss import Weighted_mse_mae
from nowcasting.train_and_test import train_and_test

from experiments.net_params import convlstm_encoder_params, convlstm_forecaster_params


# 此文件中内容为convlstm的实例
### Config

# 参数初始化

batch_size = cfg.GLOBAL.BATCH_SZIE
max_iterations = 100000 # 总迭代轮数
test_iteration_interval = 1000 # 隔多少轮打印模型状态
test_and_save_checkpoint_iterations = 1000 # 多少轮一保存
LR_step_size = 20000
gamma = 0.7

LR = 1e-4

# 损失函数定义,这里选取的是均方误差与平均绝对误差
criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)
# 编码器的选取,其中covlstm_encoder_params列表为卷积lstm模型的参数设置及实例化(在net_params.py中定义)
encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
# 预测器的选取
forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

# 使用EF类将编码器与预测器进行拼接
encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

# 挑选优化器
optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR)
# 设置学习率步长
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)

folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]

# 调用训练测试方法对模型进行训练与测试
train_and_test(encoder_forecaster, optimizer, criterion, exp_lr_scheduler, batch_size, max_iterations, test_iteration_interval, test_and_save_checkpoint_iterations, folder_name)