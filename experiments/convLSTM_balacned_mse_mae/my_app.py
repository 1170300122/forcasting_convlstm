import sys
import os
sys.path.insert(0, '../../')
o_path = os.getcwd() # 返回当前工作目录
print(o_path)
sys.path.append(o_path)
import torch
from nowcasting.config import cfg
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from nowcasting.models.model import EF
from torch.optim import lr_scheduler
from nowcasting.models.loss import Weighted_mse_mae
from nowcasting.train_and_test import train_and_test
import joblib
from experiments.net_params import convlstm_encoder_params, convlstm_forecaster_params
import numpy as np
from nowcasting.hko import image
from nowcasting.hko.evaluation import HKOEvaluation
# from nowcasting.helpers.visualization import save_hko_movie

convlstm_encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
convlstm_forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
convlstm_encoder_forecaster = EF(convlstm_encoder, convlstm_forecaster).to(cfg.GLOBAL.DEVICE)
# convlstm_encoder_forecaster.load_state_dict(joblib.load('convLSTM_balacned_mse_mae.pkl'))
net = joblib.load('convLSTM_balacned_mse_mae.pkl')
print(net)
batch_size = 2
seq_lenth = 10
height = 100
width = 100
train_data = np.zeros((seq_lenth, batch_size, 1, height, width),
                                  dtype=np.uint8)

hit_inds = []
for i in range(0,seq_lenth):
    for j in range(0,batch_size):
        hit_inds.append([i, j])
paths = []
for i in range(1, seq_lenth * 2 + 1):
    paths.append('new_data_set\\'+str(i)+'.png')
all_frame_dat = image.quick_read_frames(path_list=paths,
                                                    im_h=height,
                                                    im_w=width,
                                                    grayscale=True)
hit_inds = np.array(hit_inds, dtype=np.int)
print(hit_inds)
print(all_frame_dat.shape)
train_data[hit_inds[:, 0], hit_inds[:, 1], :, :, :] = all_frame_dat
output = net.predict(train_data)
# label = valid_label[:, 0, 0, :, :]
# output = output[:, 0, 0, :, :]
# mask = mask[:, 0, 0, :, :].astype(np.uint8)
# save_hko_movie(label, sample_datetimes[0], mask, masked=True,
#                        save_path='ground_truth.mp4')
# save_hko_movie(output, sample_datetimes[0], mask, masked=True,
#                save_path='pred.mp4')

