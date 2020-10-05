import pickle
import pandas as pd
import sys
import os
sys.path.insert(0, '../../')
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path)
# 查看pkl文件(我后加的)
# f = open('hko_data/pd/hko7_rainy_train.pkl','rb')
# data = pickle.load(f)
d = pd.read_pickle('convLSTM_balacned_mse_mae.pkl')
print(d.)