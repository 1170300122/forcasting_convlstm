import pickle
import pandas as pd
# 查看pkl文件(我后加的)
# f = open('hko_data/pd/hko7_rainy_train.pkl','rb')
# data = pickle.load(f)
d = pd.read_pickle('hko_data/pd/hko7_all.pkl')
print(d)