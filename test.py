import pickle

file_path = '/users/jty/deepconf/online-dpsk/deepthink_online_qid0_rid0_20250921_214220.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data)  # 查看文件内容