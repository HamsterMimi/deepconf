import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

file_path = 'online-dpsk/deepthink_online_qid0_rid0_20250922_130404.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data["all_traces"][0].keys())

# 分别收集有答案和无答案的置信度
answer_confs_list = []
no_answer_confs_list = []

for trace in data["all_traces"]:
    confs = trace["confs"][:6000]  # 取前1000个置信度值
    # 检查是否有有效答案
    if trace["extracted_answer"] == "2^{99}":
        answer_confs_list.append(confs)
    else:
        no_answer_confs_list.append(confs)

print(f"有答案样本数量: {len(answer_confs_list)}")
print(f"无答案样本数量: {len(no_answer_confs_list)}")

def extract_statistical_features(confs_list):
    """从置信度序列中提取统计特征"""
    features = []
    for confs in confs_list:
        if len(confs) > 0:
            features.append([
                np.mean(confs),           # 平均值
                np.std(confs),            # 标准差
                np.min(confs),            # 最小值
                np.max(confs),            # 最大值
                np.median(confs),         # 中位数
                np.percentile(confs, 25), # 25%分位数
                np.percentile(confs, 75), # 75%分位数
                np.percentile(confs, 90), # 90%分位数
                np.var(confs),            # 方差
                np.mean(np.diff(confs)),  # 一阶差分的平均值
                np.std(np.diff(confs)),   # 一阶差分的标准差
                len([x for x in confs if x > 0.5]),  # 大于0.5的置信度数量
                len([x for x in confs if x > 0.8]),  # 大于0.8的置信度数量
            ])
        else:
            # 如果序列为空，用0填充所有特征
            features.append([0] * 13)
    return np.array(features)

# 提取统计特征
X_answer = extract_statistical_features(answer_confs_list)
X_no_answer = extract_statistical_features(no_answer_confs_list)

# 创建标签
y_answer = np.ones(len(X_answer))  # 标签1表示有答案
y_no_answer = np.zeros(len(X_no_answer))  # 标签0表示无答案

# 合并数据集
X = np.vstack((X_answer, X_no_answer))
y = np.hstack((y_answer, y_no_answer))

print(f"总样本数量: {X.shape[0]}")
print(f"特征数量: {X.shape[1]}")

# # 定义特征名称
feature_names = [
    'mean',
    'std',
    'min',
    'max',
    'median',
    'q25',
    'q75',
    'q90',
    'variance',
    'mean_diff',
    'std_diff',
    'count_gt_0.5',
    'count_gt_0.8'
]

# 创建DataFrame便于查看
df = pd.DataFrame(X, columns=feature_names)
df['label'] = y
print("\n数据集统计信息:")
print(df.describe())

# 检查类别平衡
print(f"\n类别分布:")
print(f"有答案样本: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
print(f"无答案样本: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练SVM模型
print("\n训练SVM模型...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale',
                probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 预测
y_pred = svm_model.predict(X_test_scaled)
y_pred_proba = svm_model.predict_proba(X_test_scaled)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred,
                           target_names=['无答案', '有答案'],
                           digits=4))
