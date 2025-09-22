import pickle
import numpy as np
import matplotlib.pyplot as plt

file_path = 'online-dpsk/deepthink_online_qid0_rid0_20250922_130404.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())  # 查看文件内容
print(data["all_traces"][0].keys())

# 分别收集有答案和无答案的置信度
answer_confs_list = []
no_answer_confs_list = []


for trace in data["all_traces"]:
    confs = trace["confs"][:1000]
    # 检查是否有有效答案
    if trace["extracted_answer"] == "2^{99}":
        answer_confs_list.append(confs)
    else:
        no_answer_confs_list.append(confs)



# for trace in data["all_traces"]:
#     mean_confs = np.mean(trace["confs"])
#     # 检查是否有有效答案
#     if trace["extracted_answer"] == "2^{99}":
#         answer_confs_list.append(mean_confs)
#     else:
#         no_answer_confs_list.append(mean_confs)
#
# print(f"answer: {len(answer_confs_list)}")
# print(f"no_answer: {len(no_answer_confs_list)}")
#
# # 计算统计信息
# print(f"answer mean: {np.mean(answer_confs_list):.4f}")
# print(f"no answer mean: {np.mean(no_answer_confs_list):.4f}")
#
# # 使用箱线图或小提琴图可能更适合比较两组数据的分布
# plt.style.use('seaborn-v0_8-whitegrid')
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 创建数据列表和标签
# data_to_plot = [answer_confs_list, no_answer_confs_list]
# labels = ['yes', 'no']
#
# # 创建箱线图
# boxplot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
#
# # 设置颜色
# colors = ['lightgreen', 'lightcoral']
# for patch, color in zip(boxplot['boxes'], colors):
#     patch.set_facecolor(color)
#
# # 美化图表
# ax.set_ylabel('mean_conf', fontsize=12)
# ax.set_title('yes/no', fontsize=14)
# ax.grid(True, linestyle='--', alpha=0.7)
#
# # 添加数据点（可选，可以显示数据分布）
# for i, group in enumerate(data_to_plot):
#     # 添加一些随机抖动以避免重叠
#     x = np.random.normal(i+1, 0.04, size=len(group))
#     ax.plot(x, group, 'o', alpha=0.5, color='gray', markersize=4)
#
# plt.tight_layout()
#
# # 保存图片
# plt.savefig("answer_no_answer_confs.png", dpi=300)
# plt.show()
#
# # 如果您仍然想要柱状图，这里是一个替代方案（显示平均值）
# fig, ax = plt.subplots(figsize=(8, 5))
# means = [np.mean(answer_confs_list), np.mean(no_answer_confs_list)]
# stds = [np.std(answer_confs_list), np.std(no_answer_confs_list)]
#
# bars = ax.bar(labels, means, yerr=stds, capsize=10,
#               color=['lightgreen', 'lightcoral'], alpha=0.7)
#
# ax.set_ylabel('mean_confs', fontsize=12)
# ax.set_title('yes/no', fontsize=14)
# ax.grid(True, linestyle='--', alpha=0.7, axis='y')
#
# # 在柱子上添加数值
# for bar, mean in zip(bars, means):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
#             f'{mean:.4f}', ha='center', va='bottom', fontsize=12)
#
# plt.tight_layout()
# plt.savefig("answer_no_answer_mean_confs.png", dpi=300)
# plt.show()

