from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

file_path = 'online-dpsk/deepthink_online_qid0_rid0_20250922_130404.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())  # 查看文件内容
print(type(data["final_traces"]))
print(data["final_traces"][0].keys())
correct_list = []
reasoning_chains = []
for i in range(len(data["final_traces"])):
    if data["final_traces"][i]['extracted_answer'] and len(data["final_traces"][i]['extracted_answer']) < 10:
        correct_list.append((i, data["final_traces"][i]["num_tokens"], data["final_traces"][i]['extracted_answer']))
        reasoning_chains.extend(data["final_traces"][i]["text"].split("\n"))
print(correct_list)
# print(reasoning_chains)
reasoning_chains = [s for s in reasoning_chains if s.strip() != ""]
reasoning_chains = reasoning_chains[:2000]
# 1. 加载多语言模型
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 2. 句子向量
embeddings = model.encode(reasoning_chains, convert_to_tensor=False)

# 3. 聚类
clustering = AgglomerativeClustering(
    n_clusters=10,
    # distance_threshold=0.8, # 相似度阈值
    metric="cosine",
    linkage="average"
)
labels = clustering.fit_predict(embeddings)

# 4. 整理结果
clusters = {}
for idx, label in enumerate(labels):
    clusters.setdefault(label, []).append(idx)

# 5. 选取每个簇的代表性句子（中心句）
print("句子聚类结果（含代表性句子）：")
for label, indices in clusters.items():
    cluster_embs = embeddings[indices]
    centroid = np.mean(cluster_embs, axis=0, keepdims=True)
    sims = cosine_similarity(cluster_embs, centroid).reshape(-1)
    rep_idx = indices[np.argmax(sims)]  # 找最接近中心的句子
    print(f"\n簇 {label} (代表: {reasoning_chains[rep_idx]})")
    for i in indices:
        print(" -", reasoning_chains[i])
