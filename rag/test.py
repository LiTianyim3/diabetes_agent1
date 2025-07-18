import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# 数据集路径
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_prediction_dataset.csv')
df = pd.read_csv(DATASET_PATH)

# 打印实际列名，调试用
print("数据集列名:", df.columns.tolist())

# 数值和分类特征列
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
cat_cols = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes']

# 特征预处理：标准化数值特征，编码分类特征
def preprocess_features(df):
    # 检查列是否存在
    for col in num_cols + cat_cols:
        if col not in df.columns:
            raise KeyError(f"列名 '{col}' 不在数据集，请检查数据集实际列名！")
    # 标准化数值特征
    scaler = StandardScaler()
    num_data = scaler.fit_transform(df[num_cols].astype(float))
    # 分类特征编码
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_data = encoder.fit_transform(df[cat_cols].astype(str))
    # 拼接
    feature_matrix = np.hstack([num_data, cat_data])
    return feature_matrix, scaler, encoder

# 初始化模型，自动选择设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

print("正在加载语义模型，请稍候...")
bert_model = SentenceTransformer(r'D:\edge_download\all-MiniLM-L6-v2', device=device)
print("语义模型加载完成。")

# 生成病例文本描述
def case_to_text(row):
    def get_value(obj, col):
        if isinstance(obj, dict):
            return obj.get(col, '')
        else:
            return getattr(obj, col, '')
    return ', '.join([f"{col}:{get_value(row, col)}" for col in num_cols + cat_cols])

print("正在生成病例文本描述和嵌入，请稍候...")
case_texts = []
for row in tqdm(df.itertuples(index=False), total=len(df), desc="生成文本"):
    case_texts.append(case_to_text(row))
case_embeddings = bert_model.encode(case_texts, convert_to_numpy=True, show_progress_bar=True)
print("病例嵌入生成完成。")

# 预处理数据集，获取特征矩阵和预处理器
feature_matrix, scaler, encoder = preprocess_features(df)
cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()

# 构建 FAISS 索引（使用 IndexIVFFlat 优化性能）
embedding_dim = case_embeddings.shape[1]
nlist = 100  # 聚类数量
quantizer = faiss.IndexFlatL2(embedding_dim)
faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
faiss_index.train(case_embeddings)
faiss_index.add(case_embeddings)
faiss_index.nprobe = 10  # 搜索时检查的聚类数量
print(f"FAISS 索引已创建，包含 {faiss_index.ntotal} 条记录")

# 保存索引和预处理器
faiss.write_index(faiss_index, 'diabetes_index.faiss')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')
df.to_pickle('dataset.pkl')
print("索引和预处理器已保存。")

# 向量化用户特征
def vectorize_user_features(user_features):
    alias_map = {
        '年龄': 'age', 'age': 'age',
        'BMI': 'bmi', 'bmi': 'bmi',
        '血糖': 'blood_glucose_level', 'blood_glucose_level': 'blood_glucose_level',
        'HbA1c': 'HbA1c_level', 'hba1c': 'HbA1c_level', 'HbA1c_level': 'HbA1c_level',
        '糖尿病': 'diabetes', 'diabetes': 'diabetes',
        '性别': 'gender', 'gender': 'gender',
        '高血压': 'hypertension', 'hypertension': 'hypertension',
        '心脏病': 'heart_disease', 'heart_disease': 'heart_disease',
        '吸烟史': 'smoking_history', 'smoking_history': 'smoking_history'
    }
    num_vec = []
    for col in num_cols:
        v = None
        for k, v_ in user_features.items():
            if alias_map.get(k.lower(), k.lower()) == col:
                v = v_
                break
        if v is None or pd.isna(v):
            v = float(df[col].mean())
        else:
            v = float(v)
            # 修改：不合理值自动填充均值，不报错
            if col == 'age' and (v < 0 or v > 120):
                v = float(df[col].mean())
            if col == 'bmi' and (v < 10 or v > 50):
                v = float(df[col].mean())
            if col == 'HbA1c_level' and (v < 0 or v > 20):
                v = float(df[col].mean())
            if col == 'blood_glucose_level' and (v < 0 or v > 500):
                v = float(df[col].mean())
        v = (v - df[col].mean()) / df[col].std()
        num_vec.append(v)
    
    cat_df = pd.DataFrame({col: [user_features.get(col, df[col].mode()[0])] for col in cat_cols})
    cat_vec = encoder.transform(cat_df.astype(str))
    return np.hstack([num_vec, cat_vec[0]]).reshape(1, -1), case_to_text(user_features)

# 查找相似病例
def find_similar_cases_semantic(user_features: dict, top_k=200):
    user_vec, user_text = vectorize_user_features(user_features)
    user_embedding = bert_model.encode([user_text], convert_to_numpy=True)
    D, I = faiss_index.search(user_embedding, top_k)
    results = df.iloc[I[0]].to_dict(orient='records')
    return results, D[0], I[0]

# 生成科学建议
def generate_scientific_advice(user_features: dict, top_k=200):
    cases, distances, indices = find_similar_cases_semantic(user_features, top_k)
    diabetes_count = 0
    non_diabetes_count = 0
    result_col = 'diabetes'
    for case in cases:
        if case[result_col] == 1:
            diabetes_count += 1
        else:
            non_diabetes_count += 1
    # 只打印统计结果
    print(f"相似病例统计：糖尿病 {diabetes_count} 例，非糖尿病 {non_diabetes_count} 例")
    # 返回统计信息给app.py
    return f"相似病例统计：糖尿病 {diabetes_count} 例，非糖尿病 {non_diabetes_count} 例"

# 测试代码
if __name__ == "__main__":
    test_features = {
        'gender': "Female",
        'age': 50,
        'hypertension': 0,
        'heart_disease': 1,
        'smoking_history': "never",
        'bmi': 25.19,
        'HbA1c_level': 6.6,
        'blood_glucose_level': 140,
        'diabetes': 0
    }
    print("测试特征:", test_features)
    print("--- RAG检索统计结果 ---")
    # 只打印统计结果，不打印详细病例
    print(generate_scientific_advice(test_features))