import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# 数据集路径
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_prediction_dataset.csv')

# .pkl 文件路径
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'encoder.pkl'
DATASET_PKL_PATH = 'dataset.pkl'

# 数值和分类特征列
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
cat_cols = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes']

# 特征预处理：标准化数值特征，编码分类特征
def preprocess_features(df):
    for col in num_cols + cat_cols:
        if col not in df.columns:
            raise KeyError(f"列名 '{col}' 不在数据集，请检查数据集实际列名！")
    scaler = StandardScaler()
    num_data = scaler.fit_transform(df[num_cols].astype(float))
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_data = encoder.fit_transform(df[cat_cols].astype(str))
    feature_matrix = np.hstack([num_data, cat_data])
    return feature_matrix, scaler, encoder

# 检查并加载或生成 .pkl 文件
def load_or_preprocess_data():
    global df
    if os.path.exists(DATASET_PKL_PATH):
        print(f"找到现有数据集文件 {DATASET_PKL_PATH}，正在加载...")
        df = pd.read_pickle(DATASET_PKL_PATH)
    else:
        print(f"未找到数据集文件 {DATASET_PKL_PATH}，正在从 CSV 加载...")
        df = pd.read_csv(DATASET_PATH)
        df.to_pickle(DATASET_PKL_PATH)
        print(f"数据集已保存至 {DATASET_PKL_PATH}")

    print("数据集列名:", df.columns.tolist())

    if os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
        print(f"找到现有预处理器文件 {SCALER_PATH} 和 {ENCODER_PATH}，正在加载...")
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        # 重新生成 feature_matrix 以确保一致性
        num_data = scaler.transform(df[num_cols].astype(float))
        cat_data = encoder.transform(df[cat_cols].astype(str))
        feature_matrix = np.hstack([num_data, cat_data])
    else:
        print(f"未找到预处理器文件，正在生成新的 scaler 和 encoder...")
        feature_matrix, scaler, encoder = preprocess_features(df)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        print(f"预处理器已保存至 {SCALER_PATH} 和 {ENCODER_PATH}")

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

# 生成或加载病例嵌入
def load_or_generate_embeddings():
    EMBEDDINGS_PATH = 'case_embeddings.pt'
    if os.path.exists(EMBEDDINGS_PATH):
        print(f"找到现有嵌入文件 {EMBEDDINGS_PATH}，正在加载...")
        case_embeddings = torch.load(EMBEDDINGS_PATH, map_location=device)
        case_texts = list(df.apply(case_to_text, axis=1))
    else:
        print("正在生成病例文本描述和嵌入，请稍候...")
        case_texts = []
        case_embeddings = []
        for row in df.itertuples(index=False):
            text = case_to_text(row)
            case_texts.append(text)
            case_embeddings.append(bert_model.encode(text, convert_to_tensor=True, device=device))
        case_embeddings = torch.stack(case_embeddings).to(device)
        torch.save(case_embeddings, EMBEDDINGS_PATH)
        print(f"病例嵌入已保存至 {EMBEDDINGS_PATH}")
    return case_texts, case_embeddings

# 加载数据集和预处理器
feature_matrix, scaler, encoder = None, None, None
case_texts, case_embeddings = None, None

def ensure_loaded():
    global feature_matrix, scaler, encoder, case_texts, case_embeddings, df
    if feature_matrix is None or scaler is None or encoder is None:
        feature_matrix, scaler, encoder = load_or_preprocess_data()
    if case_texts is None or case_embeddings is None:
        case_texts, case_embeddings = load_or_generate_embeddings()

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

# 查找相似病例（基于大模型的余弦相似度）
def find_similar_cases_semantic(user_features: dict, top_k=200, similarity_threshold=0.7):
    _, user_text = vectorize_user_features(user_features)
    user_embedding = bert_model.encode(user_text, convert_to_tensor=True, device=device)
    
    # 计算余弦相似度
    similarities = util.cos_sim(user_embedding, case_embeddings)[0].cpu().numpy()
    
    # 筛选满足相似度阈值的病例
    valid_indices = [(i, sim) for i, sim in enumerate(similarities) if sim >= similarity_threshold]
    if not valid_indices:
        print(f"没有找到相似度大于 {similarity_threshold} 的病例！")
        return [], [], []
    
    # 按相似度降序排序
    valid_indices = sorted(valid_indices, key=lambda x: x[1], reverse=True)[:top_k]
    valid_I, valid_similarities = zip(*valid_indices)
    
    results = df.iloc[list(valid_I)].to_dict(orient='records')
    return results, np.array(valid_similarities), np.array(valid_I)

# 生成科学建议
def generate_scientific_advice(user_features: dict, top_k=200, similarity_threshold=0.7):
    ensure_loaded()
    cases, similarities, indices = find_similar_cases_semantic(user_features, top_k, similarity_threshold)
    if not cases:
        return "未找到符合相似度阈值的病例！"
    
    diabetes_count = sum(1 for case in cases if case['diabetes'] == 1)
    non_diabetes_count = len(cases) - diabetes_count
    
    print(f"相似病例统计：糖尿病 {diabetes_count} 例，非糖尿病 {non_diabetes_count} 例")
    print(f"相似度范围：{min(similarities):.3f} - {max(similarities):.3f}")
    
    return f"相似病例统计：糖尿病 {diabetes_count} 例，非糖尿病 {non_diabetes_count} 例"

# 测试代码
if __name__ == "__main__":
    test_features = {
        'gender': "Female",
        'age': 52,
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
    print(generate_scientific_advice(test_features, similarity_threshold=0.7))