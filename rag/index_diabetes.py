



import os
import pandas as pd
import numpy as np

# 加载糖尿病预测数据集
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_prediction_dataset.csv')
df = pd.read_csv(DATASET_PATH)

def find_similar_cases(user_features: dict, top_k=200):
    """
    根据用户输入的医学指标，检索最相似的历史病例。
    user_features: {'Age': 50, 'BMI': 24.5, 'Glucose': 7.2, ...}
    返回 top_k 个最相似病例及诊断结果。
    """
    # 只选用数值型特征做简单欧氏距离检索
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 自动检测诊断结果列名
    result_col = None
    for col in ['Outcome', 'Diabetes', 'Result', 'diagnosis', 'Label']:
        if col in num_cols:
            result_col = col
            break
    # 增加性别等非数值型特征用于展示
    all_cols = list(df.columns)
    feature_cols = [col for col in num_cols if col != result_col]
    # 特征名映射（支持大写、别名、中文）
    key_map = {k.lower(): k for k in feature_cols}
    # 支持常见别名和中文
    alias_map = {
        'age': ['age', '年龄', 'Age'],
        'bmi': ['bmi', '体重指数', 'BMI'],
        'blood_glucose_level': ['blood_glucose_level', '血糖', '血糖值', 'glucose', 'Glucose'],
        'hba1c_level': ['hba1c_level', '糖化血红蛋白', 'HbA1c'],
        'hypertension': ['hypertension', '高血压'],
        'heart_disease': ['heart_disease', '心脏病'],
        'smoking_history': ['smoking_history', '吸烟史'],
        'gender': ['gender', 'sex', '性别'],
    }
    # 自动将用户输入key映射为数据集实际列名
    mapped_features = {}
    for col in feature_cols:
        # 先查别名
        for alias in alias_map.get(col.lower(), [col]):
            for k in user_features:
                if k.lower() == alias.lower():
                    mapped_features[col] = user_features[k]
        # 若无别名则尝试直接匹配
        if col not in mapped_features and col in user_features:
            mapped_features[col] = user_features[col]
    # 构造用户特征向量
    user_vec = np.array([mapped_features.get(col, np.nan) for col in feature_cols])
    data_mat = df[feature_cols].values
    # 处理缺失值（简单填充均值）
    for i, v in enumerate(user_vec):
        if np.isnan(v):
            user_vec[i] = np.nanmean(data_mat[:,i])

    # 优先筛选年龄接近的病例（±8岁）
    age_col = None
    for col in feature_cols:
        if col.lower() in ['age', '年龄']:
            age_col = col
            break
    filtered_idx = None
    if age_col and age_col in mapped_features:
        user_age = mapped_features[age_col]
        # 只保留年龄在±8岁内的病例
        age_arr = df[age_col].values
        close_mask = np.abs(age_arr - user_age) <= 8
        if np.any(close_mask):
            data_mat_close = data_mat[close_mask]
            dists_close = np.linalg.norm(data_mat_close - user_vec, axis=1)
            idxs_close = np.argsort(dists_close)[:top_k]
            idxs = np.where(close_mask)[0][idxs_close]
            # 如果数量不足top_k，再补全
            if len(idxs) < top_k:
                dists_all = np.linalg.norm(data_mat - user_vec, axis=1)
                idxs_all = np.argsort(dists_all)
                # 去重
                idxs_extra = [i for i in idxs_all if i not in idxs][:top_k-len(idxs)]
                idxs = np.concatenate([idxs, idxs_extra])
        else:
            # 没有接近年龄的，退化为全局最相似
            dists = np.linalg.norm(data_mat - user_vec, axis=1)
            idxs = np.argsort(dists)[:top_k]
    else:
        # 没有年龄信息，直接全局最相似
        dists = np.linalg.norm(data_mat - user_vec, axis=1)
        idxs = np.argsort(dists)[:top_k]
    # 展示时增加性别等原始字段
    select_cols = feature_cols.copy()
    # 增加性别等非数值型特征用于展示
    for col in ['gender', 'sex', '性别']:
        if col in all_cols and col not in select_cols:
            select_cols.insert(0, col)
    if result_col:
        select_cols.append(result_col)
    results = df.iloc[idxs][select_cols].to_dict(orient='records')
    return results

def generate_scientific_advice(user_features: dict):
    """
    综合相似病例和数据集，生成科学严谨的糖尿病管理建议，可供大模型参考。
    """
    cases = find_similar_cases(user_features, top_k=200)
    advice = []
    # 自动检测诊断结果列名
    result_col = None
    if cases and isinstance(cases[0], dict):
        for col in ['Outcome', 'Diabetes', 'Result', 'diagnosis', 'Label']:
            if col in cases[0]:
                result_col = col
                break
    for i, case in enumerate(cases):
        summary = ', '.join([f"{k}:{v}" for k, v in case.items() if k != result_col])
        if result_col:
            val = case[result_col]
            # 二分类情况
            if val in [0, 1]:
                diag = '糖尿病' if val == 1 else '非糖尿病'
            else:
                diag = str(val)
            advice.append(f"相似病例{i+1}：{summary}，诊断结果：{diag}")
        else:
            advice.append(f"相似病例{i+1}：{summary}")
    return "\n".join(advice)

# RAG功能测试代码
if __name__ == "__main__":
    # 构造测试特征（支持中英文、别名、大写混输）
    test_features = {
        '年龄': 50,
        'BMI': 50.5,
        '血糖':12.4,
        'HbA1c': 20,
    }
    print("测试特征:", test_features)
    print("--- RAG检索结果 ---")
    print(generate_scientific_advice(test_features))
