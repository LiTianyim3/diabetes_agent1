from client.zhipu_llm import ZhipuLLM

def gen_nutrition_advice(severity: str, prefs: dict) -> str:
    prompt = (
        f"你是注册营养师，"
        f"为“{severity}”糖尿病患者，"
        f"根据偏好{prefs}，设计7天饮食方案，"
        "每餐给出食材、重量及注意事项。"
    )
    return ZhipuLLM()._call(prompt)
