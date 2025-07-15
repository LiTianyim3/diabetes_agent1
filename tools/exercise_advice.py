from client.zhipu_llm import ZhipuLLM

def gen_exercise_advice(severity: str, prefs: dict) -> str:
    prompt = (
        f"你是运动健康师，"
        f"为“{severity}”糖尿病患者，"
        f"结合喜好{prefs}，设计一周运动处方，"
        "包括频率、时长、强度和安全注意事项。"
    )
    return ZhipuLLM()._call(prompt)
