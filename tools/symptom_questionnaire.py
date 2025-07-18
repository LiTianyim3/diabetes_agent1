QUESTIONS = [
    "请您如实填写：年龄（数字）和性别（男/女）。格式示例：{'age':45,'gender':'男'}",
    "您近期是否有以下症状？多饮、多尿、多食、体重下降等。请简要描述或填写'无'。",
    "您的直系亲属（父母、兄弟姐妹）中是否有人患有糖尿病？请回答'有'或'无'。",
    "您是否长期服用影响血糖的药物（如糖皮质激素等）？请简要说明药物名称及用药情况，或填写'无'。"
]

def run_questionnaire(answer_history: list) -> dict:
    """
    answer_history: [{'q':..., 'a':...}, ...]
    返回最新收集到的字段，比如 {'age':45, 'family_history':True, ...}
    """
    data = {}
    for qa in answer_history:
        # 简单映射：真实项目里可用正则或 JSON parse
        if "age" in qa["a"]:
            import json; data.update(json.loads(qa["a"]))
        if "家族" in qa["q"]:
            data["family_history"] = ("有" in qa["a"])
        # ……按需扩展
    return data
