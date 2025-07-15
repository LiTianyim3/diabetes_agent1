QUESTIONS = [
    "请问您的年龄和性别？格式示例：{'age':45,'gender':'男'}",
    "是否有多饮、多尿、多食、多瘦症状？请简要描述。",
    "家族中是否有糖尿病患者？（有/无）",
    "是否长期服用影响血糖的药物？如糖皮质激素。"
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
