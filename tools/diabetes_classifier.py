from typing import Dict

def classify_diabetes(data: Dict) -> Dict:
    """返回 {'is_diabetes': bool, 'reason': str}"""
    fpg  = data.get("fasting_glucose")
    hba1c= data.get("hba1c")
    ogtt = data.get("ogtt_2h")
    # ADA 诊断
    if (fpg and fpg>=7.0) or (hba1c and hba1c>=6.5) or (ogtt and ogtt>=11.1):
        return {"is_diabetes": True,  "reason": f"指标: FPG={fpg}, HbA1c={hba1c}, OGTT2h={ogtt}"}
    return {"is_diabetes": False, "reason": "未达到糖尿病诊断阈值"}
