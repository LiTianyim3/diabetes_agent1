
import requests
import base64
import json
import re

ZHIPU_API_KEY = "134e1f9197e14a76a5026e281b24146d.itdkjpynPIlGrmXl"
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

def parse_lab_report(file_bytes: bytes) -> dict:
    """
    直接调用智谱GLM多模态API，识别图片医学指标。
    """
    # 图片base64编码
    img_b64 = base64.b64encode(file_bytes).decode("utf-8")
    # 构造prompt
    prompt = (
        "请从这份医学检验报告图片中，尽可能多地提取以下医学指标，并以严格的 JSON 格式返回，所有key必须为中文。"
        "每个指标请同时识别其常见的中文、英文缩写或别名："
        "空腹血糖（GLU、FPG、Fasting Glucose）、半小时血糖、"
        "一小时血糖、两小时血糖（2hPG、2小时血糖）、三小时血糖、"
        "糖化血红蛋白（HbA1c、糖化）、BMI、身高（Height）、体重（Weight）、"
        "收缩压（SBP、Systolic）、舒张压（DBP、Diastolic）、心率（HR、Heart Rate）、体温（T、Temp、Temperature）。"
        "如果某项没有提及请返回 null。示例："
        '{"空腹血糖":6.8,"半小时血糖":null,"一小时血糖":null,"两小时血糖":9.1,"三小时血糖":null,"糖化血红蛋白":6.2,"BMI":24.5,"身高":170,"体重":68,"收缩压":120,"舒张压":80,"心率":75,"体温":36.5}'
    )

    # 智谱GLM多模态API请求体
    payload = {
        "model": "glm-4v",  # 可根据实际模型名调整
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}
        ]
    }
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(ZHIPU_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # 取大模型回复内容
        reply = data["choices"][0]["message"]["content"]
        # 尝试直接解析JSON
        try:
            return json.loads(reply)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", reply, re.S)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"error": "未能解析模型返回的结构化信息。", "raw": reply}
    except Exception as e:
        return {"error": f"多模态API调用失败：{e}"}

    # 纯文本解析已废弃，全部走多模态API

