import pdfplumber, pytesseract
from PIL import Image
import re
from io import BytesIO
from typing import Dict
import json
from client.zhipu_llm import ZhipuLLM

# 使用大模型驱动检验报告文本解析
llm = ZhipuLLM()

def parse_lab_report(file_bytes: bytes) -> dict:
    text = ""
    # PDF 先读文本页
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except:
        # 如果不是 PDF，则当图片 OCR
        img = Image.open(BytesIO(file_bytes))
        text = pytesseract.image_to_string(img, lang='chi_sim+eng')
    # 正则提取常用指标
    def find_first(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "fasting_glucose": find_first(r"空腹.*?([\d\.]+)\s*mmol"),
        "hba1c":            find_first(r"HbA1c[:： ]*([\d\.]+)\s*%"),
        "ogtt_2h":          find_first(r"OGTT.*?2\s*h.*?([\d\.]+)\s*mmol"),
        "bmi":              find_first(r"BMI[:： ]*([\d\.]+)"),
    }

def parse_lab_report_text(text: str) -> dict:
    """
    使用大模型从检验报告纯文本中提取关键指标：
    空腹血糖 (fasting_glucose)、HbA1c、OGTT 2小时血糖 (ogtt_2h)、BMI。
    返回严格的 JSON 格式字典。
    """
    if not text:
        return {}
    # 构造 prompt，要求返回纯 JSON
    prompt = (
        "请从以下检验报告中提取空腹血糖、HbA1c、OGTT 2小时血糖、BMI，"
        "并以严格的 JSON 格式返回，例如 {\"fasting_glucose\":6.8,\"hba1c\":6.2,\"ogtt_2h\":9.1,\"bmi\":24.5}。"
        f"\n报告文本：{text}"
    )
    response = llm._call(prompt)
    # 尝试直接解析 JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # 尝试从响应中抽取 JSON 子串
        match = re.search(r"\{.*\}", response, re.S)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    # 如果解析失败，返回空字典
    return {}

