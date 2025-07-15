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
    except Exception as e_pdf:
        # 如果不是 PDF，则当图片 OCR
        try:
            img = Image.open(BytesIO(file_bytes))
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            print("[OCR图片内容]:\n" + text)
            
        except Exception as e_img:
            # 友好提示，返回结构化错误信息
            return {"error": f"图片解析失败：{e_img}。请确保上传的是清晰的图片或标准PDF。"}
    # 直接交给大模型分析文本内容，提取医学指标
    return parse_lab_report_text(text)

def parse_lab_report_text(text: str) -> dict:
    """
    使用大模型从检验报告纯文本中提取常见医学指标，要求返回严格的中文key的JSON：
    空腹血糖、半小时血糖、一小时血糖、两小时血糖、三小时血糖、糖化血红蛋白、BMI、身高、体重、收缩压、舒张压、心率、体温。
    """
    if not text:
        return {}
    prompt = (
        "请从以下检验报告文本中，尽可能多地提取以下医学指标，并以严格的 JSON 格式返回，所有key必须为中文。"
        "每个指标请同时识别其常见的中文、英文缩写或别名："
        "空腹血糖（GLU、FPG、Fasting Glucose）、半小时血糖、"
        "一小时血糖、两小时血糖（2hPG、2小时血糖）、三小时血糖、"
        "糖化血红蛋白（HbA1c、糖化）、BMI、身高（Height）、体重（Weight）、"
        "收缩压（SBP、Systolic）、舒张压（DBP、Diastolic）、心率（HR、Heart Rate）、体温（T、Temp、Temperature）。"
        "如果某项没有提及请返回 null。示例："
        '{"空腹血糖":6.8,"半小时血糖":null,"一小时血糖":null,"两小时血糖":9.1,"三小时血糖":null,"糖化血红蛋白":6.2,"BMI":24.5,"身高":170,"体重":68,"收缩压":120,"舒张压":80,"心率":75,"体温":36.5}'
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

