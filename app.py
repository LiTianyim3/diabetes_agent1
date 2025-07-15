import json
import logging
import gradio as gr
import re
from client.zhipu_llm import ZhipuLLM
from tools.lab_report_parser import parse_lab_report  # 新增导入

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

llm = ZhipuLLM()
user_info = {}

KEYWORDS = ["血糖", "糖尿病", "高血糖", "低血糖", "HbA1c", "OGTT", "多尿", "多饮", "多食", "家族史"]

def extract_info(user_message: str):
    # 简单正则/关键词提取，可扩展
    info = {}
    if re.search(r"\d+(\.\d+)?\s*mmol", user_message):
        info["blood_glucose"] = True
    if any(k in user_message for k in ["糖尿病", "高血糖", "低血糖"]):
        info["diabetes"] = True
    # ...可扩展更多字段...
    return info

def need_more_info(info: dict):
    # 判断是否需要补充信息
    # 这里只做简单判断，实际可更复杂
    if not info.get("blood_glucose") and not info.get("diabetes"):
        return True
    return False

def summarize_output(text: str) -> str:
    """
    用大模型将输出内容总结为一句简洁明了的摘要。
    """
    if not text:
        return ""
    prompt = (
        "请将以下健康建议内容总结为一句简洁明了的摘要，突出重点，避免冗长：\n"
        f"{text}"
    )
    try:
        return llm._call(prompt)
    except Exception as e:
        return f"摘要失败: {str(e)}"

def summarize_history(history):
    """
    对历史对话进行摘要，只总结机器人回复内容。
    """
    if not history:
        return ""
    # 只取机器人回复
    bot_texts = [msg[1] for msg in history if len(msg) > 1 and msg[1]]
    text = "\n".join(bot_texts)
    return summarize_output(text)

def answer_question_simple(user_message, history):
    global user_info
    logger.info(f"answer_question_simple called with user_message: '{user_message}'")
    history = history or []
    bot_msg = "正在生成，请稍候..."
    history.append([user_message, bot_msg])

    # 关键词检测
    hit = any(k in user_message for k in KEYWORDS)
    info = extract_info(user_message)
    user_info.update(info)

    if hit:
        if need_more_info(user_info):
            prompt = (
                f"用户输入：{user_message}\n"
                "请判断用户是否需要补充健康检查报告或描述具体症状，"
                "如果信息不足，请专业地引导用户补充相关信息；"
                "如果信息充足，则给出专业建议。"
            )
        else:
            prompt = (
                f"用户输入：{user_message}\n"
                "请结合用户已提供的信息，给出专业的健康建议。"
            )
    else:
        prompt = (
            f"用户输入：{user_message}\n"
            "请用专业简明的自然语言安抚用户，并引导其补充健康检查报告或描述具体症状。"
        )

    try:
        bot_msg = llm._call(prompt)
    except Exception as e:
        bot_msg = f"发生错误：{str(e)}"
    history[-1][1] = bot_msg
    # 不再自动生成摘要
    return history, history

def parse_report_file(file):
    if file is None:
        return ""
    try:
        with open(file.name, "rb") as f:
            file_bytes = f.read()
        result = parse_lab_report(file_bytes)
        if not result:
            return "未识别到有效指标，请确认文件内容。"
        # 格式化摘要
        summary = []
        if result.get("fasting_glucose") is not None:
            summary.append(f"空腹血糖: {result['fasting_glucose']} mmol/L")
        if result.get("hba1c") is not None:
            summary.append(f"HbA1c: {result['hba1c']} %")
        if result.get("ogtt_2h") is not None:
            summary.append(f"OGTT 2小时血糖: {result['ogtt_2h']} mmol/L")
        if result.get("bmi") is not None:
            summary.append(f"BMI: {result['bmi']}")
        return "\n".join(summary) if summary else "未识别到有效指标。"
    except Exception as e:
        return f"解析失败: {str(e)}"

# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 糖医助手 🩸")
    with gr.Row():
        # 左侧：聊天记录、输入区
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话记录")
            user_input = gr.Textbox(
                label="请输入您的问题",
                placeholder="如：我最近血糖有点高，怎么办？",
                lines=2
            )
            send_btn = gr.Button("发送")
           
        # 右侧：文件上传和报告摘要
        with gr.Column(scale=2):
            report_file = gr.File(label="检验报告文件上传（PDF/图片）")
            report_summary = gr.Textbox(label="报告摘要", lines=5)  # 报告摘要框移到文件上传框下方
            summarize_btn = gr.Button("总结报告摘要")
    state = gr.State([])

    # 发送按钮和输入框提交时，不再输出摘要
    send_btn.click(
        fn=answer_question_simple,
        inputs=[user_input, state],
        outputs=[chatbot, state]
    )
    user_input.submit(
        fn=answer_question_simple,
        inputs=[user_input, state],
        outputs=[chatbot, state]
    )
    # 总结按钮点击时，输出摘要到报告摘要框
    summarize_btn.click(
        fn=summarize_history,
        inputs=[state],
        outputs=[report_summary]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
