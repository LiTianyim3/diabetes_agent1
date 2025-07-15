import os
import base64
import logging
import gradio as gr
from client.zhipu_llm import ZhipuLLM

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# 初始化智谱 LLM
llm = ZhipuLLM()

def on_file_upload(file_path, history, case_text):
    history = history or []
    if not file_path:
        return history, history, case_text
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    md_img = f"![上传的报告](data:image/{ext};base64,{b64})"
    history.append({
        "role": "system",
        "content": f"已上传文件：{os.path.basename(file_path)}\n\n{md_img}"
    })
    return history, history, case_text

def on_send(text, file_path, history):
    history = history or []
    user_msg = text or ""
    if file_path:
        user_msg = (user_msg + "\n") if user_msg else ""
        user_msg += f"[已上传文件：{os.path.basename(file_path)}]"
    history.append({"role": "user", "content": user_msg})

    prompt = f"用户消息：{user_msg}\n请你基于此给出专业的糖尿病检测/管理建议。"
    logger.info("Prompt to LLM: %s", prompt)
    try:
        reply = llm._call(prompt)
    except Exception as e:
        reply = f"模型调用出错：{e}"
    history.append({"role": "assistant", "content": reply})

    # 生成病例记录
    hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    case_prompt = (
        f"请根据以下对话内容和历史，生成一份结构化的糖尿病患者病例记录：\n\n{hist_text}\n\n"
        "病例记录应包括：基本信息、主诉、现病史、既往史、检查结果、初步诊断、管理建议。"
    )
    logger.info("Case prompt to LLM: %s", case_prompt)
    try:
        case_record = llm._call(case_prompt)
    except Exception as e:
        case_record = f"生成病例出错：{e}"

    return history, history, case_record

with gr.Blocks() as demo:
    gr.Markdown("## 糖尿病助手 🩸 — 左：对话交互；右：病例记录示例")

    # 初始欢迎语
    welcome_msg = [{"role": "assistant", "content": "您好，我是糖尿病专业助手，有什么可以帮您的？"}]

    with gr.Row():
        # 左侧：聊天区
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type="messages",
                label="对话记录",
                height=500,
                value=welcome_msg  # 设置初始消息
            )
            with gr.Row():
                upload_btn = gr.UploadButton(
                    "📎 上传图片",
                    file_types=[".png", ".jpg", ".jpeg"],
                    type="filepath"
                )
                text_input = gr.Textbox(
                    placeholder="请输入问题或备注（可选）",
                    lines=1,
                    show_label=False
                )
                send_btn = gr.Button("发送")
            # 将Examples放在左侧聊天区底部
            gr.Examples(
                examples=[
                    "糖尿病如何控制血糖？",
                    "胰岛素使用注意事项？",
                    "低血糖处理方式",
                    "糖尿病饮食有哪些禁忌？",
                    "运动对血糖影响",
                    "如何监测血糖变化？",
                    "糖尿病并发症有哪些？",
                    "胰岛素泵的适用性",
                    "血糖高有哪些症状？",
                    "我最近血糖有点高，怎么缓解？"
                ],
                inputs=[text_input]
            )

        # 右侧：病例记录区
        with gr.Column(scale=2):
            case_record = gr.Markdown(
                "**病例记录**\n\n尚无内容",
                label="生成的病例记录",
                elem_id="case-record"
            )

    state = gr.State(welcome_msg)  # 初始状态包含欢迎语

    # 绑定上传事件
    upload_btn.upload(
        fn=on_file_upload,
        inputs=[upload_btn, state, case_record],
        outputs=[chatbot, state, case_record]
    )
    # 绑定发送事件
    send_btn.click(
        fn=on_send,
        inputs=[text_input, upload_btn, state],
        outputs=[chatbot, state, case_record]
    )
    text_input.submit(
        fn=on_send,
        inputs=[text_input, upload_btn, state],
        outputs=[chatbot, state, case_record]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
