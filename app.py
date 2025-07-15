import os
import base64
import logging
import gradio as gr
from client.zhipu_llm import ZhipuLLM
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)
llm = ZhipuLLM()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# CSS：左侧 × 图标 & 悬停高亮
css = """
#file-selector .gr-checkbox {
  padding: 8px 8px 8px 28px;
  position: relative;
  border-radius: 4px;
  transition: background-color 0.2s;
}
/* × 图标放在左侧 */
#file-selector .gr-checkbox:hover::before {
  content: "×";
  position: absolute;
  left: 8px;
  top: 50%;
  transform: translateY(-50%);
  color: #e00;
  font-size: 16px;
  cursor: pointer;
}
/* 悬停高亮 */
#file-selector .gr-checkbox:hover {
  background-color: #f5f5f5;
}
/* 限制清除按钮宽度 */
#clear-btn {s
  min-width: 100px;
  max-width: 120px;
}
"""

def save_case_pdf(case_text: str) -> str:
    # 按时间命名
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(DATA_DIR, f"case_{ts}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # 注册中文字体（优先使用 SimSun，如果没有则用 STSong）
    try:
        font_path = os.path.join(os.path.dirname(__file__), "simsun.ttf")
        pdfmetrics.registerFont(TTFont("SimSun", font_path))
        font_name = "SimSun"
    except Exception:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        font_name = "STSong-Light"

    c.setFont(font_name, 12)
    text_obj = c.beginText(40, height - 50)
    for line in case_text.splitlines():
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.save()
    return pdf_path

def on_file_upload(file_paths, history, file_list):
    history   = history   or []
    file_list = file_list or []

    # 如果没选新文件，仅刷新列表
    if not file_paths:
        opts = [os.path.basename(p) for p in file_list]
        return history, history, file_list, gr.update(choices=opts, value=[])

    paths = file_paths if isinstance(file_paths, list) else [file_paths]
    for p in paths:
        if p in file_list:
            continue
        file_list.append(p)
        name = os.path.basename(p)
        ext  = os.path.splitext(name)[1].lower().lstrip(".")

        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        if ext in ("png","jpg","jpeg"):
            md = f"![{name}](data:image/{ext};base64,{b64})"
            history.append({"role":"system", "content":f"已上传图片：{name}\n\n{md}"})
        elif ext == "pdf":
            # PDF 以链接形式
            md = f"[📄 {name}](data:application/pdf;base64,{b64})"
            history.append({"role":"system","content":f"已上传 PDF：{md}"})
        else:
            history.append({"role":"system","content":f"已上传文件：{name}"})

    opts = [os.path.basename(p) for p in file_list]
    return history, history, file_list, gr.update(choices=opts, value=[])

def on_delete(selected, file_list):
    # 点击勾选即删除
    file_list = file_list or []
    remaining = [p for p in file_list if os.path.basename(p) not in (selected or [])]
    opts = [os.path.basename(p) for p in remaining]
    return remaining, gr.update(choices=opts, value=[])

def on_send(text, file_list, history):
    history   = history or []
    user_msg  = text or ""
    if file_list:
        names = ", ".join(os.path.basename(p) for p in file_list)
        user_msg = (user_msg + "\n" if user_msg else "") + f"[已上传文件：{names}]"
    history.append({"role":"user","content":user_msg})

    # LLM 建议
    prompt = f"用户消息：{user_msg}\n请基于此给出专业的糖尿病检测/管理建议。"
    logger.info("Prompt to LLM: %s", prompt)
    try: reply = llm._call(prompt)
    except Exception as e: reply = f"模型调用出错：{e}"
    history.append({"role":"assistant","content":reply})

    # 发送后清空已上传列表，不再自动生成病例
    return history, history, [], gr.update(choices=[], value=[])

def on_generate_case(history):
    if not history or len(history) == 0:
        return "**病例记录**\n\n尚无内容", None, gr.update(visible=True, value=None)
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    case_p = (
        f"请根据以下对话生成结构化糖尿病病例：\n\n{hist}\n\n"
        "病例应包括：用户个人信息、主诉、现病史、既往史、检查结果、初步诊断、管理建议。控制字数在500字之内"
    )
    logger.info("Case prompt to LLM: %s", case_p)
    try:
        case = llm._call(case_p)
    except Exception as e:
        case = f"生成病例出错：{e}"
        return case, None, gr.update(visible=True, value=None)
    pdf_path = save_case_pdf(case)
    # 返回病例文本、PDF路径，显示下载框
    return case, pdf_path, gr.update(visible=True, value=pdf_path)

def on_clear_history():
    welcome_msg = [{"role": "assistant", "content": "您好，我是糖尿病专业助手，请您提供详细病例信息，以便我为您量身定制医学建议。"}]
    return welcome_msg, welcome_msg, "**病例记录**\n\n尚无内容"

with gr.Blocks(css=css) as demo:
    gr.Markdown("## 糖尿病助手 🩸 — 左：对话交互；右：病例记录示例")

    with gr.Row():
        # 左侧对话区域
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", label="对话记录", height=500)
            with gr.Row():
                upload_btn = gr.UploadButton(
                    "📎 上传文件",
                    file_types=[".png",".jpg",".jpeg",".pdf"],
                    file_count="multiple",
                    type="filepath",
                    elem_id="upload-btn",
                    scale=1
                )
                text_input = gr.Textbox(
                    placeholder="请输入问题或备注（可选）",
                    lines=1,
                    show_label=False,
                    elem_id="text-input",
                    scale=2
                )
                send_btn = gr.Button("发送", elem_id="send-btn", scale=1)
            file_list = gr.State([])  # 用于存储文件路径
            file_selector = gr.CheckboxGroup(
                choices=[],
                label="已上传文件（点击 × 删除）",
                elem_id="file-selector"
            )
            with gr.Row():
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
                clear_btn = gr.Button("清除对话历史", elem_id="clear-btn", scale=1)

        # 右侧病例记录
        with gr.Column(scale=2):
            case_md = gr.Markdown("**病例记录**\n\n尚无内容")
            gen_case_btn = gr.Button("生成病例报告单", elem_id="gen-case-btn")
            # PDF下载框，始终可见，点击下载
            pdf_file = gr.File(label="点击下载病例报告PDF", visible=True)
            download_tip = gr.Textbox(value="", show_label=False, interactive=False, visible=True)

    state = gr.State([{"role": "assistant", "content": "您好，我是糖尿病专业助手，请您提供详细病例信息，以便我为您量身定制医学建议。"}])

    # 上传 -> 更新聊天 & 文件列表
    upload_btn.upload(
        fn=on_file_upload,
        inputs=[upload_btn, state, file_list],
        outputs=[chatbot, state, file_list, file_selector]
    )
    # 勾选即删除
    file_selector.change(
        fn=on_delete,
        inputs=[file_selector, file_list],
        outputs=[file_list, file_selector]
    )
    # 发送 -> 生成回复，并清空文件列表
    send_btn.click(
        fn=on_send,
        inputs=[text_input, file_list, state],
        outputs=[chatbot, state, file_list, file_selector]
    )
    text_input.submit(
        fn=on_send,
        inputs=[text_input, file_list, state],
        outputs=[chatbot, state, file_list, file_selector]
    )
    # 清除对话历史按钮
    clear_btn.click(
        fn=on_clear_history,
        inputs=None,
        outputs=[chatbot, state, case_md]
    )
    # 生成病例报告单按钮
    gen_case_btn.click(
        fn=on_generate_case,
        inputs=[state],
        outputs=[case_md, pdf_file, pdf_file]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
