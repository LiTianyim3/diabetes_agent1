import os
import base64
import logging
import gradio as gr
from client.zhipu_llm import ZhipuLLM
import datetime

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
#clear-btn {
  min-width: 30px;
  max-width: 60px;
}
"""

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

def on_send(text, file_list, history, name, age, weight, gender, past_history):
    history   = history or []
    user_msg  = text or ""
    # 仅拼接已上传文件信息
    if file_list:
        names = ", ".join(os.path.basename(p) for p in file_list)
        user_msg = (user_msg + "\n" if user_msg else "") + f"[已上传文件：{names}]"
    # 用户消息直接传递
    history.append({"role":"user","content":user_msg})

    # LLM 建议
    prompt = f"用户消息：{user_msg}\n请基于此给出专业的糖尿病检测/管理建议。"
    logger.info("Prompt to LLM: %s", prompt)
    try: reply = llm._call(prompt)
    except Exception as e: reply = f"模型调用出错：{e}"
    history.append({"role":"assistant","content":reply})

    # 发送后清空已上传列表，不再自动生成病例
    return history, history, [], gr.update(choices=[], value=[]), gr.update(value="")

def on_generate_case(history, name=None, age=None, weight=None, gender=None, past_history=None):
    # 判断个人信息是否填写
    info_filled = any([name, age, weight, gender, past_history])
    # 判断是否有对话内容（排除初始欢迎语）
    dialog_filled = history and any(
        m["role"] == "user" and m["content"].strip() for m in history if m["role"] == "user"
    )

    # 情况1：个人信息和对话都没有
    if not info_filled and not dialog_filled:
        return "没有信息可以生成病例报告单，请先填写个人信息或进行对话。"

    # 情况2：只有个人信息
    if info_filled and not dialog_filled:
        personal_info = (
            f"姓名：{name or '未填写'}\n"
            f"年龄：{age or '未填写'}\n"
            f"体重：{weight or '未填写'}\n"
            f"性别：{gender or '未填写'}\n"
            f"既往史：{past_history or '未填写'}"
        )
        return f"**病例报告单**\n\n{personal_info}"

    # 情况3：只有对话内容
    if not info_filled and dialog_filled:
        personal_info = (
            f"姓名：无\n年龄：无\n体重：无\n性别：无\n既往史：无"
        )
        hist = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        case_p = (
            f"请根据以下对话生成结构化糖尿病病例：\n\n"
            f"{personal_info}\n\n{hist}\n\n"
            "病例应包括：用户个人信息(姓名，年龄，体重，性别)、主诉、现病史、既往史、检查结果、初步诊断、管理建议。控制字数在500字之内"
        )
        logger.info("Case prompt to LLM: %s", case_p)
        try:
            case = llm._call(case_p)
        except Exception as e:
            case = f"生成病例出错：{e}"
            return case
        return case

    # 情况4：个人信息和对话都有
    personal_info = (
        f"姓名：{name or '未填写'}；年龄：{age or '未填写'}；体重：{weight or '未填写'}；"
        f"性别：{gender or '未填写'}；既往史：{past_history or '未填写'}"
    )
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    case_p = (
        f"请根据以下对话和个人信息生成结构化糖尿病病例：\n\n"
        f"{personal_info}\n\n{hist}\n\n"
        "病例应包括：用户个人信息(姓名，年龄，体重，性别)、主诉、现病史、既往史、检查结果、初步诊断、管理建议。控制字数在500字之内"
    )
    logger.info("Case prompt to LLM: %s", case_p)
    try:
        case = llm._call(case_p)
    except Exception as e:
        case = f"生成病例出错：{e}"
        return case
    return case

def on_clear_history():
    welcome_msg = [{"role": "assistant", "content": "您好，我是糖尿病专业助手，请您提供详细病例信息，以便我为您量身定制医学建议。"}]
    return welcome_msg, welcome_msg, "**病例记录**\n\n尚无内容"

with gr.Blocks(css=css) as demo:
    gr.Markdown("## 糖尿病助手 🩸 — 左：对话交互；右：病例记录")

    # 新增：个人信息输入框
    with gr.Row():
        name_input = gr.Textbox(label="姓名", placeholder="请输入姓名", lines=1)
        age_input = gr.Textbox(label="年龄", placeholder="请输入年龄", lines=1)
        weight_input = gr.Textbox(label="体重（kg）", placeholder="请输入体重", lines=1)
        gender_input = gr.Dropdown(label="性别", choices=["男", "女"], value=None)
        history_input = gr.Textbox(label="既往史", placeholder="请输入既往史", lines=1)

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
                        "我最近血糖有点高，怎么缓解？",
                        "糖尿病饮食有哪些禁忌？",
                        "运动对血糖影响",
                        "如何监测血糖变化？",
                        "糖尿病并发症有哪些？",
                        "胰岛素泵的适用性",
                        "血糖高有哪些症状？",
                    ],
                    inputs=[text_input]
                )
                clear_btn = gr.Button("清除对话历史", elem_id="clear-btn", scale=1)

        # 右侧病例记录
        with gr.Column(scale=2):
            case_md = gr.Markdown("**病例记录**\n\n尚无内容")
            gen_case_btn = gr.Button("生成病例报告单", elem_id="gen-case-btn")

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
    # 发送 -> 生成回复，并清空文件列表和输入框
    send_btn.click(
        fn=on_send,
        inputs=[text_input, file_list, state, name_input, age_input, weight_input, gender_input, history_input],
        outputs=[chatbot, state, file_list, file_selector, text_input]
    )
    text_input.submit(
        fn=on_send,
        inputs=[text_input, file_list, state, name_input, age_input, weight_input, gender_input, history_input],
        outputs=[chatbot, state, file_list, file_selector, text_input]
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
        inputs=[state, name_input, age_input, weight_input, gender_input, history_input],
        outputs=[case_md]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
