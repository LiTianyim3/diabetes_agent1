import os
import base64
import logging
import gradio as gr
from client.zhipu_llm import ZhipuLLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)
llm = ZhipuLLM()

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
"""

def on_file_upload(file_paths, history, file_list):
    history   = history   or []
    file_list = file_list or []

    # 如果没选新文件，仅刷新列表
    if not file_paths:
        opts = [os.path.basename(p) for p in file_list]
        return history, history, file_list, gr.update(choices=opts, value=[])

    paths = file_paths if isinstance(file_paths, list) else [file_paths]

    from tools.lab_report_parser import parse_lab_report

    for p in paths:
        if p in file_list:
            continue
        file_list.append(p)
        name = os.path.basename(p)
        ext  = os.path.splitext(name)[1].lower().lstrip(".")

        with open(p, "rb") as f:
            file_bytes = f.read()
            b64 = base64.b64encode(file_bytes).decode("utf-8")

        # 聊天区插入图片/文件
        if ext in ("png","jpg","jpeg"):
            md = f"![{name}](data:image/{ext};base64,{b64})"
            history.append({"role":"system", "content":f"已上传图片：{name}\n\n{md}"})
            # 自动解析图片医学指标
            result = parse_lab_report(file_bytes)
            print('result:' + str(result))
            # 自动遍历所有非空医学指标并展示
            if any(result.values()):
                field_map = {
                    "name": "姓名",
                    "age": "年龄",
                    "gender": "性别",
                    "sex": "性别",
                    "fasting_glucose": "空腹血糖(mmol/L)",
                    "hba1c": "HbA1c(%)",
                    "ogtt_2h": "OGTT 2小时血糖(mmol/L)",
                    "bmi": "BMI",
                    "height": "身高(cm)",
                    "weight": "体重(kg)",
                    "systolic_bp": "收缩压(mmHg)",
                    "diastolic_bp": "舒张压(mmHg)",
                    "heart_rate": "心率(次/分)",
                    "temperature": "体温(℃)"
                }
                summary = []
                for k, v in result.items():
                    if v is not None:
                        label = field_map.get(k, k)
                        summary.append(f"{label}: {v}")
                if summary:
                    history.append({"role": "system", "content": "自动识别信息：\n" + "\n".join(summary)})
        elif ext == "pdf":
            md = f"[📄 {name}](data:application/pdf;base64,{b64})"
            history.append({"role":"system","content":f"已上传 PDF：{md}"})
            # 自动解析 PDF 医学指标
            result = parse_lab_report(file_bytes)
            if any(result.values()):
                field_map = {
                    "name": "姓名",
                    "age": "年龄",
                    "gender": "性别",
                    "sex": "性别",
                    "fasting_glucose": "空腹血糖(mmol/L)",
                    "hba1c": "HbA1c(%)",
                    "ogtt_2h": "OGTT 2小时血糖(mmol/L)",
                    "bmi": "BMI",
                    "height": "身高(cm)",
                    "weight": "体重(kg)",
                    "systolic_bp": "收缩压(mmHg)",
                    "diastolic_bp": "舒张压(mmHg)",
                    "heart_rate": "心率(次/分)",
                    "temperature": "体温(℃)"
                }
                summary = []
                for k, v in result.items():
                    if v is not None:
                        label = field_map.get(k, k)
                        summary.append(f"{label}: {v}")
                if summary:
                    history.append({"role": "system", "content": "自动识别信息：\n" + "\n".join(summary)})
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
    # 检查是否有图片/报告自动识别信息
    auto_info = None
    for m in reversed(history):
        if m["role"] == "system" and m["content"].startswith("自动识别信息："):
            auto_info = m["content"]
            break
    if file_list:
        names = ", ".join(os.path.basename(p) for p in file_list)
        user_msg = (user_msg + "\n" if user_msg else "") + f"[已上传文件：{names}]"
    history.append({"role":"user","content":user_msg})

    # LLM 建议
    if auto_info:
        # 有自动识别信息，优先让LLM基于图片/报告结构化内容给建议
        prompt = (
            f"用户上传了医学报告或图片，系统自动识别出如下结构化信息：\n{auto_info}\n"
            f"请基于这些医学信息，结合用户消息“{user_msg}”，给出专业的糖尿病检测/管理建议。"
            "如果信息不全可适当说明，但不要说无法识别图片。"
        )
    else:
        prompt = f"用户消息：{user_msg}\n请基于此给出专业的糖尿病检测/管理建议。"
    logger.info("Prompt to LLM: %s", prompt)
    try: reply = llm._call(prompt)
    except Exception as e: reply = f"模型调用出错：{e}"
    history.append({"role":"assistant","content":reply})

    # 生成病例记录
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    case_p = (
        f"请根据以下对话生成结构化糖尿病病例：\n\n{hist}\n\n"
        "病例应包括：基本信息、主诉、现病史、既往史、检查结果、初步诊断、管理建议。"
    )
    logger.info("Case prompt to LLM: %s", case_p)
    try: case = llm._call(case_p)
    except Exception as e: case = f"生成病例出错：{e}"

    # **发送后清空已上传列表**
    return history, history, case, [], gr.update(choices=[], value=[])

with gr.Blocks(css=css) as demo:
    gr.Markdown("## 糖尿病助手 🩸 — 左：对话；已上传文件列表带×可点删；每次发送后清空")

    # 初始引导消息
    initial_message = {
        "role": "assistant",
        "content": (
            "您好，我是您的智能糖尿病健康管理助手，可以为您提供糖尿病相关的检测解读、健康建议和个性化管理方案。\n"
            "请问您的姓名、年龄、性别、糖尿病类型、诊断时间等基本信息，以及目前的主要健康关注点是什么？"
        )
    }

    with gr.Row():
        # 左侧对话区域
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=400, value=[initial_message])
            with gr.Row():
                upload_btn = gr.UploadButton(
                    "📎 上传文件",
                    file_types=[".png",".jpg",".jpeg",".pdf"],
                    file_count="multiple",
                    type="filepath"
                )
                text_input = gr.Textbox(placeholder="输入...", lines=1, show_label=False)
                send_btn   = gr.Button("发送")
            # 文件列表：勾选即删
            file_selector = gr.CheckboxGroup(
                choices=[], label="已上传文件列表", elem_id="file-selector"
            )

        # 右侧病例记录
        with gr.Column(scale=2):
            case_md = gr.Markdown("**病例记录**\n\n尚无内容")

    state     = gr.State([])  # 聊天历史
    file_list = gr.State([])  # 文件路径列表

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
    # 发送 -> 生成回复&病例，并清空文件列表
    send_btn.click(
        fn=on_send,
        inputs=[text_input, file_list, state],
        outputs=[chatbot, state, case_md, file_list, file_selector]
    )
    text_input.submit(
        fn=on_send,
        inputs=[text_input, file_list, state],
        outputs=[chatbot, state, case_md, file_list, file_selector]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
