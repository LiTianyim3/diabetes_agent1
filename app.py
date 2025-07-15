import os
import base64
import logging
import gradio as gr
from client.zhipu_llm import ZhipuLLM
import datetime
from neo4j import GraphDatabase

neo_driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "415zc415")
)

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
  min-width: 50px;
  max-width: 80px;
}
"""

QUESTIONS = [
    "1/4 您是否有多尿、多饮、多食等高血糖症状？（有 / 无）",
    "2/4 您的空腹血糖（FPG）是多少？（mmol/L）",
    "3/4 您的餐后 2 小时血糖是多少？（mmol/L）",
    "4/4 您的 HbA1c（糖化血红蛋白）是多少？（%）",
]

def _query_graph(answers: dict):
    """
    answers = {"sym": "有/无", "fpg": "7.8", "pp2h": "12.0", "hb": "6.8"}
    返回去重后的诊断名称列表（可能为空）
    """
    cypher = """
    WITH $sym AS sym,
         toFloat($fpg)  AS fpg,
         toFloat($pp2h) AS pp2h,
         toFloat($hb)   AS hb
    OPTIONAL MATCH (s:Symptom {name:'典型高血糖症状'})-[:LEADS_TO]->(d1:Diagnosis)
      WHERE sym = '有'
    OPTIONAL MATCH (i1:Indicator {name:'FPG'})-[:LEADS_TO]->(d2:Diagnosis)
      WHERE fpg >= i1.threshold
    OPTIONAL MATCH (i2:Indicator {name:'PostPrandial2H'})-[:LEADS_TO]->(d3:Diagnosis)
      WHERE pp2h >= i2.threshold
    OPTIONAL MATCH (i3:Indicator {name:'HbA1c'})-[:LEADS_TO]->(d4:Diagnosis)
      WHERE hb >= i3.threshold
    RETURN
      collect(d1.name) + collect(d2.name) +
      collect(d3.name) + collect(d4.name) AS all_names
    """
    with neo_driver.session(bookmarks=None) as s:
        rec = s.run(cypher, **answers).single()
        raw = rec["all_names"] if rec else []
        # 过滤 None / 空串，再去重
        return list({n for n in raw if n})

def gen_followup_question(answers: dict, asked: list[str]) -> str:
    """
    answers : 已收集的回答 dict
    asked   : 已经问过的问题文本列表
    """
    prompt = (
        "你是一名内分泌科医生，正在做糖尿病问诊。\n"
        "以下问题已经问过，禁止重复：\n" +
        "\n".join(f"- {q}" for q in asked) + "\n\n"
        "已获回答：\n"
        f"典型症状={answers.get(0,'未回答')}；"
        f"FPG={answers.get(1,'未回答')}；"
        f"餐后2h={answers.get(2,'未回答')}；"
        f"HbA1c={answers.get(3,'未回答')}\n"
        "请提出下一条最关键且不重复的问题，"
        "要求：用中文、简洁，且只输出问题本身。"
    )
    try:
        q = llm._call(prompt).strip()
    except Exception:
        q = ""
    return q or "请提供其他与血糖相关的检查或症状信息？"

def guided_on_send(text, file_list, history_state,
                   name, age, weight, gender, past_history):
    history, step, answers = history_state
    user_text = (text or "").strip()

    # 收集上一步回答
    if step > 0:
        answers[step-1] = user_text
        history.append({"role":"user","content":user_text})

    # 还没问完 ⇒ 继续提问
    if step < len(QUESTIONS):
        q = QUESTIONS[step]
        history.append({"role":"assistant","content":q})
        step += 1
        return history, (history, step, answers), file_list, gr.update(), gr.update(value="")

    diag_list = _query_graph({
        "sym":  answers.get(0, "无"),
        "fpg":  answers.get(1, "0"),
        "pp2h": answers.get(2, "0"),
        "hb":   answers.get(3, "0")
    })
    MAX_EXTRA_QUESTIONS = 5   # 最多继续追问 5 次
    MAX_RETRY = 3          # 生成不重复问题最多重试 3 次
    
    if diag_list:
        reply = "初步可能诊断类型：" + "、".join(diag_list)
        history.append({"role":"assistant","content":reply})
        # 结束：重置 state
        new_state = ([{"role":"assistant","content":"如需再次问诊，请输入任意内容。"}], 0, {})
        return history, new_state, [], gr.update(), gr.update(value="")

    # ── 若仍未诊断，自动生成下一条问题 ──
    if step - 4 >= MAX_EXTRA_QUESTIONS:
        reply = "已追问多次仍无法给出诊断，建议携带完整检查报告就医。"
        history.append({"role":"assistant","content":reply})
        new_state = ([{"role":"assistant","content":"如需再次问诊，请输入任意内容。"}], 0, {})
        return history, new_state, [], gr.update(), gr.update(value="")

    asked_texts = [m["content"] for m in history if m["role"] == "assistant" and m["content"].endswith("？")]
    for _ in range(MAX_RETRY):
        next_q = gen_followup_question(answers, asked_texts)
        if next_q not in asked_texts:
            break
    else:  # 三次都重复，就给兜底问题
        next_q = "请告诉我任何最近血糖相关的异常检查项目？"

    history.append({"role":"assistant","content":next_q})
    step += 1
    return history, (history, step, answers), file_list, gr.update(), gr.update(value="")



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
            # 自动遍历所有非空医学指标并展示（直接用中文key）
            if any(result.values()):
                summary = []
                for k, v in result.items():
                    if v is not None:
                        summary.append(f"{k}: {v}")
                if summary:
                    history.append({"role": "system", "content": "自动识别信息：\n" + "\n".join(summary)})
        elif ext == "pdf":
            md = f"[📄 {name}](data:application/pdf;base64,{b64})"
            history.append({"role":"system","content":f"已上传 PDF：{md}"})
            # 自动解析 PDF 医学指标
            result = parse_lab_report(file_bytes)
            if any(result.values()):
                summary = []
                for k, v in result.items():
                    if v is not None:
                        summary.append(f"{k}: {v}")
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

def on_send(text, file_list, history, name, age, weight, gender, past_history):
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
    # 用户消息直接传递
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

    # 发送后清空已上传列表，不再自动生成病例
    return history, history, [], gr.update(choices=[], value=[]), gr.update(value="")

def on_generate_case(history, name=None, age=None, weight=None, gender=None, past_history=None):
    if not history or len(history) == 0:
        return "**病例记录**\n\n尚无内容"
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    personal_info = (
        f"姓名：{name or '未填写'}；年龄：{age or '未填写'}；体重：{weight or '未填写'}；"
        f"性别：{gender or '未填写'}；既往史：{past_history or '未填写'}"
    )
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

    init_msg = [{"role":"assistant","content":"您好，我是糖尿病智能问诊助手，开始问诊。"}]
    state = gr.State((init_msg, 0, {}))   # (history, step, answers)

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
        fn=guided_on_send,
        inputs=[text_input, file_list, state, name_input, age_input, weight_input, gender_input, history_input],
        outputs=[chatbot, state, file_list, file_selector, text_input]
    )
    text_input.submit(
        fn=guided_on_send,
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
