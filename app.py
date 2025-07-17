import os
import base64
import logging
import gradio as gr
from client.zhipu_llm import ZhipuLLM
import datetime
from tools.case_json_manager import save_case_json
import json


KEY_MAP = {
    "空腹血糖":       "空腹血糖(GLUO)",
    "半小时血糖":     "半小时血糖(GLU0.5)",   # 如果解析器输出“半小时血糖”想填到“二小时血糖(GLU2)”里，请按需调整
    "一小时血糖":     "一小时血糖(GLU1)",
    "两小时血糖":     "两小时血糖(GLU2)",
    "三小时血糖":     "三小时血糖(GLU3)",
    "糖化血红蛋白":   "糖化血红蛋白(HbA1C)",
    "空腹血糖(GLUO)": "空腹血糖",
    "半小时血糖(GLUO0.5)":     "半小时血糖",   # 如果解析器输出“半小时血糖”想填到“二小时血糖(GLU2)”里，请按需调整
    "一小时血糖(GLU1)":     "一小时血糖",
    "两小时血糖(GLU2)":     "两小时血糖",
    "三小时血糖(GLU3)":     "三小时血糖",
    "糖化血红蛋白(HbA1C)":   "糖化血红蛋白",
    "BMI":           "BMI",
    "身高":           None,  # 不需要提示时可设为 None
    "体重":           None,
    "收缩压":         "血压收缩压",
    "舒张压":         "血压舒张压",
    "心率":           "静息心率",
    "体温":           "体温",
    # ……如有其它字段，按实际补充
}

INDICATORS = {
    "症状": {
        "prompt":
            "为了更好地了解您的情况，"
            "请您回想最近是否出现以下症状——"
            "如口渴、排尿增多或食量明显增加？",
        "value": None
    },
    "空腹血糖": {
        "prompt":
            "清晨空腹时（未进食至少8小时），"
            "您的血糖大约是多少？"
            "请直接输入数值（mmol/L）。",
        "value": None
    },
    "两小时血糖": {
        "prompt": "在进餐后两小时内测得的血糖值是多少？",
        "value": None
    },
    "糖化血红蛋白": {
        "prompt": "最近一次糖化血红蛋白（HbA1c）检测结果是多少？",
        "value": None
    },
    "BMI": {
        "prompt":
            "BMI（体质指数）用于评估体重是否在健康范围，"
            "与糖尿病风险密切相关。"
            "请您告诉我您的 BMI 值（kg/m²，仅数字），",
        "value": None
    },
    "血压收缩压": {
        "prompt": "请提供收缩压 SBP（mmHg，仅数字）",
        "value": None
    },
    "血压舒张压": {
        "prompt": "请提供舒张压 DBP（mmHg，仅数字）",
        "value": None
    },
    "静息心率": {
        "prompt": "请提供静息心率 HR（次/分，仅数字）",
        "value": None
    },
    "亲属糖尿病病史": {
        "prompt": "您是否有一级亲属糖尿病病史或者家族史？",
        "value": None
    },
    "其他重要说明": {
        "prompt":
            "如果您有正在使用的药物、特殊饮食或运动习惯等，"
            "这会帮助我们更全面地了解您的健康状况。"
            "请您补充其他重要说明：",
        "value": None
    }
}

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
#clear-btn {
  min-width: 30px;
  max-width: 60px;
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
            print(result)
            # 自动遍历所有非空医学指标并展示（直接用中文key）
            if any(result.values()):
                summary = []
                user_features = {}
                # key映射：中文key->数据集英文key
                key_map = {
                    "性别": "gender",
                    "年龄": "age",
                    "高血压": "hypertension",
                    "心脏病": "heart_disease",
                    "吸烟史": "smoking_history",
                    "BMI": "bmi",
                    "糖化血红蛋白": "HbA1c_level",
                    "空腹血糖": "blood_glucose_level",
                    "两小时血糖": "blood_glucose_level",
                    "糖尿病": "diabetes",
                    # 可根据实际数据集继续补充
                }
                import re
                for k, v in result.items():
                    if v is not None:
                        summary.append(f"{k}: {v}")
                        mapped_key = key_map.get(k, k)
                        num = None
                        try:
                            num = float(v)
                        except Exception:
                            match = re.search(r"[-+]?[0-9]*\\.?[0-9]+", str(v))
                            if match:
                                try:
                                    num = float(match.group())
                                except Exception:
                                    pass
                        if num is not None:
                            user_features[mapped_key] = num
                        # 自动填充到INDICATORS value字段
                        if k in INDICATORS and v not in (None, ""):
                            INDICATORS[k]["value"] = str(v)
                if summary:
                    history.append({"role": "system", "content": "自动识别信息：\n" + "\\n".join(summary)})
                # 自动触发RAG检索
                if user_features:
                    from rag.index_diabetes import generate_scientific_advice
                    rag_info = generate_scientific_advice(user_features)
                    history.append({"role": "system", "content": "【数据集相似病例参考】\n" + rag_info})
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

    if file_list:
        names = ", ".join(os.path.basename(p) for p in file_list)
        suffix = f"[已上传文件：{names}]"
        user_msg = f"{user_msg}\n{suffix}" if user_msg else suffix
    history.append({"role":"user","content":user_msg})

    
    # 优先判断是否有自动识别的数值型指标
    auto_features = {}
    for key, info in INDICATORS.items():
        value = info["value"]
        try:
            num = float(value)
            auto_features[key] = num
        except Exception:
            pass
    if auto_features:
        # 已有自动识别数值，直接进入LLM建议和RAG
        pass
    else:
        # 没有自动识别数值，逐项问询
        for key, info in INDICATORS.items():
            prompt_text = info["prompt"]
            value = info["value"]
            print(f"指标名：{key}，提示语：{prompt_text}，已填值：{value}")
            if value is None:
                history.append({
                    "role": "assistant",
                    "content": prompt_text
                })
                INDICATORS[key]["value"] = user_msg
                return (
                    history,                                     
                    history,                                     
                    file_list,                                   
                    gr.update(choices=[os.path.basename(p) for p in file_list], value=[]), 
                    gr.update(value="")                         
                )

    # 仅拼接已上传文件信息
    if file_list:
        names = ", ".join(os.path.basename(p) for p in file_list)
        user_msg = (user_msg + "\n" if user_msg else "") + f"[已上传文件：{names}]"
    # 用户消息直接传递（前端不显示个人信息）
    history.append({"role":"user","content":user_msg})

    # 拼接个人信息（后端传给模型，不显示在聊天区）
    personal_info = (
        f"姓名：{name or '未填写'}；年龄：{age or '未填写'}；体重：{weight or '未填写'}；"
        f"性别：{gender or '未填写'}；既往史：{past_history or '未填写'}"
    )

    # 检查是否有图片/报告自动识别信息
    auto_info = None
    for m in reversed(history):
        if m["role"] == "system" and m["content"].startswith("自动识别信息："):
            auto_info = m["content"]
            break

    # LLM 建议
    if auto_info:
        # 从history中查找rag检索结果
        rag_info = ''
        for m in reversed(history):
            if m["role"] == "system" and m["content"].startswith("【数据集相似病例参考】"):
                rag_info = m["content"]
                break
        if rag_info:
            rag_info = f"{rag_info}\n"
        prompt = (
            f"用户个人信息：{personal_info}\n"
            f"用户上传了医学报告或图片，系统自动识别出如下结构化信息：\n{auto_info}\n"
            f"{rag_info}"
            f"请基于这些医学信息，结合用户消息“{user_msg}”，并恢复在user_msg中了解了什么，不用解释了解的信息。"
            "如果信息不全可适当说明，但不要说无法识别图片。"
        )
    else:
        prompt = (
            f"用户个人信息：{personal_info}\n"
            f"用户消息：{user_msg}\n并恢复在user_msg中了解了什么，不用解释了解的信息。"
        )
    logger.info("Prompt to LLM: %s", prompt)
    try: reply = llm._call(prompt)
    except Exception as e: reply = f"模型调用出错：{e}"
    history.append({"role":"assistant","content":reply})

    # 发送后清空已上传列表，不再自动生成病例
    return history, history, [], gr.update(choices=[], value=[]), gr.update(value="")

def on_generate_case(history, name=None, age=None, weight=None, gender=None, past_history=None):
    info_filled = any([name, age, weight, gender, past_history])
    dialog_filled = history and any(
        m["role"] == "user" and m["content"].strip() for m in history if m["role"] == "user"
    )

    # 情况1：个人信息和对话都没有
    if not info_filled and not dialog_filled:
        return "没有信息可以生成病例报告单，请先填写个人信息或进行对话。"

    # 情况2：只有个人信息
    if info_filled and not dialog_filled:
        case_dict = {
            "姓名": name or "未填写",
            "年龄": age or "未填写",
            "体重": weight or "未填写",
            "性别": gender or "未填写",
            "既往史": past_history or "未填写"
        }
        save_case_json(case_dict, name, DATA_DIR)
        personal_info = (
            f"姓名：{case_dict['姓名']}\n"
            f"年龄：{case_dict['年龄']}\n"
            f"体重：{case_dict['体重']}\n"
            f"性别：{case_dict['性别']}\n"
            f"既往史：{case_dict['既往史']}"
        )
        return f"**病例报告单**\n\n{personal_info}"

    # 情况3：只有对话内容
    if not info_filled and dialog_filled:
        personal_info = {
            "姓名": "无", "年龄": "无", "体重": "无", "性别": "无", "既往史": "无"
        }
        hist = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        case_p = (
            f"请根据以下对话生成结构化糖尿病病例：\n\n"
            f"姓名：无\n年龄：无\n体重：无\n性别：无\n既往史：无\n\n{hist}\n\n"
            "病例应包括：用户个人信息(姓名，年龄，体重，性别)、主诉、现病史、既往史、检查结果、初步诊断、管理建议。控制字数在500字之内"
        )
        logger.info("Case prompt to LLM: %s", case_p)
        try:
            case = llm._call(case_p)
        except Exception as e:
            case = f"生成病例出错：{e}"
            return case
        # 尝试解析为 dict 并保存
        try:
            case_dict = json.loads(case)
        except Exception:
            case_dict = {"内容": case}
        save_case_json(case_dict, None, DATA_DIR)
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
    # 尝试解析为 dict 并保存
    try:
        case_dict = json.loads(case)
    except Exception:
        case_dict = {"内容": case}
    save_case_json(case_dict, name, DATA_DIR)
    return case

def on_clear_history():
    welcome_msg = [{"role": "assistant", "content": "您好，我是糖尿病专业助手，请您提供详细病例信息，以便我为您量身定制医学建议。你有关于最近的报告可以给我看看吗"}]
    return welcome_msg, welcome_msg, "**病例记录**\n\n尚无内容"

from ui.custom_ui import build_ui

demo = build_ui(
    on_file_upload=on_file_upload,
    on_delete=on_delete,
    on_send=on_send,
    on_clear_history=on_clear_history,
    on_generate_case=on_generate_case
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
