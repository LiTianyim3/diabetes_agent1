import os
import base64
import logging
import gradio as gr
from client.zhipu_llm import ZhipuLLM
import datetime


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
                for k, v in result.items():
                    if v is not None:
                        summary.append(f"{k}: {v}")
                    if k in INDICATORS and v not in ("", None):
                        INDICATORS[k]["value"] = str(v)
                print(INDICATORS)
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

    if file_list:
        names = ", ".join(os.path.basename(p) for p in file_list)
        suffix = f"[已上传文件：{names}]"
        user_msg = f"{user_msg}\n{suffix}" if user_msg else suffix
    history.append({"role":"user","content":user_msg})

    
    for key, info in INDICATORS.items():
        prompt_text = info["prompt"]
        value = info["value"]
        # 打印调试信息可选
        print(f"指标名：{key}，提示语：{prompt_text}，已填值：{value}")
        if value is None:
            # 把这个指标的 prompt 发给用户
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
        prompt = (
            f"用户个人信息：{personal_info}\n"
            f"用户上传了医学报告或图片，系统自动识别出如下结构化信息：\n{auto_info}\n,并恢复在auto_info中了解了什么"
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
