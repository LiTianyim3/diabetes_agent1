import json
import logging
import gradio as gr
import concurrent.futures
import time
from client.zhipu_llm import ZhipuLLM
from langchain.agents import Tool, AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from tools.diabetes_classifier import classify_diabetes
from tools.exercise_advice import gen_exercise_advice
from tools.lab_report_parser import parse_lab_report_text
from tools.nutrition_advice import gen_nutrition_advice
from tools.severity_scoring import score_severity

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# 初始化智谱 AI LLM 和 Agent
llm = ZhipuLLM()
tools = [
    Tool("parse_lab", func=parse_lab_report_text,   description="解析检验报告文本，提取血糖、HbA1c等指标"),
    Tool("classify_dm", func=classify_diabetes,     description="判断是否糖尿病"),
    Tool("score_sev", func=score_severity,          description="分级：轻/中/重"),
    Tool("nutrition", func=gen_nutrition_advice,    description="生成营养建议"),
    Tool("exercise", func=gen_exercise_advice,      description="生成运动建议"),
]

# 用 ChatPromptTemplate 构造 prompt（简化为与可用项目一致）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业的糖尿病健康助手，请根据工具和用户输入给出专业建议。"),
    ("user", "{input}"),
    ("system", "可用工具包括：{tool_names}"),
    ("system", "工具执行记录：{agent_scratchpad}"),
    ("system", "工具列表：{tools}")
])

# 创建 AgentExecutor（与可用项目一致）
agent = AgentExecutor.from_agent_and_tools(
    agent=create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt),
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=15,           # 增加迭代次数限制
    max_execution_time=60,       # 增加最大执行时间（秒）
)

# 解析报告摘要
def parse_report(report_text: str) -> str:
    logger.info("parse_report called with report_text: %r", report_text)
    if not report_text:
        return ""
    # 提取并日志输出结构化数据
    data = parse_lab_report_text(report_text)
    logger.info("Extracted lab data: %s", data)
    # 拼接摘要 prompt
    lines = []
    if data.get("fasting_glucose") is not None:
        lines.append(f"- 空腹血糖: {data['fasting_glucose']} mmol/L")
    if data.get("hba1c") is not None:
        lines.append(f"- HbA1c: {data['hba1c']} %")
    if data.get("ogtt_2h") is not None:
        lines.append(f"- OGTT 2h 血糖: {data['ogtt_2h']} mmol/L")
    if data.get("bmi") is not None:
        lines.append(f"- BMI: {data['bmi']}")
    bullet_str = "\n".join(lines)
    prompt = (
        f"以下是患者的检查指标：\n{bullet_str}\n"
        "请用一段专业、简洁的自然语言，概括上述检查结果及潜在风险。"
    )
    logger.info("Summary prompt: %r", prompt)
    summary = llm._call(prompt)
    logger.info("Generated summary: %r", summary)
    return summary

# 回答用户疑问（不强制 JSON，直接自然语言）
def answer_question(report_summary, user_message, history):
    logger.info(f"answer_question called with report_summary: '{report_summary}' user_message: '{user_message}'")
    # 快速问候分支
    if user_message.strip() in ["你好", "您好", "hi", "hello"]:
        bot_msg = "您好，我是糖医助手，有什么可以帮您？"
        history = history or []
        history.append([user_message, bot_msg])
        return history, history

    history = history or []
    bot_msg = "正在生成，请稍候..."
    history.append([user_message, bot_msg])

    # 构造多轮对话上下文
    chat_history_str = ""
    for q, a in history[:-1]:  # 不包括本轮
        chat_history_str += f"用户：{q}\n助手：{a}\n"
    chat_history_str += f"用户：{user_message}\n"

    # 针对有无报告摘要分别处理
    if report_summary and report_summary.strip():
        prompt = (
            f"患者报告摘要：\n{report_summary}\n\n"
            f"对话历史：\n{chat_history_str}"
            "请结合上述信息，直接用专业简明的自然语言回答用户问题。"
        )
    else:
        prompt = (
            f"对话历史：\n{chat_history_str}"
            "请用专业简明的自然语言安抚用户，并引导其补充健康检查报告或描述具体症状，无需重复追问。"
        )

    logger.info(f"Question prompt: '{prompt}'")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.invoke, {"input": prompt}, handle_parsing_errors=True)
            agent_result = future.result(timeout=15)
        logger.info(f"Agent response: {agent_result}")
        if isinstance(agent_result, dict) and "output" in agent_result:
            bot_msg = agent_result["output"]
        else:
            bot_msg = str(agent_result)
    except concurrent.futures.TimeoutError:
        bot_msg = "响应超时，请稍后再试或简化问题。"
        logger.warning("Agent response timeout.")
    except Exception as e:
        bot_msg = f"发生错误：{str(e)}"
    history[-1][1] = bot_msg  # 更新最后一条回复
    return history, history

# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 糖医助手 🩸 — 报告摘要与疑问解答")
    with gr.Row():
        with gr.Column(scale=2):
            report_input = gr.Textbox(
                label="检验报告（纯文本）",
                placeholder="将检验报告文字粘贴在此处",
                lines=6
            )
            parse_btn = gr.Button("解析报告要点")
            summary_output = gr.Textbox(
                label="报告摘要",
                interactive=False,
                lines=6
            )
            user_input = gr.Textbox(
                label="请输入您的问题",
                placeholder="如：我最近口渴、多尿，有家族史",
                lines=2
            )
            send_btn = gr.Button("发送")
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话记录")

    state = gr.State([])

    # 解析摘要
    parse_btn.click(
        fn=parse_report,
        inputs=[report_input],
        outputs=[summary_output]
    )
    # 发送疑问
    send_btn.click(
        fn=answer_question,
        inputs=[summary_output, user_input, state],
        outputs=[chatbot, state]
    )
    user_input.submit(
        fn=answer_question,
        inputs=[summary_output, user_input, state],
        outputs=[chatbot, state]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
