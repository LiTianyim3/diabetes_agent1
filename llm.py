import streamlit as st
import json
import requests
from streamlit.runtime.scriptrunner import RerunException, get_script_run_ctx

ZHIPU_API_KEY = "0515ad67847f4d3d847e1e6806696ecf.L8ONlqhUqB38sxlv"
ZHIPU_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
MODEL_NAME = "glm-4-flash"

# 调用智谱API
def call_zhipu_simple(prompt_text):
    url = f"{ZHIPU_API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.7,
        "top_p": 0.9
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        st.error(f"API调用失败: {response.status_code} {response.text}")
        return None

# 强制刷新页面
def rerun():
    raise RerunException(get_script_run_ctx())

def main():
    st.title("饮食计划生成中，请稍候...")

    if 'diet_requirement' not in st.session_state:
        st.error("缺少饮食需求数据，请返回重新填写。")
        if st.button("返回填写页"):
            st.session_state['current_page'] = "page1"
            rerun()
        return

    diet_req = st.session_state['diet_requirement']

    try:
        with open("plan_prompt.txt", "r", encoding="utf-8") as f:
            plan_prompt = f.read()
    except Exception as e:
        st.error(f"读取plan_prompt.txt失败: {e}")
        return

    prompt = (
        "请结合以下两个内容，帮我生成一个合理的饮食计划（JSON格式）：\n\n"
        "【饮食要求】\n"
        f"{json.dumps(diet_req, ensure_ascii=False, indent=2)}\n\n"
        "【提示信息】\n"
        f"{plan_prompt}"
    )

    if st.button("开始生成饮食计划"):
        with st.spinner("正在为您生成饮食计划，请稍候..."):
            output = call_zhipu_simple(prompt)
            if output:
                try:
                    with open("plan_data.json", "w", encoding="utf-8") as f:
                        f.write(output)  # 直接写入原始输出
                    st.session_state['plan_data'] = output
                    st.session_state['plan_ready'] = True
                    st.success("饮食计划生成成功！")
                except Exception as e:
                    st.error(f"写入plan_data.json失败: {e}")

    if st.session_state.get("plan_ready", False):
        if st.button("查看饮食计划"):
            st.session_state['current_page'] = "page2"
            rerun()