import streamlit as st
import json
import re

def load_case_info():
    with open("data\李军.json", "r", encoding="utf-8") as f:
        case_data = json.load(f)
    case_text = case_data.get("病例内容", "")
    info = {}
    patterns = {
        "age": r"年龄[:：]\s*(\d+)",
        "gender": r"性别[:：]\s*(\S+)",
        "diagnosis": r"初步诊断[:：]?\s*\n*([\s\S]*?)(?=\s*管理建议[:：]|$)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, case_text)
        if match:
            info[key] = match.group(1).strip()
    return info

def load_plan_data():
    with open("plan_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    st.title("用户定制饮食计划")

    try:
        with open("diet_requirement.json", "r", encoding="utf-8") as f:
            diet_req = json.load(f)
    except FileNotFoundError:
        st.error("饮食偏好数据不存在，请先填写“填写饮食偏好”页面。")
        st.stop()

    case_info = load_case_info()
    PLAN_DATA = load_plan_data()

    # 显示用户信息
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"""
        <div style="text-align:center;">
        <div style="font-size:28px; font-weight:bold; color:#2f6fdb;">{case_info.get('age', '未知')}岁 <span style="font-size:22px;">{case_info.get('gender', '未知')}</span></div>
        <div style="font-size:14px; color:#888;">基本信息</div>
        </div>
    """, unsafe_allow_html=True)
    col2.markdown(f"""
        <div style="text-align:center;">
        <div style="font-size:28px; font-weight:bold; color:#2c8c2c;">{case_info.get('diagnosis', '糖尿病前期')}</div>
        <div style="font-size:14px; color:#888;">健康状况</div>
        </div>
    """, unsafe_allow_html=True)
    col3.markdown(f"""
        <div style="text-align:center;">
        <div style="font-size:28px; font-weight:bold; color:#9558b2;">{diet_req.get('activity', '中')}</div>
        <div style="font-size:14px; color:#888;">活动水平</div>
        </div>
    """, unsafe_allow_html=True)
    prefs = diet_req.get("preference", [])
    formatted_prefs = "<br>".join(prefs)
    col4.markdown(f"""
        <div style="text-align:center;">
            <div style="font-size:16px; font-weight:500; color:#e36e0a; line-height:1.6; max-width:160px; margin:auto; word-wrap:break-word;">
                {formatted_prefs}
            </div>
            <div style="font-size:14px; color:#888; margin-top:4px;">饮食偏好</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("本周营养概览")
    nutrition = PLAN_DATA["weekly_nutrition"]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div style="text-align:center;"><div style="font-size:28px; font-weight:bold;">{nutrition['avg_calories']}</div><div style="font-size:14px; color:#888;">平均热量</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div style="text-align:center;"><div style="font-size:28px; font-weight:bold;">{nutrition['carb_percent']}%</div><div style="font-size:14px; color:#888;">碳水</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div style="text-align:center;"><div style="font-size:28px; font-weight:bold;">{nutrition['protein_percent']}%</div><div style="font-size:14px; color:#888;">蛋白</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div style="text-align:center;"><div style="font-size:28px; font-weight:bold;">{nutrition['fat_percent']}%</div><div style="font-size:14px; color:#888;">脂肪</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    for day_key, day_plan in PLAN_DATA["daily_meals"].items():
        st.subheader(f"📅 {day_plan['label']}")
        left_col, right_col = st.columns([1, 5])
        with left_col:
            st.button(day_plan["label"], disabled=True)
            st.markdown(f"全天总热量\n\n### {day_plan['total_calories']} 卡")
        with right_col:
            meal_columns = st.columns(3)
            for i, meal in enumerate(day_plan["meals"]):
                with meal_columns[i]:
                    meal_color = {"早餐": "#f8d7b3", "午餐": "#d8f3dc", "晚餐": "#c7d9f1"}.get(meal["name"], "#ddd")
                    time_c = {"早餐": "#c47c29", "午餐": "#4c8c4a", "晚餐": "#3667c7"}.get(meal["name"], "#666")
                    st.markdown(f"""<div style="background:{meal_color}; padding:6px 12px; border-radius:8px; width:fit-content; font-size:20px; font-weight:bold; color:{time_c};">{meal['name']} ({meal['time']})</div>""", unsafe_allow_html=True)
                    for item in meal["items"]:
                        st.markdown(f"""
                            <div style="background-color:#f0f0f0; padding:10px 12px; border-radius:10px; margin-bottom:10px;">
                                <div style="font-weight:bold; font-size:16px; margin-bottom:6px;">{item['name']}</div>
                                <div style="color:#555; font-size:14px;">
                                    🔥 {item['calories']}卡 &nbsp;&nbsp; ⏰ {item['cook_time']}分钟<br>
                                    GI值: {item.get('GI', '无')}<br>
                                    {'<br>'.join(f'- {d}' for d in item['details'])}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown(f"**小计: {meal['subtotal']} 卡**")
