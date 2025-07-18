import streamlit as st
import json

def main():
    st.title("填写饮食偏好")

    preferences = []

    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("少肉主义"):
            preferences.append("少肉主义")
        if st.checkbox("不吃海鲜"):
            preferences.append("不吃海鲜")
        if st.checkbox("低脂饮食"):
            preferences.append("低脂饮食")
        if st.checkbox("无乳糖"):
            preferences.append("无乳糖")
        if st.checkbox("不吃辛辣"):
            preferences.append("不吃辛辣")
    with col2:
        if st.checkbox("低盐饮食"):
            preferences.append("低盐饮食")
        if st.checkbox("不吃乳制品"):
            preferences.append("不吃乳制品")
        if st.checkbox("无麸质"):
            preferences.append("无麸质")
        if st.checkbox("不吃坚果"):
            preferences.append("不吃坚果")
        if st.checkbox("不吃生食"):
            preferences.append("不吃生食")

    st.write("宗教饮食限制")
    col3, col4 = st.columns(2)
    with col3:
        if st.checkbox("清真饮食"):
            preferences.append("清真饮食")
        if st.checkbox("忌猪肉"):
            preferences.append("忌猪肉")
    with col4:
        if st.checkbox("忌牛肉"):
            preferences.append("忌牛肉")
        if st.checkbox("纯素食"):
            preferences.append("纯素食")

    st.write("过敏与忌口")
    col5, col6 = st.columns(2)
    with col5:
        if st.checkbox("对坚果过敏"):
            preferences.append("对坚果过敏")
        if st.checkbox("对牛奶过敏"):
            preferences.append("对牛奶过敏")
    with col6:
        if st.checkbox("对海鲜过敏"):
            preferences.append("对海鲜过敏")
        if st.checkbox("对鸡蛋过敏"):
            preferences.append("对鸡蛋过敏")

    other_restriction = st.text_input("其他忌口")
    if other_restriction.strip():
        preferences.append(f"其他忌口：{other_restriction.strip()}")

    activity_level = st.selectbox("活动水平", ["请选择您的日常活动水平", "低", "中", "高"])

    if st.button("保存饮食需求"):
        data = {
            "preference": preferences,
            "activity": activity_level if activity_level != "请选择您的日常活动水平" else "中"
        }
        # 保存文件
        with open("diet_requirement.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # 保存到session_state
        st.session_state['diet_requirement'] = data

        st.success("保存成功！正在跳转到饮食计划生成页...")
        st.session_state['current_page'] = "llm_page"
        st.rerun()  # <--- 更新为 st.rerun()

# 如果你是用 app.py 统一调度，可以加：
if __name__ == "__main__":
    main()
