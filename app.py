import streamlit as st
import pages.page1 as pref_page
import pages.llm_page as llm_page
import pages.page2 as plan_page

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "page1"

if st.session_state['current_page'] == "page1":
    pref_page.main()
elif st.session_state['current_page'] == "llm_page":
    llm_page.main()
elif st.session_state['current_page'] == "page2":
    plan_page.main()
