import gradio as gr

def build_ui(
    on_file_upload,
    on_delete,
    on_send,
    on_clear_history,
    on_generate_case
):
    css = """
    body, .gradio-container {
        background: linear-gradient(120deg, #e3f0ff 0%, #f8fbff 100%) !important;
        min-height: 100vh;
        font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif;
    }
    #main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1976d2;
        margin-bottom: 0.7em;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px #e3f0ff;
    }
    .gr-box, .gr-group, .gradio-container .gr-block.gr-box {
        border-radius: 14px !important;
        box-shadow: 0 2px 12px #e3f0ff;
        background: #ffffff;
        padding: 18px 18px 10px 18px;
        margin-bottom: 18px;
        border: none !important;
    }
    .gr-input, .gr-textbox, .gr-dropdown {
        border-radius: 8px !important;
        border: 1px solid #cfd8dc !important;
        background: #f5faff !important;
        font-size: 1.05rem;
        min-height: 38px;
    }
    #personal-info-row {
        margin-bottom: 0.7em;
        gap: 1.2em;
    }
    #case-panel {
        background: #f5faff !important;
        border-radius: 14px !important;
        min-height: 340px;
        padding: 18px;
        box-shadow: 0 2px 8px #e3f0ff;
    }
    #gen-case-btn {
        background: linear-gradient(90deg, #1976d2 0%, #64b5f6 100%);
        color: #fff !important;
        font-weight: bold;
        border-radius: 8px !important;
        margin-top: 1.2em;
        min-height: 44px;
        font-size: 1.08rem;
        box-shadow: 0 2px 8px #1976d222;
        border: none !important;
        transition: background 0.2s;
    }
    #gen-case-btn:hover {
        background: linear-gradient(90deg, #64b5f6 0%, #1976d2 100%);
    }
    #clear-btn {
        min-width: 90px;
        max-width: 200px;
        background: #e3f0ff !important;
        color: #1976d2 !important;
        border-radius: 8px !important;
        font-weight: bold;
        border: none !important;
        box-shadow: 0 1px 4px #e3f0ff;
    }
    #clear-btn:hover {
        background: #bbdefb !important;
        color: #0d47a1 !important;
    }
    #file-selector .gr-checkbox {
        padding: 8px 8px 8px 28px;
        position: relative;
        border-radius: 4px;
        transition: background-color 0.2s;
    }
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
    #file-selector .gr-checkbox:hover {
        background-color: #e3f0ff;
    }
    .gradio-container .gr-block.gr-chatbot {
        background: #f5faff !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 8px #e3f0ff;
    }
    .gradio-container .gr-block.gr-markdown {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .gradio-container .gr-block label {
        font-weight: 500;
        color: #1976d2;
        font-size: 1.05rem;
    }
    .gradio-container .gr-block input, .gradio-container .gr-block textarea {
        font-size: 1.05rem;
    }
    """

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            "<div id='main-title'>糖尿病助手 🩸 <span style='font-size:1.5rem;font-weight:normal;color:#222;'>— 左：对话交互     — 右：病例记录</span></div>"
        )

        with gr.Row(elem_id="personal-info-row"):
            name_input = gr.Textbox(label="姓名", placeholder="请输入姓名", lines=1, elem_classes="gr-input")
            age_input = gr.Textbox(label="年龄", placeholder="请输入年龄", lines=1, elem_classes="gr-input")
            weight_input = gr.Textbox(label="体重（kg）", placeholder="请输入体重", lines=1, elem_classes="gr-input")
            gender_input = gr.Dropdown(label="性别", choices=["男", "女"], value=None, elem_classes="gr-input")
            history_input = gr.Textbox(label="既往史", placeholder="请输入既往史", lines=1, elem_classes="gr-input")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group():
                    chatbot = gr.Chatbot(
                        type="messages",
                        label="对话记录",
                        height=500,
                        value=[{"role": "assistant", "content": "您好，我是糖尿病专业助手，请您提供详细病例信息，以便我为您量身定制医学建议。"}]
                    )
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
                        scale=2,
                        elem_classes="gr-input"
                    )
                    send_btn = gr.Button("发送", elem_id="send-btn", scale=1)
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
                            "糖尿病并发症有哪些？",
                            "血糖高有哪些症状？",
                        ],
                        inputs=[text_input]
                    )
                    clear_btn = gr.Button("清除对话历史", elem_id="clear-btn", scale=1)

            with gr.Column(scale=2):
                with gr.Group(elem_id="case-panel"):
                    case_md = gr.Markdown("**病例记录**\n\n尚无内容")
                gen_case_btn = gr.Button("生成病例报告单", elem_id="gen-case-btn")

        file_list = gr.State([])
        state = gr.State([{"role": "assistant", "content": "您好，我是糖尿病专业助手，请您提供详细病例信息，以便我为您量身定制医学建议。"}])

        upload_btn.upload(
            fn=on_file_upload,
            inputs=[upload_btn, state, file_list],
            outputs=[chatbot, state, file_list, file_selector]
        )
        file_selector.change(
            fn=on_delete,
            inputs=[file_selector, file_list],
            outputs=[file_list, file_selector]
        )
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
        clear_btn.click(
            fn=on_clear_history,
            inputs=None,
            outputs=[chatbot, state, case_md]
        )
        gen_case_btn.click(
            fn=on_generate_case,
            inputs=[state, name_input, age_input, weight_input, gender_input, history_input],
            outputs=[case_md]
        )

    return demo
