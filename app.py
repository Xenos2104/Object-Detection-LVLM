"""
主应用程序 - 集成UI和功能模块
"""

import gradio as gr
import torch

from config import (
    CUSTOM_CSS, HTML_HEADER, HTML_INFO, 
    HTML_IMAGE_UPLOAD, HTML_DETECTION_RESULT, 
    HTML_TEXT_QUERY, HTML_AI_ANALYSIS,
    USE_LOCAL_MODEL
)
from core.detection import process_detection, clear_inputs

# 创建Gradio界面
with gr.Blocks(title="LVLM目标检测系统", theme=gr.themes.Soft(), css=CUSTOM_CSS) as app:

    # 标题区域
    gr.HTML(HTML_HEADER)

    # 系统说明
    gr.HTML(HTML_INFO)



    # 图像上传和标注结果区域
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(HTML_IMAGE_UPLOAD)
            image_input = gr.Image(
                label="输入图像",
                type="pil",
                height=500,
            )

        with gr.Column(scale=1):
            gr.HTML(HTML_DETECTION_RESULT)
            image_output = gr.Image(
                label="标注图像",
                height=460,
                show_download_button=True,
            )

    # 文本查询和分析结果区域
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(HTML_TEXT_QUERY)
            text_input = gr.Textbox(
                show_label=False,
                placeholder="请输入您的检测需求，例如：\n• 检测图像中的人物\n• 识别车辆类型\n• 统计目标数量",
                lines=15,
                max_lines=15
            )

        with gr.Column(scale=1):
            gr.HTML(HTML_AI_ANALYSIS)
            text_output = gr.Markdown(
                label="系统分析结果",
                value="欢迎使用目标检测系统，请上传图像并输入查询内容。",
                show_copy_button=True,
                height=250,
            )

    # 控制按钮
    with gr.Row():
        with gr.Column(scale=2):  # 左侧空白
            gr.HTML("")
        with gr.Column(scale=3):  # 按钮区域
            with gr.Row():
                detect_btn = gr.Button("🚀 开始检测", variant="primary")
                clear_btn = gr.Button("🗑️ 清空", variant="secondary")
        with gr.Column(scale=2):  # 右侧空白
            gr.HTML("")

        # 模型选择区域
    with gr.Row(visible=torch.cuda.is_available()):  # 只有在有GPU时才显示模型选择
        with gr.Column(scale=1):
            gr.HTML("")
        with gr.Column(scale=3):
            with gr.Group(elem_classes=["model-toggle"]):
                model_choice = gr.Radio(
                    ["API模式", "本地模式"],
                    label="选择LVLM",
                    value="API模式" if not USE_LOCAL_MODEL else "本地模式",
                    info="API模式需要网络连接，本地模式至少需要一张RTX 3090 24GB"
                )
        with gr.Column(scale=1):
            gr.HTML("")


    # 模型选择功能
    def update_model_choice(choice):
        import importlib
        import sys
        
        # 动态修改配置
        config_module = sys.modules['main.config.config']
        setattr(config_module, 'USE_LOCAL_MODEL', choice == "本地模型模式")
        
        # 重新加载核心模块以应用新配置
        if 'main.core.detection' in sys.modules:
            importlib.reload(sys.modules['main.core.detection'])
        
        return f"已切换到{choice}，请点击检测按钮进行测试。"

    model_choice.change(
        fn=update_model_choice,
        inputs=[model_choice],
        outputs=[text_output]
    )

    # 事件绑定
    detect_btn.click(
        fn=process_detection,
        inputs=[image_input, text_input],
        outputs=[text_output, image_output]
    )

    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[image_input, text_input, text_output, image_output]
    )

    # 支持回车键触发检测
    text_input.submit(
        fn=process_detection,
        inputs=[image_input, text_input],
        outputs=[text_output, image_output]
    )

# 启动应用
if __name__ == "__main__":
    # 检查是否有CUDA支持
    if USE_LOCAL_MODEL and not torch.cuda.is_available():
        print("警告：本地模型模式需要CUDA支持，但未检测到可用的GPU。将自动切换到API模式。")
        # 动态修改配置
        import sys
        config_module = sys.modules['main.config.config']
        setattr(config_module, 'USE_LOCAL_MODEL', False)
    
    app.launch(
        share=False,
        debug=True,
        server_name="127.0.0.1",
        server_port=7860
    )