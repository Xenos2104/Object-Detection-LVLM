import importlib

import gradio as gr
import torch

from config import *
from core.detect import detect, clear

# 创建Gradio界面
with gr.Blocks(title="LVLM目标检测系统", theme=gr.themes.Soft(), css=CSS) as app:
    # 标题区域
    gr.HTML(HTML_HEADER)
    # 系统说明
    gr.HTML(HTML_INFO)
    # 图像上传和标注结果区域
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(HTML_IMAGE_UPLOAD)
            image_input = gr.Image(label="输入图像",
                                   type="pil",
                                   height=500)

        with gr.Column(scale=1):
            gr.HTML(HTML_DETECTION_RESULT)
            image_output = gr.Image(label="标注图像",
                                    height=460,
                                    show_download_button=True)
    # 文本查询和分析结果区域
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(HTML_TEXT_QUERY)
            text_input = gr.Textbox(show_label=False,
                                    placeholder="请输入您的检测需求。",
                                    lines=15,
                                    max_lines=15)

        with gr.Column(scale=1):
            gr.HTML(HTML_AI_ANALYSIS)
            text_output = gr.Markdown(label="分析结果",
                                      value="请上传图像并输入查询内容。",
                                      show_copy_button=True,
                                      height=250)
    # 控制按钮
    with gr.Row():
        with gr.Column(scale=2):  # 左侧空白
            gr.HTML("")
        with gr.Column(scale=3):  # 按钮区域
            with gr.Row():
                detect_btn = gr.Button("🚀 检测", variant="primary")
                clear_btn = gr.Button("🗑️ 清空", variant="secondary")
        with gr.Column(scale=2):  # 右侧空白
            gr.HTML("")

    # 模型选择
    with gr.Row(visible=torch.cuda.is_available()):
        with gr.Column(scale=1):
            gr.HTML("")
        with gr.Column(scale=3):
            with gr.Group(elem_classes=["model-toggle"]):
                model_choice = gr.Radio(["API模式", "本地模式"],
                                        label="选择LVLM",
                                        value="API模式" if not USE_LOCAL_MODEL else "本地模式",
                                        info="API模式需要网络连接，本地模式至少需要一张RTX 3090 24GB")
        with gr.Column(scale=1):
            gr.HTML("")


    def update_model_choice(choice):
        import sys
        # 动态修改配置
        config_module = sys.modules['config']
        setattr(config_module, 'USE_LOCAL_MODEL', choice == "本地模式")

        # 重新加载核心模块以应用新配置
        if 'core.detect' in sys.modules:
            importlib.reload(sys.modules['core.detect'])

        return f"已切换到{choice}。"


    model_choice.change(fn=update_model_choice,
                        inputs=[model_choice],
                        outputs=[text_output])

    # 事件绑定
    detect_btn.click(fn=detect,
                     inputs=[image_input, text_input],
                     outputs=[text_output, image_output])

    clear_btn.click(fn=clear,
                    inputs=[],
                    outputs=[image_input, text_input, text_output, image_output])

    # 回车键
    text_input.submit(fn=detect,
                      inputs=[image_input, text_input],
                      outputs=[text_output, image_output])

# 启动应用
if __name__ == "__main__":
    if USE_LOCAL_MODEL and not torch.cuda.is_available():
        print("本地模式需要CUDA支持，但未检测到可用的GPU。将自动切换到API模式。")
        import sys

        config_module = sys.modules['config']
        setattr(config_module, 'USE_LOCAL_MODEL', False)

    app.launch(share=True,
               debug=True,
               server_name="127.0.0.1",
               server_port=7860)
