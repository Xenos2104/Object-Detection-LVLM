import os
import dotenv

dotenv.load_dotenv()

# 模型配置
USE_LOCAL_MODEL = True
MODEL_PATH = "./model/Qwen2.5-VL-3B-Instruct-X"  # 本地模型路径
MAX_TOKENS = 2048  # 生成的token数

# API配置
API_KEY = os.getenv('DASHSCOPE_API_KEY')
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen2.5-vl-72b-instruct"

# 图像配置
MIN_PIXELS = 512 * 28 * 28  # 最小像素数
MAX_PIXELS = 2048 * 28 * 28  # 最大像素数
SAVE_OUTPUT = 'output.jpg' # 保存标注结果

# UI配置
CSS = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 1rem;
}

.header-section {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border: 1px solid #dee2e6;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.info-section {
    background: #ffffff;
    border: 1px solid #e3f2fd;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
    border-left: 4px solid #2196f3;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

.section-title {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 0.8rem;
    margin-bottom: 1rem;
    text-align: center;
    font-weight: 600;
    color: #495057;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.gradio-button.primary {
    background: linear-gradient(135deg, #ffffff 0%, #f0f8ff 100%) !important;
    border: 2px solid #2196f3 !important;
    color: #2196f3 !important;
    font-weight: 600 !important;
}

.gradio-button.primary:hover {
    background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%) !important;
    border-color: #1976d2 !important;
    color: #1976d2 !important;
}

.gradio-button.secondary {
    background: #ffffff !important;
    border: 1px solid #dee2e6 !important;
    color: #6c757d !important;
}

.gradio-button.secondary:hover {
    background: #f8f9fa !important;
    border-color: #adb5bd !important;
}

.model-toggle {
    margin-top: 10px;
    padding: 10px;
    background: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #dee2e6;
}
"""

# 系统提示
SYSTEM_PROMPT = "You are a helpful assistant."
WELCOME_MESSAGE = "欢迎使用目标检测系统，请上传图像并输入查询内容。"

# HTML内容
HTML_HEADER = """
<div class="header-section">
    <h1 style="margin: 0; color: #343a40; font-size: 2.2rem; font-weight: 700;">
        机器视觉与目标检测期末作业
    </h1>
    <p style="margin: 0.8rem 0 0 0; color: #6c757d; font-size: 1.1rem;">
        基于LVLM的智能目标检测系统
    </p>
</div>
"""

HTML_INFO = """
<div class="info-section">
    <h4 style="margin-top: 0; color: #2196f3; font-size: 1.1rem;">📋 系统说明</h4>
    <p style="margin: 0; color: #495057; line-height: 1.6; font-size: 0.95rem;">
        本系统实现了基于LVLM的目标检测功能，支持自然语言查询和图像分析。
        用户可以上传图像并输入检测需求，系统将返回文本分析结果和标注后的图像。
        系统支持API模式和本地模式，可以根据需要切换。
    </p>
</div>
"""

HTML_IMAGE_UPLOAD = '<div class="section-title">🖼️ 图像上传</div>'
HTML_DETECTION_RESULT = '<div class="section-title">🎨 检测结果</div>'
HTML_TEXT_QUERY = '<div class="section-title">💬 文本查询</div>'
HTML_AI_ANALYSIS = '<div class="section-title">🤖 分析报告</div>'