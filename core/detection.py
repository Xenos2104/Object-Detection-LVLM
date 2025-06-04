"""
核心处理逻辑 - 处理用户请求和生成结果
"""

import json
from PIL import Image
import numpy as np

from config import MIN_PIXELS, MAX_PIXELS, WELCOME_MESSAGE, USE_LOCAL_MODEL
from service.api import call_vision_api, parse_json_response
from .annotate import annotate_image
from prompt import format_prompt, DETECTION_PROMPT
from service.local import LocalModelService
from qwen_vl_utils import smart_resize  # 外部库

# 创建本地模型服务实例
local_model_service = None
if USE_LOCAL_MODEL:
    local_model_service = LocalModelService()


def process_detection(image, text_query):
    """
    处理目标检测请求
    
    参数:
        image: 输入图像（PIL Image、路径或numpy数组）
        text_query: 文本查询
        
    返回:
        tuple: (文本回答, 标注后的图像)
    """
    # 输入验证
    if not text_query.strip():
        return "请输入检测查询内容以开始分析。", None
    if image is None:
        return "请先上传图像进行检测。", None

    # 处理不同类型的输入图像
    if isinstance(image, str):
        # 如果是文件路径，打开图片
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        # 如果是numpy数组，转换为PIL Image
        image = Image.fromarray(image)
    else:
        # 如果已经是PIL Image对象，复制一份
        image = image.copy()

    # 获取图像尺寸并进行智能调整
    width, height = image.size
    input_height, input_width = smart_resize(
        height, width,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS
    )

    # 格式化提示
    prompt = format_prompt(DETECTION_PROMPT, query=text_query)

    # 根据配置选择使用本地模型或API
    if USE_LOCAL_MODEL and local_model_service:
        # 使用本地模型进行推理
        api_response, input_height, input_width = local_model_service.inference(
            image,
            prompt
        )
    else:
        # 使用API进行推理
        api_response = call_vision_api(
            image,
            prompt,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS
        )

    # 解析API响应
    json_str = parse_json_response(api_response)
    try:
        result = json.loads(json_str)

        # 提取结果
        text_response = result.get("answer", "")
        detections = result.get("detections", [])

        # 在图像上标注检测结果
        annotated_image = annotate_image(
            image,
            detections,
            output_path=None,  # 不保存到文件
            input_width=input_width,
            input_height=input_height
        )

        return text_response, annotated_image
    except json.JSONDecodeError:
        return "解析结果时出错，请重试。", None


def clear_inputs():
    """
    清空所有输入和输出
    
    返回:
        tuple: (None, "", 欢迎消息, None) 分别对应图像输入、文本输入、文本输出、图像输出
    """
    return None, "", WELCOME_MESSAGE, None