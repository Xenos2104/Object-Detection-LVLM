import json

from qwen_vl_utils import smart_resize

from config import *
from prompt import format_prompt, PROMPT
from service.api import api_model
from service.local import LocalModel
from .annotate import annotate
from utils import parse_json

# 本地模型
local_model = None
if USE_LOCAL_MODEL:
    local_model = LocalModel()


def detect(image, text):
    if not text.strip():
        return "请输入检测查询内容以开始分析。", None
    if image is None:
        return "请先上传图像进行检测。", None

    # 获取图像尺寸并进行调整
    width, height = image.size
    input_height, input_width = smart_resize(height, width, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    # 格式化提示
    prompt = format_prompt(PROMPT, query=text)

    # 选择使用本地模型或API
    if USE_LOCAL_MODEL and local_model:
        # 使用本地模型
        response, input_height, input_width = local_model.inference(image, prompt)
    else:
        # 使用API
        response = api_model(image, prompt, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    # 解析JSON
    response = parse_json(response)
    try:
        response = json.loads(response)
        # 提取结果
        answer = response.get("answer", "")
        detections = response.get("detections", [])
        # 标注结果
        detections = annotate(image,
                              detections,
                              output_path=SAVE_OUTPUT,
                              input_width=input_width,
                              input_height=input_height)

        return answer, detections
    except json.JSONDecodeError:
        return "解析结果时出错，请重试。", None


def clear():
    return None, "", WELCOME_MESSAGE, None
