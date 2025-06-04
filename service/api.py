"""
API服务 - 负责与大模型API的通信
"""

from openai import OpenAI

from config import API_KEY, API_BASE_URL, MODEL_ID, SYSTEM_PROMPT
from core.annotate import encode_image


def parse_json_response(json_output):
    """
    解析API返回的JSON格式响应

    参数:
        json_output: 包含JSON的字符串，可能包含markdown格式

    返回:
        str: 提取出的JSON字符串
    """
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # 移除```json前的内容
            json_output = json_output.split("```")[0]  # 移除结束```后的内容
            break
    return json_output


def call_vision_api(image, prompt, min_pixels, max_pixels):
    """
    调用视觉语言模型API进行图像分析

    参数:
        image: PIL Image对象或图像路径
        prompt: 提示文本
        min_pixels: 最小像素数
        max_pixels: 最大像素数

    返回:
        str: API返回的原始响应文本
    """
    # 将图像编码为base64
    base64_image = encode_image(image)

    # 创建OpenAI客户端
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    # 构建请求消息
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 发送请求并获取响应
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
    )

    return completion.choices[0].message.content