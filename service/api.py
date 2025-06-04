from openai import OpenAI

from config import API_KEY, API_BASE_URL, MODEL_NAME, SYSTEM_PROMPT
from utils import encode_image, create_messages


def api_model(image, prompt, min_pixels, max_pixels):
    image = encode_image(image)
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    # 请求消息
    messages = create_messages(image, prompt, SYSTEM_PROMPT, min_pixels, max_pixels)
    # 发送请求并获取响应
    completion = client.chat.completions.create(model=MODEL_NAME, messages=messages)
    return completion.choices[0].message.content
