import base64
import io

from PIL import Image


def encode_image(image):
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        # 保存为JPEG格式到缓冲区
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    else:
        raise TypeError(f"不支持的图像类型: {type(image)}")


def parse_json(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            text = "\n".join(lines[i + 1:])
            text = text.split("```")[0]
            break
    return text


def create_messages(image, prompt, system_prompt=None, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28):
    messages = []

    # 添加系统消息
    if system_prompt:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system_prompt}]})
    # 添加用户消息
    user_content = [{"type": "image_url",
                     "min_pixels": min_pixels, "max_pixels": max_pixels,
                     "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                    {"type": "text",
                     "text": prompt}, ]

    messages.append({"role": "user",
                     "content": user_content})
    return messages
