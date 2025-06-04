"""
图像处理服务 - 负责图像编码、处理和标注
"""

import base64
import io
from PIL import Image, ImageDraw, ImageFont


def encode_image(image_input):
    """
    将图像编码为base64格式

    参数:
        image_input: 可以是文件路径(str)或PIL Image对象

    返回:
        str: base64编码后的图像字符串
    """
    if isinstance(image_input, str):
        # 如果是文件路径
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image_input, Image.Image):
        # 如果是PIL Image对象
        buffer = io.BytesIO()
        # 保存为JPEG格式到内存缓冲区
        image_input.save(buffer, format='JPEG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    else:
        raise TypeError(f"不支持的图像类型: {type(image_input)}")


def annotate_image(image, detections, output_path=None, input_width=None, input_height=None):
    """
    在图像上标注检测结果

    参数:
        image: PIL Image对象或图像路径
        detections: 检测结果列表，每个元素包含bbox_2d和label
        output_path: 可选，输出图像保存路径
        input_width: 可选，原始输入图像宽度（用于坐标转换）
        input_height: 可选，原始输入图像高度（用于坐标转换）

    返回:
        PIL.Image: 标注后的图像对象
    """
    # 如果输入是路径，转换为PIL Image对象
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, Image.Image):
        image = image.copy()  # 创建副本避免修改原图

    # 获取当前图像尺寸
    current_width, current_height = image.size

    draw = ImageDraw.Draw(image)

    # 尝试加载字体（中文支持）
    try:
        font = ImageFont.truetype("simhei.ttf", 40)
    except:
        try:
            font = ImageFont.truetype("simhei.ttf", size=14)
        except:
            font = ImageFont.load_default()

    # 颜色列表，用于不同对象的标注
    colors = [
        'red', 'green', 'blue', 'yellow', 'purple', 'orange',
        'pink', 'brown', 'gray', 'turquoise', 'cyan', 'magenta',
        'lime', 'navy', 'maroon', 'teal', 'olive', 'coral',
        'lavender', 'violet', 'gold'
    ]

    # 为每个检测结果绘制边界框和标签
    for i, detection in enumerate(detections):
        bbox = detection['bbox_2d']
        label = detection['label']

        # 选择颜色（循环使用颜色列表）
        color = colors[i % len(colors)]

        # 坐标转换逻辑
        if len(bbox) == 4:
            if input_width is not None and input_height is not None:
                # 如果提供了原始尺寸，进行坐标转换
                abs_x1 = int(bbox[0] / input_width * current_width)
                abs_y1 = int(bbox[1] / input_height * current_height)
                abs_x2 = int(bbox[2] / input_width * current_width)
                abs_y2 = int(bbox[3] / input_height * current_height)

                # 确保坐标顺序正确
                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1
                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1

                x1, y1, x2, y2 = abs_x1, abs_y1, abs_x2, abs_y2
            else:
                # 直接使用原始坐标
                x1, y1, x2, y2 = bbox

            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

            # 绘制标签
            text = f"{label}"

            # 获取文本边界框并绘制背景和文本
            try:
                bbox_text = draw.textbbox((x1, y1 - 30), text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]

                # 绘制文本背景
                draw.rectangle([x1, y1 - 30, x1 + text_width, y1], fill=color)

                # 绘制文本
                draw.text((x1, y1 - 30), text, fill='white', font=font)
            except:
                # 如果textbbox不支持，使用简单方式
                draw.text((x1 + 8, y1 + 6), text, fill=color, font=font)

    # 保存图片（如果提供了输出路径）
    if output_path:
        image.save(output_path)

    return image