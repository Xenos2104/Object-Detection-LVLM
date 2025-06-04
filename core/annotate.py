from PIL import ImageDraw, ImageFont


def annotate(image, detections, output_path=None, input_width=None, input_height=None):
    # 获取当前图像尺寸
    width, height = image.size
    draw = ImageDraw.Draw(image)

    # 加载字体
    try:
        font = ImageFont.truetype("simhei.ttf", 40)
    except:
        font = ImageFont.load_default()

    # 颜色列表，用于不同对象的标注
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange',
              'pink', 'brown', 'gray', 'turquoise', 'cyan', 'magenta',
              'lime', 'navy', 'maroon', 'teal', 'olive', 'coral',
              'lavender', 'violet', 'gold']

    # 为检测结果绘制边界框和标签
    for i, detection in enumerate(detections):
        bbox = detection['bbox_2d']
        label = detection['label']

        # 选择颜色
        color = colors[i % len(colors)]

        # 坐标转换逻辑
        if len(bbox) == 4:
            if input_width is not None and input_height is not None:
                abs_x1 = int(bbox[0] / input_width * width)
                abs_y1 = int(bbox[1] / input_height * height)
                abs_x2 = int(bbox[2] / input_width * width)
                abs_y2 = int(bbox[3] / input_height * height)

                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1
                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1

                x1, y1, x2, y2 = abs_x1, abs_y1, abs_x2, abs_y2
            else:
                x1, y1, x2, y2 = bbox

            text = f"{label}"
            bbox_text = draw.textbbox((x1, y1 - 30), text, font=font)
            text_width = bbox_text[2] - bbox_text[0]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.rectangle([x1, y1 - 30, x1 + text_width, y1], fill=color)
            draw.text((x1, y1 - 30), text, fill='white', font=font)


    # 保存图片
    if output_path:
        image.save(output_path)

    return image
