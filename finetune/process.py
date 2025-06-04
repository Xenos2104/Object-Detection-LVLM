import json
import os

from datasets import load_dataset
from qwen_vl_utils import smart_resize
from tqdm import tqdm


def convert_to_qwen25vl_format(bbox, old_height, old_width, factor=28,
                               min_pixels=56 * 56,
                               max_pixels=14 * 14 * 4 * 1280):
    new_height, new_width = smart_resize(old_height, old_width, factor, min_pixels, max_pixels)
    scale_w = new_width / old_width
    scale_h = new_height / old_height

    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)

    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))

    return [x1_new, y1_new, x2_new, y2_new]


def convert_to_sft_format(data_path, save_path, type='train'):
    # 加载数据集
    dataset = load_dataset(data_path, split='train')

    # 每个数据保存到一个jsonl文件中
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建 JSONL
    jsonl_file = os.path.join(save_path, f"{type}.jsonl")
    with open(jsonl_file, 'w', encoding='utf-8') as jsonl_out:
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            if type == 'train':
                if idx >= 3000:
                    break
            elif type == 'test':
                if idx < 3000 or idx >= 3100:
                    continue
            # 保存图片
            image = sample['image']
            # 生成文件名
            filename = f"{idx + 1:06d}.jpg"
            jpg_path = os.path.join(save_path, type)
            if not os.path.exists(jpg_path):
                os.makedirs(jpg_path)
            output_path = os.path.join(jpg_path, filename)
            # 保存图片
            image.save(output_path)

            old_bbox = sample['bbox']
            image_width, image_height = image.size
            x1, y1, w, h = old_bbox
            new_bboxes = [x1, y1, x1 + w, y1 + h]
            # 转换坐标
            qwen25_bboxes = convert_to_qwen25vl_format(new_bboxes, image_height, image_width)
            bbox_dict = {"bbox_2d": qwen25_bboxes}
            formatted_json = json.dumps(bbox_dict, indent=None)
            data = {"image": [output_path],
                    "query": sample['question'],
                    "response": formatted_json}

            jsonl_out.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    convert_to_sft_format(data_path='./textvqa_bbox', save_path='./data', type='train')
    convert_to_sft_format(data_path='./textvqa_bbox', save_path='./data', type='test')
