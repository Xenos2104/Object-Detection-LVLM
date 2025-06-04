import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from config import *


class LocalModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.processor = None

    def load(self):
        try:
            print("正在加载本地模型，这可能需要一些时间")
            # bfloat16
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path,
                                                                            torch_dtype=torch.bfloat16,
                                                                            device_map="auto")
            self.processor = AutoProcessor.from_pretrained('model/Qwen2.5-VL-3B-Instruct')
            print("本地模型加载完成")
            return True
        except Exception as e:
            print(f"模型加载失败：{str(e)}")
            return False

    def inference(self, image, prompt, system_prompt=SYSTEM_PROMPT, max_tokens=MAX_TOKENS):
        if self.model is None or self.processor is None:
            success = self.load()
            if not success:
                return "模型加载失败，请检查模型路径或环境配置。", None, None

        # 处理图像输入
        if isinstance(image, str):
            image = Image.open(image)

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "image": image
                    }
                ]
            }
        ]

        # 应用模板
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 处理输入
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)
        # 生成输出
        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)

        # 获取输入图像尺寸
        input_height = inputs['image_grid_thw'][0][1] * 14
        input_width = inputs['image_grid_thw'][0][2] * 14

        return output_text[0], input_height, input_width
