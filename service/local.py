"""
本地模型服务 - 使用本地部署的Qwen2.5-VL模型进行推理
"""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

class LocalModelService:
    """本地模型服务类，负责加载模型和处理推理请求"""
    
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        初始化本地模型服务
        
        参数:
            model_path: 模型路径，默认使用Hugging Face上的Qwen2.5-VL-7B-Instruct
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        
    def load_model(self):
        """
        加载模型和处理器
        
        返回:
            bool: 加载是否成功
        """
        try:
            print("正在加载本地模型，这可能需要一些时间...")
            # 使用bfloat16精度和Flash Attention 2来优化性能
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, 
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            print("本地模型加载完成！")
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False
    
    def inference(self, image, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024):
        """
        使用本地模型进行推理
        
        参数:
            image: PIL Image对象或图像路径
            prompt: 提示文本
            system_prompt: 系统提示
            max_new_tokens: 生成的最大token数
            
        返回:
            tuple: (输出文本, 输入高度, 输入宽度)
        """
        # 确保模型已加载
        if self.model is None or self.processor is None:
            success = self.load_model()
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
        
        # 应用聊天模板
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 处理输入
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)
        
        # 生成输出
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # 获取输入图像尺寸
        input_height = inputs['image_grid_thw'][0][1] * 14
        input_width = inputs['image_grid_thw'][0][2] * 14
        
        return output_text[0], input_height, input_width