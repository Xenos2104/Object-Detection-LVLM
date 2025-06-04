# 模型微调
这里包含了用于模型微调的脚本，基于开源代码。

## 目录结构

- `collator.py`: 用于处理数据
- `process.py`: 数据预处理
- `seed.py`: 随机种子设置
- `sft.py`: 监督微调
- `sft.sh`: 启动脚本

## 使用方法
1. 准备数据集并放入`textvqa_bbox`目录
2. 运行`process.py`处理数据集
3. 运行`sft.sh`脚本开始微调过程

```bash
bash sft.sh
```