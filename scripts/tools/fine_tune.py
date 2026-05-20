import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    logging
)
from PIL import Image
import json
import os
from typing import Dict, List, Tuple

# 设置日志级别
logging.set_verbosity_info()

class VisionTextDataset(Dataset):
    def __init__(self, data_dir: str, processor, max_length: int = 512):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.samples = self.load_data()
    
    def load_data(self) -> List[Dict]:
        """加载并解析数据集"""
        samples = []
        # 假设数据目录下有images文件夹和annotations.json文件
        with open(os.path.join(self.data_dir, 'annotations.json'), 'r') as f:
            annotations = json.load(f)
        
        for item in annotations:
            image_path = os.path.join(self.data_dir, 'images', item['image'])
            if os.path.exists(image_path):
                samples.append({
                    'image_path': image_path,
                    'text': item['text'],
                    'target': item.get('target', '')  # 可选的目标文本
                })
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 加载并处理图像
        image = Image.open(sample['image_path']).convert('RGB')
        
        # 处理文本
        text = sample['text']
        target = sample['target']
        
        # 使用processor处理输入
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # 准备标签
        if target:
            labels = self.processor(
                text=target,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).input_ids
        else:
            labels = inputs.input_ids.clone()
        
        # 将标签中的padding token设置为-100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'pixel_values': inputs.pixel_values.squeeze(),
            'labels': labels.squeeze()
        }

def collate_fn(batch):
    """自定义批处理函数"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }


    

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "SmolVLM-256M-Instruct"
    model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    train_dataset = VisionTextDataset(
        data_dir="datasets/train",
        processor=processor
    )
    
    val_dataset = VisionTextDataset(
        data_dir="datasets/val",
        processor=processor
    )
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    trainer.save_model("./final_model")
    processor.save_pretrained("./final_model")