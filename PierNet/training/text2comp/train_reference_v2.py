# de_token_xhb_soh.py
# 启动命令：
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# nohup torchrun --nproc_per_node=8 --master_port=29505 de_token_diff-sorp.py > 1102_de_token_diff-sorp.log 2>&1 &


import json
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os




# -----------------------------
# 数据集封装（支持 JSON 数组或 JSONL）
# -----------------------------
import json
import torch
from torch.utils.data import Dataset


class PromptNumbersDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024, expected_len=2048, skip_invalid=True):
        """
        从 JSON 文件中读取样本。每条样本形如：
        { "prompt": "已知过去两帧的空间浓度序列[[...]], 请预测..." }

        - 直接用 prompt 作为模型输入（tokenize）
        - 从 prompt 里提取所有数字（按出现顺序），作为 labels（长度应为 expected_len，默认 2048）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.expected_len = expected_len
        self.skip_invalid = skip_invalid
        self.samples = []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        num_pat = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')

        for item in data:
            prompt = (item.get("prompt") or "").strip()
            if not prompt:
                if skip_invalid:
                    continue
                else:
                    raise ValueError("Empty prompt found.")

            # 提取数字：顺序即为 label 顺序
            nums = num_pat.findall(prompt)
            # 转成 float
            try:
                nums = [float(x) for x in nums]
            except Exception:
                if skip_invalid:
                    continue
                else:
                    raise

            if self.expected_len is not None and len(nums) != self.expected_len:
                # 数量不对：要么丢弃，要么报错
                msg = f"Found {len(nums)} numbers, expected {self.expected_len}."
                if skip_invalid:
                    # 可打印告警或直接跳过
                    # print(f"[WARN] {msg}")
                    continue
                else:
                    raise ValueError(msg)

            prompt = '''
                    <|im_start|>system

            你是一个时序浓度预测助手。
            当用户消息同时包含关键词【浓度、序列、时间步、预测】时：
            - 只输出一句中文描述，并以占位符 <ans> 结尾。
            - 全文禁止出现任何数字字符（0-9 和小数点）。

            正例：
            - “好的，下一时间步的浓度分布为<ans>”

            其他情况按正常聊天回答。

            <|im_end|>
            <|im_start|>user
                ''' + prompt + '''<|im_end|>
                            <|im_start|>assistant
                            好的，下一时间步的浓度分布为'''

            input_str = prompt
            label_tensor = torch.tensor(nums, dtype=torch.float32)
            self.samples.append((input_str, label_tensor))

        if len(self.samples) == 0:
            raise ValueError("No valid samples found. Check JSON and number extraction.")

        # 方便外部读取 label 维度
        self.label_dim = self.expected_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_str, label = self.samples[idx]
        encoded = self.tokenizer(
            input_str,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": label  # shape: [2048]
        }



# -----------------------------
# 模型结构（输出维度为14）
# -----------------------------
class LMRegression2048D(nn.Module):
    def __init__(self, base_model, output_dim=2048):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.model.embed_tokens.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        masked = last_hidden * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return self.head(pooled)  # [B, 14]


# -----------------------------
# 训练主程序
# -----------------------------
def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    BASE_MODEL_PATH = "/data/models/Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True).to(device)

    model = LMRegression2048D(base_model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    loss_fn = nn.MSELoss()

    # 数据加载
    dataset = PromptNumbersDataset(
            file_path="/data/moe/sft_pde/train_data/processed_train.json",
            tokenizer=tokenizer,
            max_length=1024,
            expected_len=2048,
            skip_invalid=True
        )
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

    best_loss = float('inf')

    for epoch in range(1000000):  # 按 epoch 来控制
        sampler.set_epoch(epoch)  # 确保每个 epoch shuffle 一致
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_y = batch["labels"].to(device)

            y_pred = model(input_ids, attention_mask)
            loss = loss_fn(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0 and local_rank == 0:
                print(f"[epoch {epoch} step {step}] loss: {loss.item():.8f}")
                print("input :", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))
                print("pred  :", [f"{v:.8f}" for v in y_pred[0].tolist()])
                print("truth :", [f"{v:.8f}" for v in batch_y[0].tolist()])
                print()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    print(f"New best loss ({loss.item()}), saving model...")
                    torch.save(model.module.state_dict(), "/data/PiERN/PDEbench/model/diff-sorp/1102_diff-sorp_text2computation_best_model.pt")


if __name__ == "__main__":
    main()
