# de_token_xhb_soh.py
# 启动命令：
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# nohup torchrun --nproc_per_node=4 --master_port=29506 de_token_1d_diff-sorp.py > 0102_raw_de_token_1d_diff-sorp.log 2>&1 &
# tmux new -d -s 0103train_test3 \; send-keys 'export CUDA_VISIBLE_DEVICES=4,5,6,7 && nohup /root/data/PierNet/.conda/env/bin/torchrun --nproc_per_node=4 --master_port=29507 de_token_1d_diff-sorp.py > 0103_raw_de_token_1d_diff-sorp.log 2>&1' C-m
# tmux kill-session -t 0103train_test3


import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import tqdm




# -----------------------------
# 数据集封装（支持 JSON 数组或 JSONL）
# -----------------------------
import jsonlines
import torch
from torch.utils.data import Dataset

class PromptNumbersDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=14000, expected_len=640, skip_invalid=True):
        """
        从 JSONL 文件中读取样本。每条样本形如：
        {
            "prompt": "已知过去两帧的空间浓度序列[[...]], 请预测...",
            "label": [0.1, 0.2, 0.3, ...]  # 长度为 expected_len 的浮点数列表
        }

        - 使用 prompt 作为模型输入（tokenize）
        - 使用 label 作为标签
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.expected_len = expected_len
        self.skip_invalid = skip_invalid
        self.samples = []

        # 读取 JSONL 文件
        with jsonlines.open(file_path, "r") as reader:
            for idx, item in enumerate(tqdm.tqdm(reader, desc="Loading dataset")):
                prompt = (item.get("prompt") or "").strip()
                label = item.get("label", [])

                # 检查 prompt 是否为空
                if not prompt:
                    if skip_invalid:
                        continue
                    else:
                        raise ValueError("Empty prompt found.")

                # 检查 label 是否有效
                if not isinstance(label, list):
                    if skip_invalid:
                        continue
                    else:
                        raise ValueError(f"Label should be a list, got {type(label)}")

                # 转换 label 为浮点数列表
                try:
                    nums = [float(x) for x in label]
                except Exception as e:
                    if skip_invalid:
                        print(f"Warning: Failed to convert label to float at line {idx}: {e}")
                        continue
                    else:
                        raise ValueError(f"Failed to convert label to float: {e}")

                # 检查 label 长度
                if self.expected_len is not None and len(nums) != self.expected_len:
                    msg = f"Found {len(nums)} numbers in label, expected {self.expected_len} at line {idx}."
                    if skip_invalid:
                        print(f"Warning: {msg}")
                        continue
                    else:
                        raise ValueError(msg)

                # 包装 prompt
                wrapped_prompt = f'''<|im_start|>system

                            你是一名PDEBench求解助手，专门负责根据用户提供的任务输入进行预测。如果用户输入的是PDEBench求解任务数据（例如1d diff-sorp数据），你将输出下一时刻的预测结果；如果是普通对话任务，则按正常对话回答。

                            ======================================================================
                            【PDEBench求解任务模式】
                            触发条件：
                            - 用户输入包含PDEBench任务数据，例如时间步的数据或场分布。

                            输出要求：
                            - 根据输入数据，输出简洁的科学计算预测结果。
                            示例：
                            用户输入: "这是1d_diff-reaction任务，请根据以下经过处理的过去1帧32个网格点数据，预测下一帧的状态。调整基数复位：值被增加了 0.05，请还原。数据如下：\n[0.79756, 0.79756, ...]"
                            强制输出: “好的，科学计算预测结果为：[预测结果]。”

                            ======================================================================
                            【普通对话模式】
                            触发条件：用户输入的内容不符合PDEBench求解任务的格式。

                            输出要求：
                            - 按普通聊天模式回答一段自然语言。
                            示例：
                            用户输入: "你是谁？"
                            输出: "你好，我是PDEBench求解助手，很高兴为你提供帮助！"

                            ======================================================================
                            通用要求：
                            - 回答必须自然流畅。
                            - 如果是PDEBench任务数据，输出简洁的预测结果；其他情况下按普通对话模式回答。

                            <|im_end|>
                            <|im_start|>user
                            ''' + prompt + '''<|im_end|>
                            <|im_start|>assistant
                            <think>

                            </think>

                            好的，科学计算预测结果为：'''

                label_tensor = torch.tensor(nums, dtype=torch.float32)
                self.samples.append((wrapped_prompt, label_tensor))

        if len(self.samples) == 0:
            raise ValueError("No valid samples found. Check JSONL file and data structure.")

        print(f"Loaded {len(self.samples)} samples with label dimension {self.expected_len}")
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
            "labels": label
        }




# -----------------------------
# 模型结构（输出维度为14）
# -----------------------------
class LMRegression128D(nn.Module):
    def __init__(self, base_model, output_dim=128):
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
    BASE_MODEL_PATH = "/root/data/models/qwen/Qwen3-1.7B"
    TEXT_MODEL_PATH = "/root/data/models/qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, trust_remote_code=True)
    # base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True).to(device)

    text_model = AutoModelForCausalLM.from_pretrained(
                                                TEXT_MODEL_PATH,
                                                trust_remote_code=True,
                                                use_cache=False  # 禁用缓存
                                            ).to(device)

    # 启用梯度检查点（trade-off：训练变慢，但内存减少）
    text_model.gradient_checkpointing_enable()

    model = LMRegression128D(text_model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    loss_fn = nn.MSELoss()

    # 数据加载
    dataset = PromptNumbersDataset(
            file_path="/root/data/PDEBench_new/1d_diff-sorp/stage2_1d_diffsorp_64res_train.jsonl",
            tokenizer=tokenizer,
            max_length=2048,
            expected_len=128,
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
                    torch.save(model.module.state_dict(), "/root/data/xhb/PiERN/1.7B/text2computation_model/0103_raw_1d_diff-sorp_text2computation_best_model.pt")


if __name__ == "__main__":
    main()
