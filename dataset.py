import json
from torch.utils.data import Dataset, DataLoader
import torch


def assistant_loss_mask(input_ids, bos_id, eos_id, max_length, pad_token_id=0):
    """Mark tokens inside <s>assistant\\n ... </s>\\n spans as 1.

    Rules:
      - Mark from the first assistant content token through the end of the
        </s>\\n span, inclusive.
      - Never mark pad_token_id positions.
      - If no eos is found (truncated), mark content start .. last non-pad
        index only.
    """
    loss_mask = [0] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i : i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            found_eos = False
            while end < len(input_ids):
                if input_ids[end : end + len(eos_id)] == eos_id:
                    found_eos = True
                    break
                end += 1
            if found_eos:
                span_end = min(end + len(eos_id), max_length)
            else:
                span_end = min(end, max_length)
                while span_end > start and input_ids[span_end - 1] == pad_token_id:
                    span_end -= 1
            for j in range(start, span_end):
                if input_ids[j] != pad_token_id:
                    loss_mask[j] = 1
            i = end + len(eos_id) if found_eos else len(input_ids)
        else:
            i += 1
    return loss_mask


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        预训练数据集初始化
        
        参数:
          data_path: 数据文件路径，每行是一个JSON格式的样本
          tokenizer: 分词器，用于将文本转为 token ID
          max_length: 每个样本的最大长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载数据，返回一个样本列表，每个样本为字典格式
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        """
        从文件中逐行读取数据，并解析为 JSON 对象
        
        参数:
          path: 数据文件路径
        
        返回:
          samples: 样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        # 返回样本总数
        return len(self.samples)
    
    def __getitem__(self, index):
        # 根据索引获取单个样本数据
        sample = self.samples[index]
        # 构建输入文本，加上起始符和结束符
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        # 利用 tokenizer 将文本编码为 token IDs，固定最大长度，进行 padding 和截断
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # 获取输入 token IDs，去除多余的维度
        input_ids = encoding.input_ids.squeeze()
        # 构造损失掩码，标记非填充位置
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # X 为输入序列（去掉最后一个 token），Y 为目标序列（去掉第一个 token）
        X = input_ids[:-1].clone().detach().long()
        Y = input_ids[1:].clone().detach().long()
        loss_mask = loss_mask[1:].clone().detach().long()
        return X, Y, loss_mask

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        微调数据集初始化
        
        参数:
          jsonl_path: 数据文件路径，每行是一个 JSON 格式的对话样本
          tokenizer: 分词器，用于将文本转为 token ID
          max_length: 每个样本的最大长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载数据，返回样本列表
        self.samples = self.load_data(jsonl_path)
        # 定义开始和结束标记的 token ID（不添加特殊 token）
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        # 返回样本总数
        return len(self.samples)
    
    def load_data(self, path):
        """
        从文件中逐行读取数据，并解析为 JSON 对象
        
        参数:
          path: 数据文件路径
        
        返回:
          samples: 样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def _create_chat_prompt(self, conversations):
        """
        构建符合 ChatML 格式的对话提示
        
        参数:
          conversations: 对话轮次列表，每个元素为包含 'content' 的字典
        
        返回:
          prompt: 拼接后的对话文本
        """
        messages = []
        for i, turn in enumerate(conversations):
            # 偶数轮为用户，奇数轮为助手
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        # 使用分词器提供的模板方法构建对话提示
        # print(self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=False
        # ))
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def _generate_loss_mask(self, input_ids):
        return assistant_loss_mask(
            input_ids, self.bos_id, self.eos_id, self.max_length, self.tokenizer.pad_token_id
        )
    
    def __getitem__(self, index):
        # 获取对应索引的样本
        sample = self.samples[index]
        # 利用对话轮次构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        # 对对话提示进行编码，并限制最大长度
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 若不足最大长度则补齐 pad_token_id
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 根据输入 token IDs 生成动态的损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据：X 为输入序列（去掉最后一个 token），Y 为目标序列（去掉第一个 token）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask

class PreferenceDataset(Dataset):
    """DPO preference pairs: prompt / chosen / rejected."""

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer("<s>assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer("</s>\n", add_special_tokens=False).input_ids

    def load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        return samples

    def __len__(self):
        return len(self.samples)

    def _encode_response(self, prompt: str, answer: str):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = self.tokenizer(text).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = assistant_loss_mask(
            input_ids, self.bos_id, self.eos_id, self.max_length, self.tokenizer.pad_token_id
        )
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        cX, cY, cM = self._encode_response(sample["prompt"], sample["chosen"])
        rX, rY, rM = self._encode_response(sample["prompt"], sample["rejected"])
        return cX, cY, cM, rX, rY, rM

