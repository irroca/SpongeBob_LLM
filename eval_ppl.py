import argparse
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from model import SpongeBob
from Config import LLMConfig
from train_utils import load_weights

def json_converter(obj):
    if isinstance(obj, np.generic):  # numpy.float32, numpy.int64 等
        return obj.item()
    if isinstance(obj, np.ndarray):  # 万一有 ndarray
        return obj.tolist()
    return str(obj)  # 兜底

def wrap_pretrain_text(text: str, tokenizer) -> str:
    """Wrap raw text with BOS/EOS tokens, matching PretrainDataset's encoding
    (see dataset.py) so eval inputs are aligned with training inputs."""
    return f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"


def calculate_ppl(model, tokenizer, texts, max_length=512, batch_size=4, device="cuda"):
    """
    计算模型在给定文本上的困惑度
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        texts: 文本列表
        max_length: 最大序列长度
        batch_size: 批处理大小
        device: 设备
    
    Returns:
        ppl: 平均困惑度
        individual_ppls: 每个样本的困惑度
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    individual_ppls = []
    
    with torch.no_grad():
        # 分批处理文本
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating PPL"):
            batch_texts = texts[i:i+batch_size]
            
            # 编码文本（与 PretrainDataset 保持一致，包裹 bos/eos）
            wrapped_texts = [wrap_pretrain_text(text, tokenizer) for text in batch_texts]
            encodings = tokenizer(
                wrapped_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算损失（语言模型任务，目标是预测下一个token）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            
            # 计算每个样本的损失
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())
            
            # 应用attention mask，只计算非padding位置的损失
            loss = (loss * shift_attention_mask).sum(dim=1)
            token_counts = shift_attention_mask.sum(dim=1)
            
            # 避免除零
            valid_indices = token_counts > 0
            if valid_indices.any():
                batch_losses = loss[valid_indices] / token_counts[valid_indices]
                
                # 计算每个样本的困惑度
                batch_ppls = torch.exp(batch_losses).cpu().numpy()
                individual_ppls.extend(batch_ppls)
                
                # 累计总损失和总token数
                total_loss += loss[valid_indices].sum().item()
                total_tokens += token_counts[valid_indices].sum().item()
    
    # 计算平均困惑度
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        avg_ppl = math.exp(avg_loss)
    else:
        avg_ppl = float('inf')
    
    return avg_ppl, individual_ppls

def load_dataset(file_path, text_field="text", max_samples=None):
    """
    加载JSONL格式的数据集
    
    Args:
        file_path: 数据文件路径
        text_field: 包含文本的字段名
        max_samples: 最大样本数（用于测试）
    
    Returns:
        texts: 文本列表
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                if text_field in data:
                    texts.append(data[text_field])
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue
    
    print(f"Loaded {len(texts)} texts from {file_path}")
    return texts

def analyze_ppl_distribution(individual_ppls):
    """
    分析困惑度分布
    
    Args:
        individual_ppls: 每个样本的困惑度列表
    
    Returns:
        stats: 统计信息字典
    """
    if not individual_ppls:
        return {}
    
    ppl_array = np.array(individual_ppls)
    
    stats = {
        'mean': np.mean(ppl_array),
        'median': np.median(ppl_array),
        'std': np.std(ppl_array),
        'min': np.min(ppl_array),
        'max': np.max(ppl_array),
        'percentile_25': np.percentile(ppl_array, 25),
        'percentile_75': np.percentile(ppl_array, 75),
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate model perplexity on a dataset")
    parser.add_argument('--model_path', type=str, required=True, 
                       help="Path to the trained model checkpoint")
    parser.add_argument('--dataset_path', type=str, required=True,
                       help="Path to the JSONL dataset file")
    parser.add_argument('--text_field', type=str, default="text",
                       help="Field name containing the text in JSONL")
    parser.add_argument('--max_seq_len', type=int, default=512,
                       help="Maximum sequence length for evaluation")
    parser.add_argument('--batch_size', type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument('--max_samples', type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing)")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for evaluation")
    parser.add_argument('--output_file', default = 'eval_result.json',type=str, 
                       help="File to save evaluation results")
    
    args = parser.parse_args()
    
    # 加载tokenizer和模型
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained('./spongebob_tokenizer')
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型配置和权重
    model = SpongeBob(LLMConfig(max_seq_len=args.max_seq_len, vocab_size=tokenizer.vocab_size))
    load_weights(args.model_path, model, args.device, strict=False)
    
    model = model.to(args.device)
    model.eval()
    
    print(f"Model loaded from {args.model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # 加载数据集
    print("Loading dataset...")
    texts = load_dataset(args.dataset_path, args.text_field, args.max_samples)
    
    if not texts:
        print("No texts found in the dataset!")
        return
    
    # 计算困惑度
    print("Calculating perplexity...")
    avg_ppl, individual_ppls = calculate_ppl(
        model, tokenizer, texts, 
        max_length=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 分析困惑度分布
    stats = analyze_ppl_distribution(individual_ppls)
    
    # 打印结果
    print("\n" + "="*50)
    print("PERPLEXITY EVALUATION RESULTS")
    print("="*50)
    print(f"Dataset: {args.dataset_path}")
    print(f"Number of samples: {len(texts)}")
    print(f"Average Perplexity: {avg_ppl:.2f}")
    
    if stats:
        print(f"Perplexity Statistics:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Std: {stats['std']:.2f}")
        print(f"  Min: {stats['min']:.2f}")
        print(f"  Max: {stats['max']:.2f}")
        print(f"  25th percentile: {stats['percentile_25']:.2f}")
        print(f"  75th percentile: {stats['percentile_75']:.2f}")
    
    # 保存结果到文件
    if args.output_file:
        results = {
            'dataset_path': args.dataset_path,
            'model_path': args.model_path,
            'num_samples': len(texts),
            'average_perplexity': avg_ppl,
            'perplexity_statistics': stats,
            'individual_perplexities': individual_ppls
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=json_converter)
        
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()