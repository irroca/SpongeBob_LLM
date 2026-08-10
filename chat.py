import argparse
import random
import time
import numpy as np
import torch
import warnings
import os
import sys
from transformers import AutoTokenizer
from model import SpongeBob
from Config import LLMConfig
from train_utils import load_weights

# 彩色输出工具类
class Colors:
    """ANSI颜色代码"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def colored_text(text, color):
    """返回带颜色的文本"""
    return f"{color}{text}{Colors.END}"

def init_model(args):
    """初始化模型和分词器"""
    try:
        tokenizer = AutoTokenizer.from_pretrained('./spongebob_tokenizer')
        
        # 根据模型模式选择对应的checkpoint
        model_files = {
            0: 'pretrain.pth',  # 预训练模型
            1: 'SFT.pth',       # SFT模型
            2: 'distill.pth',   # 蒸馏模型
            3: 'dpo.pth',       # DPO模型
        }
        
        if args.model_mode not in model_files:
            print(colored_text(f"错误: 不支持的模型模式 {args.model_mode}", Colors.RED))
            print(colored_text("请使用: 0(预训练), 1(SFT), 2(蒸馏), 3(DPO)", Colors.YELLOW))
            sys.exit(1)
            
        ckp_path = os.path.join(args.save_dir, model_files[args.model_mode])
        
        if not os.path.exists(ckp_path):
            # 尝试加载checkpoint文件
            checkpoint_files = {
                0: ['pretrain_final.pth', 'latest_checkpoint.pth'],
                1: ['sft_final.pth', 'SFT.pth', 'latest_checkpoint.pth'],
                2: ['distill_final.pth', 'latest_checkpoint.pth'],
                3: ['dpo_final.pth', 'latest_checkpoint.pth'],
            }
            
            for candidate in checkpoint_files[args.model_mode]:
                candidate_path = os.path.join(args.save_dir, candidate)
                if os.path.exists(candidate_path):
                    ckp_path = candidate_path
                    print(colored_text(f"使用备用模型文件: {candidate}", Colors.YELLOW))
                    break
            else:
                print(colored_text(f"错误: 在 {args.save_dir} 中找不到模型文件", Colors.RED))
                print(colored_text("请确保已经训练了相应的模型", Colors.YELLOW))
                sys.exit(1)
        
        print(colored_text(f"加载模型: {ckp_path}", Colors.GREEN))
        
        # 初始化模型配置
        model_config = LLMConfig(
            max_seq_len=args.max_seq_len,
            vocab_size=tokenizer.vocab_size
        )
        
        model = SpongeBob(model_config)
        load_weights(ckp_path, model, args.device, strict=False)
        model = model.eval().to(args.device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(colored_text(f"模型参数量: {total_params / 1e6:.2f}M", Colors.CYAN))
        
        return model, tokenizer
        
    except Exception as e:
        print(colored_text(f"模型初始化失败: {e}", Colors.RED))
        sys.exit(1)

def format_conversation(messages, user_color=Colors.GREEN, assistant_color=Colors.BLUE):
    """格式化对话历史"""
    formatted = []
    for msg in messages:
        if msg["role"] == "user":
            formatted.append(colored_text(f"用户: {msg['content']}", user_color))
        else:
            formatted.append(colored_text(f"助手: {msg['content']}", assistant_color))
    return "\n".join(formatted)

def streaming_generation(model, tokenizer, prompt, args):
    """流式生成响应"""
    try:
        with torch.no_grad():
            # 编码输入
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
            
            # 生成响应
            outputs = model.generate(
                input_ids,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=args.repetition_penalty
            )
            
            # 流式输出
            full_response = ""
            print(colored_text("助手: ", Colors.BLUE), end='', flush=True)
            
            for token_batch in outputs:
                new_text = tokenizer.decode(token_batch[0].tolist(), skip_special_tokens=True)
                
                # 只输出新增的部分
                if new_text.startswith(full_response):
                    new_content = new_text[len(full_response):]
                    print(new_content, end='', flush=True)
                    full_response = new_text
                else:
                    # 处理解码不一致的情况
                    print(new_text, end='', flush=True)
                    full_response += new_text
                
                # 检查是否生成结束标记
                if tokenizer.eos_token in new_text:
                    break
            
            print()  # 换行
            return full_response.strip()
            
    except Exception as e:
        print(colored_text(f"\n生成过程中出错: {e}", Colors.RED))
        return ""

def main():
    parser = argparse.ArgumentParser(description="SpongeBob模型交互式对话")
    
    # 模型参数
    parser.add_argument('--save_dir', default='results', type=str, 
                       help='模型保存目录')
    parser.add_argument('--model_mode', default=1, type=int, choices=[0, 1, 2, 3],
                       help='模型模式: 0-预训练, 1-SFT聊天, 2-蒸馏, 3-DPO')
    
    # 生成参数
    parser.add_argument('--temperature', default=0.7, type=float,
                       help='生成温度，越高越随机')
    parser.add_argument('--top_p', default=0.9, type=float,
                       help='核采样概率')
    parser.add_argument('--repetition_penalty', default=1.2, type=float,
                       help='重复惩罚系数')
    parser.add_argument('--max_new_tokens', default=512, type=int,
                       help='最大生成token数')
    parser.add_argument('--max_seq_len', default=2048, type=int,
                       help='最大序列长度')
    
    # 对话参数
    parser.add_argument('--history_cnt', default=0, type=int,
                       help='保留的历史对话轮数')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                       type=str, help='运行设备')
    parser.add_argument('--show_prompt', action='store_true',
                       help='显示实际发送给模型的prompt')
    
    args = parser.parse_args()
    
    # 打印配置信息
    print(colored_text("=== SpongeBob模型对话系统 ===", Colors.BOLD + Colors.CYAN))
    print(colored_text(f"设备: {args.device}", Colors.YELLOW))
    print(colored_text(f"模型模式: {['预训练', 'SFT聊天', '蒸馏', 'DPO'][args.model_mode]}", Colors.YELLOW))
    print(colored_text("输入 'quit' 或 'exit' 退出对话", Colors.YELLOW))
    print(colored_text("=" * 40, Colors.CYAN))
    
    # 初始化模型
    model, tokenizer = init_model(args)
    
    # 对话历史
    messages = []
    conversation_count = 0
    
    while True:
        try:
            # 获取用户输入
            print(colored_text("\n用户: ", Colors.GREEN), end='')
            user_input = input().strip()
            
            # 退出条件
            if user_input.lower() in ['quit', 'exit', '退出']:
                print(colored_text("感谢使用SpongeBot对话系统！", Colors.CYAN))
                break
            if not user_input:
                continue
                
            conversation_count += 1
            print(colored_text(f"对话轮次: {conversation_count}", Colors.YELLOW))
            
            messages = messages[-(args.history_cnt) * 2:]  if args.history_cnt != 0 else []# 保留最近N轮对话
            # 更新对话历史
            messages.append({"role": "user", "content": user_input})
            
            # 构建prompt
            if args.model_mode == 0:
                # 预训练模型使用简单格式
                prompt = tokenizer.bos_token + user_input
            else:
                # SFT和蒸馏模型使用对话格式
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            
            # 截断到最大长度（按 token 计）
            max_prompt_tokens = max(8, args.max_seq_len - args.max_new_tokens)
            prompt_ids = tokenizer.encode(prompt)
            if len(prompt_ids) > max_prompt_tokens:
                prompt_ids = prompt_ids[-max_prompt_tokens:]
                prompt = tokenizer.decode(prompt_ids)
            
            if args.show_prompt:
                print(colored_text(f"实际Prompt: {prompt}", Colors.MAGENTA))
            
            # 生成响应
            start_time = time.time()
            response = streaming_generation(model, tokenizer, prompt, args)
            generation_time = time.time() - start_time
            
            if response:
                messages.append({"role": "assistant", "content": response})
                print(colored_text(f"(生成耗时: {generation_time:.2f}s)", Colors.YELLOW))
            
        except KeyboardInterrupt:
            print(colored_text("\n\n中断对话，感谢使用！", Colors.CYAN))
            break
        except Exception as e:
            print(colored_text(f"对话过程中出错: {e}", Colors.RED))
            # 清空当前对话历史，避免错误累积
            messages = messages[:-1] if messages else []

if __name__ == "__main__":
    # 检查colorama是否可用，如果不可用则使用基本的ANSI颜色
    try:
        from colorama import init, Fore, Back, Style
        init()  # 初始化colorama（Windows需要）
        # 如果colorama可用，可以替换Colors类中的定义
        Colors.RED = Fore.RED
        Colors.GREEN = Fore.GREEN
        Colors.YELLOW = Fore.YELLOW
        Colors.BLUE = Fore.BLUE
        Colors.MAGENTA = Fore.MAGENTA
        Colors.CYAN = Fore.CYAN
        Colors.WHITE = Fore.WHITE
        Colors.BOLD = Style.BRIGHT
        Colors.END = Style.RESET_ALL
    except ImportError:
        print("提示: 安装colorama可以获得更好的彩色输出支持: pip install colorama")
    
    main()