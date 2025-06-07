import argparse
import json
import time
import requests

def generate_text(url, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.9, num_return_sequences=1):
    """
    调用API生成文本
    
    Args:
        url: API地址，例如 "http://localhost:8000/generate"
        prompt: 提示文本
        max_length: 最大生成长度
        temperature: 温度参数，控制生成的随机性
        top_k: Top-K采样参数
        top_p: Top-P采样参数
        num_return_sequences: 返回的序列数量
        
    Returns:
        dict: API响应
    """
    # 构建请求数据
    data = {
        "prompt": prompt,
        "max_length": max_length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_return_sequences": num_return_sequences
    }
    
    # 发送请求
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # 如果响应状态码不是200，抛出异常
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="中文语言模型API客户端")
    
    parser.add_argument("--host", type=str, default="localhost", help="API服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="API服务端口号")
    parser.add_argument("--prompt", type=str, required=True, help="生成的提示文本")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K采样参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P采样参数")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="返回的序列数量")
    
    args = parser.parse_args()
    
    # 构建API URL
    url = f"http://{args.host}:{args.port}/generate"
    
    print(f"正在连接API服务: {url}")
    print(f"提示文本: {args.prompt}")
    print("正在生成...")
    
    # 记录开始时间
    start_time = time.time()
    
    # 调用API
    response = generate_text(
        url=url,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences
    )
    
    # 计算总时间
    total_time = time.time() - start_time
    
    # 打印结果
    if response:
        print("\n生成结果:")
        if isinstance(response["generated_text"], list):
            for i, text in enumerate(response["generated_text"]):
                print(f"\n序列 {i+1}:")
                print(text)
        else:
            print(response["generated_text"])
        
        print(f"\n生成时间: {response['generation_time']:.2f}秒")
        print(f"总时间(包括网络延迟): {total_time:.2f}秒")
    else:
        print("生成失败，请检查API服务是否正常运行")

if __name__ == "__main__":
    main()