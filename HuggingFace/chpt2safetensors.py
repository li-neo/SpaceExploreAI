import os
import argparse
import sys
import torch
from safetensors.torch import save_file


def print_help():
    """打印帮助信息和使用示例"""
    help_text = """
chpt2safetensors.py - 将PyTorch模型(.pt)转换为safetensors格式(.safetensors)

使用方法:
    1. 单文件转换 (旧格式，推荐):
       python chpt2safetensors.py model.pt [output.safetensors]
    
    2. 单文件转换 (新格式):
       python chpt2safetensors.py --input model.pt [--output output.safetensors]
    
    3. 批量转换目录中的所有.pt文件:
       python chpt2safetensors.py --dir model_directory

参数:
    model.pt                 - 输入模型文件路径
    output.safetensors       - (可选) 输出模型文件路径，默认与输入文件同名但后缀为.safetensors
    --input/-i model.pt      - 输入模型文件路径
    --output/-o output.file  - 输出模型文件路径
    --dir/-d directory       - 要批量转换的目录
    --help/-h                - 显示此帮助信息

示例:
    python chpt2safetensors.py models/model_best.pt
    python chpt2safetensors.py --input models/model_best.pt --output models/model_best.safetensors
    python chpt2safetensors.py --dir models
    """
    print(help_text)


def convert_pt_to_safetensors(pt_path, output_path=None):
    """
    将PyTorch模型文件(.pt)转换为safetensors格式(.safetensors)
    
    参数:
        pt_path: PyTorch模型文件路径
        output_path: 输出的safetensors文件路径，如果为None，则使用相同的文件名但后缀改为.safetensors
    
    返回:
        输出文件的路径
    """
    # 加载PyTorch模型
    print(f"正在加载模型 {pt_path}...")
    checkpoint = torch.load(pt_path, map_location='cpu')
    
    # 如果是包含多个组件的字典，只保留模型权重
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        # 保存元数据，用于创建更完整的safetensors文件
        metadata = {}
        if "vocab_size" in checkpoint:
            metadata["vocab_size"] = str(checkpoint["vocab_size"])
        if "hidden_size" in checkpoint:
            metadata["hidden_size"] = str(checkpoint["hidden_size"])
        if "prediction_type" in checkpoint:
            metadata["prediction_type"] = checkpoint["prediction_type"]
        print("使用模型状态字典作为转换源")
    else:
        # 直接使用加载的对象作为状态字典
        state_dict = checkpoint
        metadata = {}
        print("使用完整检查点作为转换源")
    
    # 处理不支持的张量类型，确保张量是连续的
    print("处理张量...")
    filtered_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            # 将复数张量转换为浮点数张量
            if v.dtype == torch.complex64 or v.dtype == torch.complex128:
                print(f"  - 警告: 将复数张量 '{k}' 转换为实数部分")
                tensor = v.real.cpu()
            else:
                tensor = v.cpu()
                
            # 确保张量是连续的
            if not tensor.is_contiguous():
                print(f"  - 警告: 将非连续张量 '{k}' 转换为连续")
                tensor = tensor.contiguous()
                
            filtered_dict[k] = tensor
        else:
            print(f"  - 跳过非张量项: '{k}'")
    
    # 设置输出路径
    if output_path is None:
        base_path, _ = os.path.splitext(pt_path)
        output_path = f"{base_path}.safetensors"
    
    # 保存为safetensors格式
    print(f"正在保存模型到 {output_path}...")
    save_file(filtered_dict, output_path, metadata=metadata)
    print(f"转换完成：{pt_path} -> {output_path}")
    
    return output_path


def main():
    # 打印帮助信息
    if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        print_help()
        return
    
    # 检查是否使用了旧版命令格式（直接提供文件路径作为位置参数）
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and sys.argv[1].endswith('.pt'):
        # 旧版命令格式的处理
        input_path = sys.argv[1]
        output_path = None
        if len(sys.argv) > 2 and not sys.argv[2].startswith('-'):
            output_path = sys.argv[2]
        
        print("检测到旧版命令格式，直接转换文件")
        convert_pt_to_safetensors(input_path, output_path)
        return
    
    # 新版命令行参数解析
    parser = argparse.ArgumentParser(description="将PyTorch模型(.pt)转换为safetensors格式")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", type=str, help="输入模型路径(.pt)")
    group.add_argument("--dir", "-d", type=str, help="要批量转换的目录路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出模型路径(.safetensors)，仅在单文件转换时有效")
    
    args = parser.parse_args()
    
    if args.dir:
        # 批量转换目录中的所有.pt文件
        print(f"开始批量转换目录 {args.dir} 中的文件...")
        converted_files = []
        for filename in os.listdir(args.dir):
            if filename.endswith(".pt") and not os.path.exists(os.path.join(args.dir, filename.replace('.pt', '.safetensors'))):
                input_path = os.path.join(args.dir, filename)
                output_path = None  # 使用默认输出路径
                converted_files.append(convert_pt_to_safetensors(input_path, output_path))
            elif filename.endswith(".pt"):
                print(f"跳过 {filename}，已存在对应的 safetensors 文件")
        
        print(f"批量转换完成，共转换 {len(converted_files)} 个文件:")
        for file in converted_files:
            print(f"  - {file}")
    else:
        # 转换单个文件
        if not args.input:
            parser.error("请提供 --input 参数指定输入文件路径")
        convert_pt_to_safetensors(args.input, args.output)


if __name__ == "__main__":
    main()
