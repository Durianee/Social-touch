import os
import argparse
import glob
import torch
from models import load_model, load_transform
from utils import extract_frames, load_frames

# 设置设备为CPU或GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # 加载模型并设置为评估模式
    model = load_model(args.arch)
    model.to(device)
    model.eval()

    # 确保用户提供了有效的输入
    if args.frame_folder is None and args.video_file is None:
        raise ValueError("You must provide either --frame_folder or --video_file.")
    
    if args.frame_folder:
        if not os.path.exists(args.frame_folder):
            raise FileNotFoundError(f"Frame folder '{args.frame_folder}' not found.")
        print(f'Loading frames from {args.frame_folder}')
        frame_paths = sorted(glob.glob(os.path.join(args.frame_folder, '*.jpg')))
        frames = load_frames(frame_paths)
    else:
        if not os.path.exists(args.video_file):
            raise FileNotFoundError(f"Video file '{args.video_file}' not found.")
        print(f'Extracting frames from video: {args.video_file}')
        frames = extract_frames(args.video_file, args.num_segments)

    transform = load_transform()
    
    # 根据模型架构处理输入数据的形状
    if 'resnet3d50' in args.arch or 'multi_resnet3d50' in args.arch:
        # 如果是3D ResNet模型，需要将帧堆叠在一起形成一个3D输入
        input_tensor = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0).to(device)
    else:
        # 如果是2D ResNet模型，每一帧都是一个独立的输入
        input_tensor = torch.stack([transform(frame) for frame in frames], 0).to(device)

    # 提取特征向量而不是最终的分类结果 
    with torch.no_grad():
        feature_vector = model(input_tensor, return_features=True)

    # 将特征向量保存为 .pt 文件
    torch.save(feature_vector.cpu(), args.output_file)
    print(f"Feature vectors saved as '{args.output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from a video using a ResNet model")
    parser.add_argument('--arch', type=str, default='resnet3d50', choices=['resnet50', 'resnet3d50', 'multi_resnet3d50'],
                        help="Model architecture to use")
    parser.add_argument('--video_file', type=str, help="Path to the input video file")
    parser.add_argument('--frame_folder', type=str, help="Path to the folder containing frames (optional)")
    parser.add_argument('--num_segments', type=int, default=16, help="Number of frames to extract from the video")
    parser.add_argument('--output_file', type=str, default='video_features.pt', help="Path to save the output .pt file")
    args = parser.parse_args()

    main(args)
