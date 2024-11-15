from ultralytics import YOLO
from PIL import Image
import torch
import os

if __name__ == '__main__':
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载训练好的模型
    model = YOLO("yolo-sort-demo_50_epoch.pt")
    
    # 打印模型中的类别名称
    print("\n模型类别信息：")
    names_dict = model.names
    for idx, name in names_dict.items():
        print(f"类别 {idx}: {name}")
    
    # 设置图片文件夹路径
    image_folder = "image"  # 相对路径，指向当前项目目录下的image文件夹
    
    # 对整个文件夹进行预测
    results = model.predict(image_folder, imgsz=416, save=True)
    
    # 遍历处理每个预测结果
    for result in results:
        # 获取预测结果的文件名
        image_path = result.path
        image_file = os.path.basename(image_path)
        print(f"\n处理图片: {image_file}")
        
        # 获取分类结果
        class_id = result.probs.top1
        class_name = names_dict[class_id]
        print(f"\n预测类别: {class_name}")
        print(f"预测置信度: {result.probs.top1conf:.4f}")
        
        # 显示前3个最可能的类别及其概率
        probs_data = result.probs.data
        top_3_indices = (-probs_data).argsort()[:3]
        print("\n前三个最可能的类别：")
        for idx in top_3_indices:
            idx = int(idx)
            prob = probs_data[idx]
            name = names_dict[idx]
            print(f"  - {name}: {prob:.4f}")