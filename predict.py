from ultralytics import YOLO
from PIL import Image
import torch
import os

def check_image(image_path):
    try:
        # 尝试打开图片
        img = Image.open(image_path)
        
        # 检查图片模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"警告：图片已从 {img.mode} 转换为 RGB 模式")
        
        # 打印图片信息
        print(f"图片尺寸: {img.size}")
        print(f"图片格式: {img.format}")
        
        return img
        
    except Exception as e:
        print(f"图片加载错误: {str(e)}")
        return None

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
    
    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # 遍历处理每张图片
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        print(f"\n处理图片: {image_file}")
        
        # 检查图片
        img = check_image(image_path)
        if img is None:
            print(f"图片 {image_file} 检查失败，跳过该图片")
            continue

        # 进行预测
        results = model.predict(image_path, imgsz=416)
        
        # 打印预测结果
        for result in results:
            class_id = result.probs.top1
            class_name = names_dict[class_id]  # 使用模型自带的类别名称字典
            print(f"\n预测类别: {class_name}")
            print(f"预测置信度: {result.probs.top1conf:.4f}")
            
            # 显示前3个最可能的类别及其概率
            probs_data = result.probs.data
            top_3_indices = (-probs_data).argsort()[:3]  # 使用负数来获取最大值
            print("\n前三个最可能的类别：")
            for idx in top_3_indices:
                idx = int(idx)  # 将tensor转换为整数
                prob = probs_data[idx]
                name = names_dict[idx]
                print(f"  - {name}: {prob:.4f}")

    # 方法2：对整个文件夹进行预测
    # results = model.predict("path/to/test/folder", imgsz=416, save=True) 