import os
import rioxarray as rxr
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def parse_args():
    # 创建 argparse 对象
    parser = argparse.ArgumentParser(description="Visualize DEM files and save the outputs.")
    
    # 添加输入和输出目录的命令行参数
    parser.add_argument('--dem_dir', type=str, required=True, help="Path to the input DEM file (not directory)")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory for visualizations")
    
    # 解析命令行参数
    args = parser.parse_args()
    return args

def visualize_dem(dem_file, output_dir):
    # 如果输出目录不存在，自动创建
    os.makedirs(output_dir, exist_ok=True)

    # 加载 DEM 数据
    dem_data = rxr.open_rasterio(dem_file, masked=True)

    # 如果数据是 3D，选择第一个切片
    if dem_data.ndim == 3:
        dem_data = dem_data.isel(band=0)

    # 创建绘图容器
    fig, ax = plt.subplots(figsize=(10, 8))

    # 核心修改点 1: 使用 add_colorbar=True (默认) 并通过 cbar_kwargs 定制，避免手动创建
    # 核心修改点 2: 设置 xarray 绘图到指定的 ax 上
    dem_data.plot.imshow(
        ax=ax,
        cmap='terrain', 
        cbar_kwargs={'label': 'Elevation (m)'}
    )

    # 核心修改点 3: 保持原始地理比例 (防止Y轴拉伸)
    # 'equal' 确保横纵坐标单位长度相等
    ax.set_aspect('equal')

    # 设置标题和标签
    ax.set_title(f'DEM Visualization - {os.path.basename(dem_file)}', fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 保存图像
    save_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(dem_file))[0]}_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 关闭画布释放内存
    plt.close(fig)

def process_files(dem_files, output_dir):
    # 使用 tqdm 显示进度条
    for dem_file in tqdm(dem_files, desc="Processing DEM files", unit="file"):
        print(f"Processing file: {os.path.basename(dem_file)}")
        visualize_dem(dem_file, output_dir)

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 获取文件夹下的所有 DEM 文件
    dem_files = [args.dem_dir]

    # 调用进度条处理文件
    process_files(dem_files, args.output_dir)
