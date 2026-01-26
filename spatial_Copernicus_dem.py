import os
import rioxarray as rxr
import matplotlib.pyplot as plt
import argparse

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

    # 如果数据是 3D（例如多个时间步或层），选择第一个切片
    if dem_data.ndim == 3:
        dem_data = dem_data.isel(band=0)  # 选择第一个 band 或层，确保是二维数据

    # 创建绘图
    plt.figure(figsize=(10, 8))

    # 使用 imshow 显示 DEM 数据并获取 mappable 对象
    im = dem_data.plot.imshow(cmap='terrain', figsize=(10, 8))

    # 设置标题为子区域名称
    plt.title(f'DEM Visualization - {os.path.basename(dem_file)}', fontsize=16)
    
    # 显示颜色条，传递 mappable 对象
    plt.colorbar(im, label='Elevation (m)')

    # 保存图像为 PNG 格式
    save_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(dem_file))[0]}_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 清理当前图像，以便绘制下一个
    plt.close()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 调用可视化函数
    visualize_dem(args.dem_dir, args.output_dir)
