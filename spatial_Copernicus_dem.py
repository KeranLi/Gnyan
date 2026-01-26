import os
import argparse
from tqdm import tqdm
import rioxarray as rxr
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DEM files and save the outputs.")
    parser.add_argument('--dem_dir', type=str, required=True, help="Path to the input DEM file")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()
    return args

def visualize_dem(dem_file, output_dir):
    plt.rcParams['axes.linewidth'] = 0.75
    # --- 全局字体设置 ---
    # 这里统一设置所有字体大小，可以根据需要微调这些数值
    plt.rcParams.update({
        'font.size': 7,          # 基础字体大小
        'axes.titlesize': 7,     # 标题大小
        'axes.labelsize': 7,      # 坐标轴标签大小
        'xtick.labelsize': 6,     # x轴刻度大小
        'ytick.labelsize': 6,     # y轴刻度大小
        'legend.fontsize': 6,     # 图例/Colorbar标签大小
    })

    os.makedirs(output_dir, exist_ok=True)
    dem_data = rxr.open_rasterio(dem_file, masked=True)

    if dem_data.ndim == 3:
        dem_data = dem_data.isel(band=0)

    # 稍微调大画布比例，但保持较小的字体
    fig, ax = plt.subplots(figsize=(6, 5))

    # 绘制 DEM
    # 注意：label 的字体会自动应用 rcParams 的设置
    im = dem_data.plot.imshow(
        ax=ax,
        cmap='terrain', 
        cbar_kwargs={
            'label': 'Elevation (m)',
            'shrink': 0.25,    # 建议设为 0.5-0.6 比较协调
            'pad': 0.03,      # 增加一点间距，防止字体挤在一起
            'aspect': 15      # 让 bar 变细一点，看起来更精致
        }
    )

    # 保持地理比例
    ax.set_aspect('equal')

    # 设置标题和轴标签
    ax.set_title(f'DEM@{os.path.basename(dem_file)}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 保存图像
    save_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(dem_file))[0]}_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def process_files(dem_files, output_dir):
    for dem_file in tqdm(dem_files, desc="Processing DEM files", unit="file"):
        visualize_dem(dem_file, output_dir)

if __name__ == "__main__":
    args = parse_args()
    dem_files = [args.dem_dir]
    process_files(dem_files, args.output_dir)