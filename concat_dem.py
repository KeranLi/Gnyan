import os
import argparse
from tqdm import tqdm
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import matplotlib.pyplot as plt
import gc
import dask

def parse_args():
    parser = argparse.ArgumentParser(description="Memory-efficient DEM merge.")
    parser.add_argument('--dem_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def mosaic_and_visualize(dem_dir, output_dir):
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 10})
    os.makedirs(output_dir, exist_ok=True)

    dem_files = [os.path.join(dem_dir, f) for f in os.listdir(dem_dir) 
                 if f.lower().endswith(('.tif', '.tiff', '.dem'))]
    
    if not dem_files:
        print("No files found.")
        return

    print(f"Found {len(dem_files)} files. Opening with lazy loading...")
    elements = []
    for f in tqdm(dem_files, desc="Indexing files"):
        # 核心修改 1: 添加 chunks，开启 Dask 延迟加载，不立即占用大内存
        data = rxr.open_rasterio(f, masked=True, chunks={'x': 1024, 'y': 1024})
        if data.ndim == 3:
            data = data.isel(band=0)
        elements.append(data)

    print("Merging files (Lazy loading with Dask)...")
    # 核心修改 2: 合并
    merged = merge_arrays(elements)
    
    # 释放掉不再需要的列表
    del elements
    gc.collect()

    print("Generating plot-friendly data...")
    # 核心修改 3: 如果图太大，我们可以先降低绘图分辨率（降低绘图时的内存消耗）
    # 如果还是 Killed，可以将下面的 1 改成 2 或更高（代表每隔几个像素抽样一个）
    if merged.x.size * merged.y.size > 4000000:
        # 自动计算缩放倍数，目标是降到 2000 像素左右
        scale = max(merged.x.size // 2000, merged.y.size // 2000)
        if scale > 1:
            print(f"Downsampling data by factor of {scale} for plotting...")
            plot_data = merged.coarsen(x=scale, y=scale, boundary='trim').mean()
        else:
            plot_data = merged
    else:
        plot_data = merged

    print("Computing final image...")
    plot_data = plot_data.compute()

    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = plot_data.plot.imshow(
        ax=ax,
        cmap='terrain',
        cbar_kwargs={
            'label': 'Elevation (m)',
            'shrink': 0.35,
            'pad': 0.05,
            'aspect': 20
        }
    )

    ax.set_aspect('equal')
    ax.set_title("DEM Mosaic Visualization")
    
    save_path = os.path.join(output_dir, 'dem_mosaic_visualization.png')
    # dpi 设置为 300 可能会占用较多内存，如果继续被 Killed，可以尝试降为 150
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    args = parse_args()
    mosaic_and_visualize(args.dem_dir, args.output_dir)