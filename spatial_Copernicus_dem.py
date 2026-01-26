import os
import rioxarray as rxr
import matplotlib.pyplot as plt

# 设置 DEM 数据存放目录
dem_dir = '/root/autodl-tmp/gnyan/dem'  # 修改为正确的路径

# 获取目录下的所有子区域文件
dem_files = [f for f in os.listdir(dem_dir) if f.endswith('.tif')]

# 对每个子区域进行处理和保存可视化图像
for dem_file in dem_files:
    # 构建子区域的完整文件路径
    dem_path = os.path.join(dem_dir, dem_file)
    
    # 加载 DEM 数据
    dem_data = rxr.open_rasterio(dem_path, masked=True)

    # 创建绘图
    plt.figure(figsize=(10, 8))
    
    # 绘制 DEM 数据并获取 mappable 对象
    mappable = dem_data.plot(cmap='terrain', figsize=(10, 8))
    
    # 设置标题为子区域名称
    plt.title(f'DEM Visualization - {dem_file}', fontsize=16)
    
    # 显示颜色条，传递 mappable 对象
    plt.colorbar(mappable, label='Elevation (m)')

    # 保存图像为 PNG 格式
    save_path = os.path.join(dem_dir, f'{os.path.splitext(dem_file)[0]}_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 清理当前图像，以便绘制下一个
    plt.close()
