import os
import re
import math
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# 加载数据的函数
def load_data(file_path):
    return xr.open_dataset(file_path)

# 提取年份范围（1980-1989这样的范围）
def extract_year_range_from_filename(filename):
    # 使用正则表达式提取年份范围，格式为：1980_1989
    match = re.search(r'(\d{4})_(\d{4})', filename)
    if match:
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        return start_year, end_year
    else:
        raise ValueError(f"Error extracting years from filename {filename}: invalid format")

# 提取年、月、日
def extract_year_month_day_from_time(valid_time):
    year = valid_time.dt.year
    month = valid_time.dt.month
    day = valid_time.dt.day
    return year, month, day

def visualize_monthly_data(data, variables, year, month, pressure_levels=["500", "1000"], day="07", output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)

    num_rows = len(variables)
    num_cols = len(pressure_levels)
    
    # 调整 1: 减小 figsize。8x14 对于 7行2列 比较紧凑
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 14), 
                             subplot_kw={'projection': ccrs.PlateCarree()})

    for i, var in enumerate(variables):
        for j, pressure_level in enumerate(pressure_levels):
            ax = axes[i, j]
            
            # 数据筛选
            valid_time = data['valid_time']
            # 确保这里的变量名与下方 where 逻辑一致
            y_d, m_d, d_d = extract_year_month_day_from_time(valid_time)
            
            # 修正 NameError：全部使用 y_d, m_d, d_d
            selected_data = data[var].sel(pressure_level=float(pressure_level), method='nearest').where(
                (y_d == year) & (m_d == month) & (d_d == int(day)), drop=True
            ).squeeze()

            if selected_data.size == 0:
                ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
                continue

            # 绘制地图
            im = ax.imshow(selected_data.values, 
                           extent=[78, 102, 26, 37], 
                           origin="lower", 
                           cmap='RdYlBu_r', 
                           transform=ccrs.PlateCarree())

            # 装饰瘦身
            ax.coastlines(resolution='50m', linewidth=0.5)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 7}
            gl.ylabel_style = {'size': 7}
            
            if j > 0: gl.left_labels = False
            if i < num_rows - 1: gl.bottom_labels = False

            ax.set_title(f"{var} @ {pressure_level}hPa", fontsize=9, pad=3)

            # Colorbar 紧凑化
            cb = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.04, pad=0.02, aspect=25)
            cb.ax.tick_params(labelsize=6)

    plt.suptitle(f"ERA5 Spatial Distribution ({year}-{month:02d}-{int(day):02d})", fontsize=12, y=0.96)
    
    # 调整 2: 手动控制行间距 (hspace) 和列间距 (wspace)
    # 减小 hspace 可以让行与行之间更紧凑
    plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.93, bottom=0.05, left=0.1, right=0.9)
    
    output_file = os.path.join(output_dir, f"ERA5_{year}_{month}_spatial.png")
    # 保存时 dpi 设为 300 保证清晰但文件不会过大
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

# 变量列表（按需选择）
variables = [
    "t",  # 温度对应的变量名称是 't'
    "r",  # 相对湿度
    "q",  # 具体湿度
    "crwc",  # 具体降水水分含量
    "cc",  # 云覆盖
    "u",  # 风速 U 分量
    "v"  # 风速 V 分量
]

# 文件路径列表
files = [
    'era5/P1000_P500_T_RH_SH_VcW_UcW_FCC_SRWC_1980_1989_year_d7_d14_d21_d28_t14_n37_e78_n26_e102.nc',
    'era5/P1000_P500_T_RH_SH_VcW_UcW_FCC_SRWC_1990_1999_year_d7_d14_d21_d28_t14_n37_e78_n26_e102.nc',
    'era5/P1000_P500_T_RH_SH_VcW_UcW_FCC_SRWC_2000_2013_year_d7_d14_d21_d28_t14_n37_e78_n26_e102.nc',
    'era5/P1000_P500_T_RH_SH_VcW_UcW_FCC_SRWC_2014_2026_year_d7_d14_d21_d28_t14_n37_e78_n26_e102.nc'
]

# 循环处理每个文件
for file_path in files:
    print(f"Processing file: {file_path}")
    data = load_data(file_path)

    # 提取文件名中的年份范围
    try:
        start_year, end_year = extract_year_range_from_filename(file_path)
        print(f"Extracted years: {start_year} - {end_year} from {file_path}")
    except ValueError as e:
        print(e)
        continue

    # 可视化每年每月的四天数据
    for year in range(start_year, end_year + 1):  # 处理年份范围
        for month in range(1, 13):
            print(f"Visualizing for {year}-{month}")
            visualize_monthly_data(data, variables, year, month)
