import os
import re
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 加载数据的函数
def load_data(file_path):
    """
    加载ERA5 NetCDF数据文件
    """
    return xr.open_dataset(file_path)

# 提取年份范围（1980-1989这样的范围）
def extract_year_range_from_filename(filename):
    """
    从文件名中提取年份范围
    """
    match = re.search(r'(\d{4})_(\d{4})', filename)
    if match:
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        return start_year, end_year
    else:
        raise ValueError(f"Error extracting years from filename {filename}: invalid format")

# 提取年、月、日
def extract_year_month_day_from_time(valid_time):
    """
    提取时间中的年、月、日
    """
    year = valid_time.dt.year
    month = valid_time.dt.month
    day = valid_time.dt.day
    return year, month, day

# 提取变量数据，并将其格式化为适合GraphCast的输入格式
def prepare_graphcast_input(data, variables, year, month, day="07"):
    """
    准备ERA5数据，以GraphCast的输入格式保存
    """
    input_data = []
    y_d, m_d, d_d = extract_year_month_day_from_time(data['valid_time'])
    
    # 按时间、空间、变量提取数据
    for var in variables:
        selected_data = data[var].sel(method='nearest').where(
            (y_d == year) & (m_d == month) & (d_d == int(day)), drop=True
        ).squeeze()
        
        if selected_data.size == 0:
            raise ValueError(f"No data found for {var} at {year}-{month}-{day}")
        
        # 获取数据的空间网格
        spatial_data = selected_data.values
        input_data.append(spatial_data)
    
    # 将所有变量的时空数据合并
    input_data = np.stack(input_data, axis=-1)  # shape: (lat, lon, var_count)
    
    return input_data

# 对数据进行标准化
def standardize_data(data):
    """
    对数据进行标准化处理（Z-score标准化）
    """
    scaler = StandardScaler()
    # 对每个变量进行标准化（按变量的维度进行标准化）
    data_reshaped = data.reshape(-1, data.shape[-1])  # (lat * lon, var_count)
    scaled_data = scaler.fit_transform(data_reshaped)
    return scaled_data.reshape(data.shape)

# 保存处理好的数据
def save_processed_data(data, output_path):
    """
    将处理好的数据保存到指定路径
    """
    np.save(output_path, data)
    print(f"Data saved to {output_path}")

# 主函数：循环处理多个ERA5文件
def process_era5_data(files, variables, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        # 逐年逐月处理数据
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                try:
                    print(f"Processing data for {year}-{month:02d}")
                    
                    # 准备GraphCast模型的输入数据
                    input_data = prepare_graphcast_input(data, variables, year, month)
                    
                    # 对数据进行标准化
                    standardized_data = standardize_data(input_data)
                    
                    # 保存处理后的数据
                    output_file = os.path.join(output_dir, f"graphcast_input_{year}_{month:02d}.npy")
                    save_processed_data(standardized_data, output_file)
                    
                except Exception as e:
                    print(f"Error processing {year}-{month}: {e}")
                    continue

# 变量列表（按需选择）
variables = [
    "t",  # 温度
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

# 处理数据
process_era5_data(files, variables)