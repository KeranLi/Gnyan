import os
import xarray as xr
import matplotlib.pyplot as plt
import re  # 导入正则表达式模块

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

# 可视化每年每月四天数据的空间分布并保存图片
def visualize_monthly_data(data, variables, year, month, days=["07", "14", "21", "28"], time="14:00", output_dir="./output"):
    """
    Visualize 4 days of data for a specific month and year.
    :param data: xarray.Dataset
    :param variables: List of variable names
    :param year: Year to visualize
    :param month: Month to visualize
    :param days: List of days to visualize
    :param time: Fixed time ("14:00")
    :param output_dir: Directory to save the images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))

    # For each variable, plot the data for the four days in the specified month and year
    for i, var in enumerate(variables):
        plt.subplot(2, 4, i+1)
        
        for day in days:
            # 提取有效时间 (valid_time)，然后从中提取年、月、日
            valid_time = data['valid_time']
            year_data, month_data, day_data = extract_year_month_day_from_time(valid_time)
            
            # 过滤出特定年份、月份、日期的数据
            variable_data = data[var].sel(valid_time=valid_time.sel(valid_time=(year_data == year) & 
                                                                    (month_data == month) & 
                                                                    (day_data == int(day)) & 
                                                                    (valid_time.dt.hour == 14)))
            # 检查是否找到数据
            if variable_data.size == 0:
                print(f"Warning: No data found for {var} on {day}/{month}/{year}.")
            else:
                print(f"Plotting {var} on {day}/{month}/{year}.")
                variable_data.plot()  # 绘制数据
            
            plt.title(f"{var} on {day}/{month}/{year}")

    plt.suptitle(f"Variable Distribution for {year}-{month}", fontsize=16)
    plt.tight_layout()

    # Save the figure as an image file
    output_file = os.path.join(output_dir, f"ERA5_{year}_{month}.png")
    plt.savefig(output_file)
    print(f"Saved image for {year}-{month} at {output_file}")
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