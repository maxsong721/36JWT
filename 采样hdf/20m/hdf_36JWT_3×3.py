"""
为36JWT区域S1和S2数据生成HDF文件 (Anytime模式)
Generate HDF file for 36JWT S1 and S2 data (Anytime mode)

与作者原始notebook逻辑完全一致:
AnytimeFormer-main/preprocess/hdf/3_generate_hdf_for_Germany_anytime.ipynb
"""

import os
import sys
import gc
import random
from glob import glob
from osgeo import gdal
import numpy as np
from tqdm import tqdm
from datetime import datetime
import h5py
from sklearn.preprocessing import StandardScaler


def imread(tif_file):
    ds = gdal.Open(tif_file)
    if ds is None:
        raise FileNotFoundError(f"无法打开文件: {tif_file}")
    return ds.ReadAsArray()


def int_to_doy(date_int):
    """将日期整数转换为年积日 (Day of Year)"""
    date_str = str(date_int)
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])     
    date_obj = datetime(year, month, day)  
    doy = date_obj.timetuple().tm_yday
    return doy


def sample_3x3_center(data):
    """
    对4D数组进行3×3采样，取中心点
    
    Args:
        data: (t, b, h, w) 或 (t, b, h*w) 格式的数组
    
    Returns:
        sampled_data: 采样后的数组
    """
    # 如果是reshape后的格式，先恢复
    if len(data.shape) == 3:
        # 需要知道原始的h和w，这里假设是正方形
        t, b, hw = data.shape
        h = w = int(np.sqrt(hw))
        data = data.reshape(t, b, h, w)
    
    t, b, h, w = data.shape
    
    # 裁剪到能被3整除
    h_new = (h // 3) * 3
    w_new = (w // 3) * 3
    data_cropped = data[:, :, :h_new, :w_new]
    
    # 3×3采样，取中心点（索引1）
    # 创建索引数组
    h_indices = np.arange(1, h_new, 3)  # [1, 4, 7, ...]
    w_indices = np.arange(1, w_new, 3)  # [1, 4, 7, ...]
    
    # 使用高级索引进行采样
    sampled_data = data_cropped[:, :, h_indices, :][:, :, :, w_indices]
    
    return sampled_data, h, w, h_new, w_new


def cal_mean_std(X, mask):
    """计算均值和标准差"""
    from einops import rearrange
    
    X = X.astype(np.float64) / 10000
    X[mask == 0] = np.nan
    
    X = rearrange(X, "n t b -> b (n t)")
    X = rearrange(X, "b nt -> nt b")
    X_mean = np.nanmean(X, axis=0) 
    X_std = np.nanstd(X, axis=0)
    return X_mean[np.newaxis, :], X_std[np.newaxis, :]


# ==================== 配置参数 ====================
site = "36JWT"

# HDF压缩设置 - 使用快速压缩以提高性能
HDF_COMPRESSION = "lzf"  # 快速压缩（比gzip level 9快10-20倍）
HDF_SHUFFLE = True       # 提高压缩率

# 修改为你的服务器路径
main_folder = "/public/home/xwlin/Data/songyibo/dataset_down_from_GEE/"
hdf_folder = "/public/home/xwlin/Data/songyibo/dataset_for_model/"

# 构建路径
s2_base_path = f"{main_folder}{site}/S2_remove_cloud/20m/"
s1_base_path = f"{main_folder}{site}/S1_raw/"
output_base_path = f"{hdf_folder}{site}/hdf/"

print("="*60)
print(f"配置信息")
print("="*60)
print(f"Site: {site}")
print(f"HDF压缩: {HDF_COMPRESSION} (快速压缩，比gzip level 9快10-20倍)")
print(f"HDF Shuffle: {HDF_SHUFFLE}")
print(f"S2路径: {s2_base_path}")
print(f"S1路径: {s1_base_path}")
print(f"输出路径: {output_base_path}")
# ==================== 配置参数结束 ====================

print("="*60)
print("开始处理S2数据...")
print("="*60)

# ==================== 1. 读取S2时间序列 ====================
s2_path = s2_base_path
tif_list = sorted(glob(os.path.join(s2_path, '*.tif')))
print(f"找到 {len(tif_list)} 个S2影像文件")

if len(tif_list) == 0:
    print(f"错误: 在 {s2_path} 中没有找到TIF文件")
    sys.exit(1)

s2_array = []
cloud_mask_array = []
date_list = []

no_cloud_percents = []
valiad_num = 0
non_valiad_num = 0

for tif in tqdm(tif_list, desc="读取S2影像"):
    s2 = imread(tif)
    s2_array.append(s2)

    # 提取日期 (格式: S2_L2A_20240902_10m.tif -> 20240902 或 10m)
    basename = os.path.basename(tif)
    parts = basename.replace(".tif", "").split('_')
    # 找到8位数字的日期
    date = None
    for part in parts:
        if len(part) == 8 and part.isdigit():
            date = part
            break
    if date is None:
        date = parts[-1].split(".")[0]  # 回退到原始逻辑
    date_list.append(date)
    
    # 生成云掩膜 (0表示缺失/云, 1表示有效)
    # 处理S2 NoData值和异常值：
    # - 0: NoData/云/阴影
    # - -32768: NoData标记
    # - > 10000: 异常值（超出正常反射率范围）
    cloud_mask_temp = (s2[0, :, :] > 0) & (s2[0, :, :] != -32768) & (s2[0, :, :] <= 10000)
    num_bands = s2.shape[0]  # 动态获取波段数
    cloud_mask_temp = np.repeat(cloud_mask_temp[np.newaxis, :, :], num_bands, axis=0)
    cloud_mask_array.append(cloud_mask_temp.astype(np.uint8))
    
    valiad_num_temp = np.sum(cloud_mask_temp[0, :, :])  # 使用mask而不是原始数据
    valiad_num += valiad_num_temp
    no_cloud_percents.append(valiad_num_temp / (s2.shape[1] * s2.shape[2]))
    non_valiad_num += np.sum(~cloud_mask_temp[0, :, :])

s2_array = np.stack(s2_array)    
t, b, h, w = s2_array.shape

# ==================== 3×3采样 ====================
print("\n" + "="*60)
print("开始3×3采样...")
print("="*60)
print(f"原始尺寸: {h} x {w}")

# 对S2数据进行3×3采样
s2_array_sampled, h_original, w_original, h_cropped, w_cropped = sample_3x3_center(s2_array)
t, b, h_sampled, w_sampled = s2_array_sampled.shape

print(f"裁剪后尺寸: {h_cropped} x {w_cropped}")
print(f"采样后尺寸: {h_sampled} x {w_sampled}")
print(f"数据量降低为原来的: {(h_sampled * w_sampled) / (h_original * w_original):.2%}")

# Reshape为HDF格式
s2_array = s2_array_sampled.reshape(t, b, -1)

cloud_mask_array = np.stack(cloud_mask_array)
# 对云掩膜也进行3×3采样
cloud_mask_array_sampled, _, _, _, _ = sample_3x3_center(cloud_mask_array)
cloud_mask_array = cloud_mask_array_sampled.reshape(t, b, -1)  # 0 is missing, 1 is valid

# 更新影像尺寸（用于S1处理）
IMG_HEIGHT = h_sampled
IMG_WIDTH = w_sampled
IMG_HEIGHT_ORIGINAL = h_original
IMG_WIDTH_ORIGINAL = w_original

valiad_num_for_series = valiad_num / (t * IMG_HEIGHT * IMG_WIDTH)
print(f"整个序列的有效像素占比: {valiad_num_for_series:.4f} ({valiad_num_for_series*100:.2f}%)")
print(f"采样后影像尺寸: {IMG_HEIGHT} x {IMG_WIDTH}, 时间步数: {t}, 波段数: {b}")
print(f"原始影像尺寸: {IMG_HEIGHT_ORIGINAL} x {IMG_WIDTH_ORIGINAL}")

doy_list = np.asarray([int_to_doy(d) for d in date_list])
print(f"保留影像的年积日 (DOY): {doy_list}")
print(f"保留影像的原始日期: {date_list}")

# ==================== 2. 创建HDF文件并写入S2数据 ====================
print("\n" + "="*60)
print("创建HDF文件并写入S2数据...")
print("="*60)

# 确保输出目录存在
hdf_output_dir = output_base_path
os.makedirs(hdf_output_dir, exist_ok=True)

hdf_save_path = os.path.join(hdf_output_dir, "anytime.hdf")

with h5py.File(hdf_save_path, "w") as hf:
    X_set = hf.create_group("data")
    # timelen, featuresnum, sample number → sample number, timelen, feature_num
    X_set.create_dataset("X", data=s2_array.astype(np.uint16).transpose(2, 0, 1), 
                         compression=HDF_COMPRESSION, shuffle=HDF_SHUFFLE)
    X_set.create_dataset("date", data=doy_list.astype(np.uint16), 
                         compression=HDF_COMPRESSION, shuffle=HDF_SHUFFLE)
    # 保存原始日期字符串（转换为整数）
    date_int_list = np.asarray([int(d) for d in date_list])
    X_set.create_dataset("date_original", data=date_int_list.astype(np.uint32),
                         compression=HDF_COMPRESSION, shuffle=HDF_SHUFFLE)
    
    mask_set = hf.create_group("mask")
    
    # ==================== 保存元数据 ====================
    metadata = hf.create_group("metadata")
    metadata.attrs["original_height"] = IMG_HEIGHT_ORIGINAL
    metadata.attrs["original_width"] = IMG_WIDTH_ORIGINAL
    metadata.attrs["sampled_height"] = IMG_HEIGHT
    metadata.attrs["sampled_width"] = IMG_WIDTH
    metadata.attrs["sampling_factor"] = 3
    metadata.attrs["cropped_height"] = h_cropped
    metadata.attrs["cropped_width"] = w_cropped
    
    print("\n✓ 元数据已保存:")
    print(f"  - 原始尺寸: {IMG_HEIGHT_ORIGINAL} x {IMG_WIDTH_ORIGINAL}")
    print(f"  - 裁剪尺寸: {h_cropped} x {w_cropped}")
    print(f"  - 采样尺寸: {IMG_HEIGHT} x {IMG_WIDTH}")
    print(f"  - 采样因子: 3")

print("S2数据写入完成")
print(f"✓ 已保存 {len(date_list)} 个时间点的原始日期信息")

# 释放不再需要的变量
del no_cloud_percents, valiad_num, non_valiad_num, date_int_list
gc.collect()
print("✓ 已释放临时变量内存")

# ==================== 3. 写入Mask数据 ====================
print("\n" + "="*60)
print("写入Mask数据...")
print("="*60)

random.seed(66)
np.random.seed(66)
novalid_percents = ["40%"]
candidate_cloud_masks = [0]

for cloud_mask_artifical, flag in zip(candidate_cloud_masks, novalid_percents):
    
    missing_mask_temp = cloud_mask_array.copy()  # 1 表示有效观测 | 1s represent clear pixel
    # Anytime mode 使用清晰像素的reconstruction MAE作为保存checkpoint的依据
    indicating_mask_temp = cloud_mask_array.copy()  # 1 代表保留验证; 即等于人工添加云的像素 | 1 represents the pixel that was held for testing.
    
    # 统计信息
    m_temp = np.sum(missing_mask_temp) / missing_mask_temp.size
    i_temp = np.sum(indicating_mask_temp) / indicating_mask_temp.size
    print(f"\n{flag} 掩码:")
    print(f"  missing_mask 有效观测占比: {m_temp:.4f} ({m_temp*100:.2f}%)")
    print(f"  indicating_mask 占比: {i_temp:.4f} ({i_temp*100:.2f}%)")
    
    with h5py.File(hdf_save_path, "a") as hf:
        mask_set = hf.require_group("mask")
        mask_set.create_dataset("missing_mask_" + flag, 
                               data=missing_mask_temp.astype(np.uint8).transpose(2, 0, 1), 
                               compression=HDF_COMPRESSION, shuffle=HDF_SHUFFLE)
        mask_set.create_dataset("indicating_mask_" + flag, 
                               data=indicating_mask_temp.astype(np.uint8).transpose(2, 0, 1), 
                               compression=HDF_COMPRESSION, shuffle=HDF_SHUFFLE)

print("Mask数据写入完成")

# 释放mask临时变量
del missing_mask_temp, indicating_mask_temp, m_temp, i_temp
gc.collect()
print("✓ 已释放mask临时变量内存")

# ==================== 4. 读取并处理S1数据 ====================
print("\n" + "="*60)
print("开始处理S1数据...")
print("="*60)

S1_folder = s1_base_path
tif_list = sorted(glob(S1_folder + '/*.tif'))
print(f"找到 {len(tif_list)} 个S1影像文件")

if len(tif_list) == 0:
    print(f"警告: 在 {S1_folder} 中没有找到S1文件，跳过S1处理")
    # 释放S2相关内存
    del s2_array, cloud_mask_array
    gc.collect()
    print("✓ 已释放S2数据内存")
    # 如果没有S1数据，直接结束
    print("="*60)
    print("HDF文件生成完成 (仅S2数据)")
    print(f"输出文件: {hdf_save_path}")
    print("="*60)
    sys.exit(0)

s1_array = []
s1_date_list = []

for tif in tqdm(tif_list, desc="读取S1影像"):
    # 检查文件是否存在
    if not os.path.exists(tif):
        print(f"\n警告: 文件不存在，跳过: {tif}")
        continue
    
    try:
        s1 = imread(tif)
    except Exception as e:
        print(f"\n警告: 读取文件失败，跳过: {tif}, 错误: {e}")
        continue
    
    # 使用采样后的S2影像尺寸来计算有效像素比例
    valiad_percent = np.sum(~np.isnan(s1[0, :, :])) / (IMG_HEIGHT_ORIGINAL * IMG_WIDTH_ORIGINAL)
    if valiad_percent > 0.9:
        s1_array.append(s1)
        
        # 提取日期
        basename = os.path.basename(tif)
        parts = basename.replace(".tif", "").split('_')
        date = None
        for part in parts:
            if len(part) == 8 and part.isdigit():
                date = part
                break
        if date is None:
            date = parts[-1].split(".")[0]
        s1_date_list.append(date)

print(f"有效S1影像数量: {len(s1_array)}")

if len(s1_array) > 0:
    s1_array = np.stack(s1_array)
    
    # ==================== 对S1数据进行3×3采样 ====================
    print("\n对S1数据进行3×3采样...")
    s1_array_sampled, _, _, _, _ = sample_3x3_center(s1_array)
    print(f"S1采样前: {s1_array.shape}, 采样后: {s1_array_sampled.shape}")
    
    t_s1, b_s1, h_s1, w_s1 = s1_array_sampled.shape
    s1_array = s1_array_sampled.reshape(t_s1, b_s1, -1)
else:
    print("警告: 没有有效的S1影像")
    s1_array = np.array([])
    t_s1 = b_s1 = 0

s1_doy_list = np.asarray([int_to_doy(d) for d in s1_date_list]) if len(s1_date_list) > 0 else np.array([])
print(f"有效SAR影像的年积日 (DOY): {s1_doy_list}")

# ==================== 5. 计算S1标准化参数 ====================
print("\n" + "="*60)
print("处理S1异常值...")
print("="*60)

if len(s1_array) > 0:
    # 统计异常值
    nan_count = np.isnan(s1_array).sum()
    inf_count = np.isinf(s1_array).sum()
    extreme_count = np.sum((np.abs(s1_array) > 100) & ~np.isnan(s1_array) & ~np.isinf(s1_array))
    total_pixels = s1_array.size

    print(f"S1异常值统计:")
    print(f"  NaN数量: {nan_count} ({nan_count/total_pixels*100:.4f}%)")
    print(f"  Inf数量: {inf_count} ({inf_count/total_pixels*100:.4f}%)")
    print(f"  极值数量 (|x|>100): {extreme_count} ({extreme_count/total_pixels*100:.4f}%)")

    # 处理异常值：先转换为NaN（用于后续写入HDF）
    s1_array[np.isinf(s1_array)] = np.nan
    s1_array[np.abs(s1_array) > 100] = np.nan

    print(f"✓ 已将 Inf 和极值 (|x|>100) 转换为 NaN")

    # 计算标准化参数（与作者一致：临时填充NaN为均值）
    scaler = StandardScaler()
    s1_array_ = s1_array.transpose(1, 0, 2).reshape(b_s1, -1).transpose(1, 0)

    # 临时填充NaN为均值（只用于计算标准化参数）
    s1_array_for_scaler = s1_array_.copy()
    s1_array_for_scaler[np.isnan(s1_array_for_scaler)] = np.nanmean(s1_array_for_scaler)

    scaler.fit(s1_array_for_scaler)

    mean = scaler.mean_
    std = scaler.scale_
    print(f"\nS1标准化参数:")
    print(f"  均值: {mean}")
    print(f"  标准差: {std}")

    # 释放临时变量
    del s1_array_, s1_array_for_scaler, scaler
    gc.collect()
    print("✓ 已释放S1标准化临时数据")
else:
    print("跳过S1标准化参数计算（无有效S1数据）")
    mean = np.array([0.0, 0.0])
    std = np.array([1.0, 1.0])

# ==================== 6. 写入S1数据到HDF ====================
print("\n" + "="*60)
print("写入S1数据到HDF...")
print("="*60)

if len(s1_array) > 0:
    # S1数据保留NaN（不替换），让模型在训练时处理
    # 这样更符合遥感数据的标准处理流程
    print("注意: S1数据中的NaN将保留，由模型在训练时处理")

    with h5py.File(hdf_save_path, "a") as hf:
        group = hf.require_group('data')
        
        # 直接写入，保留NaN
        group.create_dataset("X_aux", data=s1_array.astype(np.float32).transpose(2, 0, 1), 
                            compression=HDF_COMPRESSION, shuffle=HDF_SHUFFLE)
        group.create_dataset("date_aux", data=s1_doy_list.astype(np.uint16), 
                            compression=HDF_COMPRESSION, shuffle=HDF_SHUFFLE)
        
        group.create_dataset("X_aux_mean", data=mean.astype(np.float32), 
                            compression=HDF_COMPRESSION, shuffle=HDF_SHUFFLE)
        group.create_dataset("X_aux_std", data=std.astype(np.float32), 
                            compression=HDF_COMPRESSION, shuffle=HDF_SHUFFLE)

    print("S1数据写入完成")

    # 释放S1相关内存
    del s1_array, s1_date_list, s1_doy_list, mean, std
    gc.collect()
    print("✓ 已释放S1数据内存")
else:
    print("跳过S1数据写入（无有效S1数据）")

# ==================== 7. 验证HDF数据 ====================
print("\n" + "="*60)
print("验证HDF数据...")
print("="*60)

with h5py.File(hdf_save_path, "r") as hf:
    # 不加载全部数据，只读取shape和部分数据验证
    missing_mask_shape = hf["mask"]["missing_mask_40%"].shape
    indicating_mask_shape = hf["mask"]["indicating_mask_40%"].shape
    
    # 计算占比（使用分块读取避免内存不足）
    chunk_size = 1000000  # 每次读取100万个样本
    n_samples = missing_mask_shape[0]
    
    missing_sum = 0
    indicating_sum = 0
    total_elements = missing_mask_shape[0] * missing_mask_shape[1] * missing_mask_shape[2]
    
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        missing_chunk = hf["mask"]["missing_mask_40%"][start_idx:end_idx]
        indicating_chunk = hf["mask"]["indicating_mask_40%"][start_idx:end_idx]
        
        missing_sum += np.sum(missing_chunk)
        indicating_sum += np.sum(indicating_chunk)
        
        # 释放内存
        del missing_chunk, indicating_chunk
        gc.collect()
    
    m40 = missing_sum / total_elements
    i40 = indicating_sum / total_elements
    
    print(f"有效占比: {m40:.4f} ({m40*100:.2f}%)")
    print(f"保留验证像素占比: {i40:.4f} ({i40*100:.2f}%)")

# ==================== 8. 计算并写入S2均值和标准差 ====================
print("\n" + "="*60)
print("计算并写入S2均值和标准差...")
print("="*60)

with h5py.File(hdf_save_path, "a") as hf:
    # 分块读取X数据，避免内存不足
    X_shape = hf["data"]["X"].shape
    n_samples = X_shape[0]
    chunk_size = 1000000  # 每次读取100万个样本
    
    # 读取missing_mask（分块）
    print("读取missing_mask...")
    missing_mask_chunks = []
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk = hf["mask"]["missing_mask_40%"][start_idx:end_idx]
        missing_mask_chunks.append(chunk)
    missing_mask_40 = np.concatenate(missing_mask_chunks, axis=0)
    del missing_mask_chunks
    gc.collect()
    
    # 读取X数据（分块）
    print("读取X数据...")
    X_chunks = []
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk = hf["data"]["X"][start_idx:end_idx]
        X_chunks.append(chunk)
    X = np.concatenate(X_chunks, axis=0)
    del X_chunks
    gc.collect()
    
    group = hf.require_group('data')
    
    print("计算均值和标准差...")
    X_mean_40, X_std_40 = cal_mean_std(X, missing_mask_40)
    
    group.create_dataset("X_mean_40%", data=X_mean_40.astype(np.float32))
    group.create_dataset("X_std_40%", data=X_std_40.astype(np.float32))

    print(f"S2均值 (40%): {X_mean_40}")
    print(f"S2标准差 (40%): {X_std_40}")
    
    # 释放大型数组
    del X, missing_mask_40
    gc.collect()
    print("✓ 已释放验证数据内存")

# ==================== 9. 最终验证 ====================
print("\n" + "="*60)
print("最终验证...")
print("="*60)

with h5py.File(hdf_save_path, "r") as hf:
    print("\nHDF文件结构:")
    def print_structure(name, obj):
        print(f"  {name}: {type(obj).__name__}", end="")
        if isinstance(obj, h5py.Dataset):
            print(f" - shape: {obj.shape}, dtype: {obj.dtype}")
        else:
            print()
    hf.visititems(print_structure)
    
    X_mean_40, X_std_40 = hf["data"]["X_mean_40%"][:], hf["data"]["X_std_40%"][:]
    print(f"\n验证 - S2均值 (40%): {X_mean_40}")
    print(f"验证 - S2标准差 (40%): {X_std_40}")

print("\n" + "="*60)
print("HDF文件生成完成!")
print(f"保存路径: {hdf_save_path}")
print("="*60)
print("\n✓ 数据信息:")
print(f"  - Site: {site}")
print(f"  - 原始影像尺寸: {IMG_HEIGHT_ORIGINAL} x {IMG_WIDTH_ORIGINAL}")
print(f"  - 采样后尺寸: {IMG_HEIGHT} x {IMG_WIDTH}")
print(f"  - 采样因子: 3")
print(f"  - 数据量降低为: {(IMG_HEIGHT * IMG_WIDTH) / (IMG_HEIGHT_ORIGINAL * IMG_WIDTH_ORIGINAL):.2%}")
print(f"  - S2时间步数: {t}")
print(f"  - S2波段数: {b}")
print(f"  - 时间范围: {date_list[0]} 至 {date_list[-1]}")
print(f"  - 有效像素占比: {valiad_num_for_series:.2%}")
print("="*60)

# 最终内存清理
del s2_array, cloud_mask_array, date_list, doy_list, X_mean_40, X_std_40
gc.collect()
print("\n✓ 所有内存已释放，处理完成！")
