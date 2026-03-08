"""
原始分辨率推理脚本
功能：直接在原始TIF上进行Window分块推理，输出完整分辨率影像
作者：基于AnytimeFormer改编
日期：2025-01-13
"""

import os
import sys
import argparse
import h5py
import numpy as np
import torch
from osgeo import gdal, osr
from datetime import datetime
from tqdm import tqdm
import yaml
import gc

# 导入模型相关
from model import model_dict
from model.utils import (
    setup_logger, seed_torch, load_model, str2bool,
    check_path, doy_to_ymd
)


def int_to_doy(date_int):
    """将日期整数转换为DOY (Day of Year)
    
    Args:
        date_int: 日期整数，格式YYYYMMDD，例如20240901
        
    Returns:
        doy: 一年中的第几天 (1-366)
        
    Example:
        >>> int_to_doy(20240901)
        245  # 2024年的第245天
    """
    date_str = str(date_int)
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    
    date_obj = datetime(year, month, day)
    doy = date_obj.timetuple().tm_yday
    
    return doy


class FullResolutionInference:
    """原始分辨率推理类"""
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.scale = args.scale
        
        # 加载标准化参数
        self._load_normalization_params()
        
        # 扫描TIF文件
        self._scan_tif_files()
        
        # 设置anytime_ouput为S2的DOY（推理时输出所有S2时间点）
        self.args.anytime_ouput = self.s2_doy.tolist()
        
        # 将标准化参数转换为torch tensor并设置到args（模型初始化需要）
        self.args.X_mean = torch.from_numpy(self.X_mean).to(self.device)
        self.args.X_std = torch.from_numpy(self.X_std).to(self.device)
        
        # 加载模型
        self._load_model()
        
    def _load_normalization_params(self):
        """从训练HDF加载标准化参数"""
        logger.info(f"从训练HDF加载标准化参数: {self.args.train_hdf_path}")
        
        with h5py.File(self.args.train_hdf_path, "r") as hf:
            # S2标准化参数
            self.X_mean = hf["data"][f"X_mean_{self.args.ratio}"][:]
            self.X_std = hf["data"][f"X_std_{self.args.ratio}"][:]
            
            # S1标准化参数（如果使用）
            if self.args.with_X_aux:
                self.X_aux_mean = hf["data"]["X_aux_mean"][:]
                self.X_aux_std = hf["data"]["X_aux_std"][:]
            else:
                self.X_aux_mean = np.array([0.0, 0.0])
                self.X_aux_std = np.array([1.0, 1.0])
        
        logger.info(f"✓ S2标准化参数: mean={self.X_mean}, std={self.X_std}")
        if self.args.with_X_aux:
            logger.info(f"✓ S1标准化参数: mean={self.X_aux_mean}, std={self.X_aux_std}")
    
    def _scan_tif_files(self):
        """扫描S2和S1的TIF文件，提取日期"""
        logger.info(f"扫描TIF文件...")
        logger.info(f"  S2文件夹: {self.args.s2_tif_folder}")
        if self.args.with_X_aux:
            logger.info(f"  S1文件夹: {self.args.s1_tif_folder}")
        
        # 扫描S2文件
        s2_files = sorted([
            f for f in os.listdir(self.args.s2_tif_folder)
            if f.endswith('.tif') and 'S2' in f
        ])
        self.s2_file_paths = [
            os.path.join(self.args.s2_tif_folder, f) for f in s2_files
        ]
        
        # 提取S2日期
        self.s2_dates = []
        for f in s2_files:
            # 从文件名提取日期，例如 S2_L2A_20240902_10m.tif -> 20240902
            date_str = f.split('_')[2]  # 假设格式固定
            self.s2_dates.append(int(date_str))
        
        # 转换为DOY
        self.s2_doy = np.array([int_to_doy(date) for date in self.s2_dates])
        
        logger.info(f"✓ 找到 {len(self.s2_file_paths)} 个S2文件")
        logger.info(f"  日期范围: {self.s2_dates[0]} - {self.s2_dates[-1]}")
        logger.info(f"  DOY范围: {self.s2_doy[0]} - {self.s2_doy[-1]}")
        
        # 扫描S1文件（如果使用）
        if self.args.with_X_aux:
            s1_files = sorted([
                f for f in os.listdir(self.args.s1_tif_folder)
                if f.endswith('.tif') and 'S1' in f
            ])
            self.s1_file_paths = [
                os.path.join(self.args.s1_tif_folder, f) for f in s1_files
            ]
            
            # 提取S1日期
            self.s1_dates = []
            for f in s1_files:
                # 从文件名提取日期，例如 S1_GRD_20240907.tif -> 20240907
                date_str = f.split('_')[2].split('.')[0]
                self.s1_dates.append(int(date_str))
            
            # 转换为DOY
            self.s1_doy = np.array([int_to_doy(date) for date in self.s1_dates])
            
            logger.info(f"✓ 找到 {len(self.s1_file_paths)} 个S1文件")
            logger.info(f"  日期范围: {self.s1_dates[0]} - {self.s1_dates[-1]}")
            logger.info(f"  DOY范围: {self.s1_doy[0]} - {self.s1_doy[-1]}")
        else:
            self.s1_file_paths = []
            self.s1_dates = []
            self.s1_doy = np.array([0.0])
        
        # 获取影像尺寸和地理信息
        ds = gdal.Open(self.s2_file_paths[0])
        self.height = ds.RasterYSize
        self.width = ds.RasterXSize
        self.n_bands = ds.RasterCount
        self.projection = ds.GetProjection()
        self.geotransform = ds.GetGeoTransform()
        self.datatype = ds.GetRasterBand(1).DataType
        ds = None  # 关闭数据集
        
        logger.info(f"✓ 影像尺寸: {self.height} × {self.width}")
        logger.info(f"✓ 波段数: {self.n_bands}")
    
    def _load_model(self):
        """加载训练好的模型"""
        logger.info(f"加载模型: {self.args.saved_model_path}")
        
        self.model = model_dict[self.args.model_name](self.args)
        self.model = self.model.to(self.device)
        load_model(self.model, self.args.saved_model_path, logger)
        self.model.eval()
        
        logger.info("✓ 模型加载完成")
    
    def _read_time_series_block(self, x_off, y_off, x_size, y_size):
        """读取指定位置的时序数据（使用GDAL）
        
        Args:
            x_off: 列起始位置
            y_off: 行起始位置
            x_size: 列数（宽度）
            y_size: 行数（高度）
            
        Returns:
            s2_block: (t, b, h, w) S2时序数据
            s1_block: (t, b, h, w) S1时序数据（如果使用）
        """
        # 读取S2时序
        s2_list = []
        for s2_path in self.s2_file_paths:
            ds = gdal.Open(s2_path)
            # 读取所有波段
            bands_data = []
            for i in range(1, ds.RasterCount + 1):
                band = ds.GetRasterBand(i)
                data = band.ReadAsArray(x_off, y_off, x_size, y_size)
                bands_data.append(data)
            s2_list.append(np.stack(bands_data, axis=0))  # (b, h, w)
            ds = None
        s2_block = np.stack(s2_list, axis=0)  # (t, b, h, w)
        
        # 读取S1时序（如果使用）
        if self.args.with_X_aux:
            s1_list = []
            for s1_path in self.s1_file_paths:
                ds = gdal.Open(s1_path)
                bands_data = []
                for i in range(1, ds.RasterCount + 1):
                    band = ds.GetRasterBand(i)
                    data = band.ReadAsArray(x_off, y_off, x_size, y_size)
                    bands_data.append(data)
                s1_list.append(np.stack(bands_data, axis=0))  # (b, h, w)
                ds = None
            s1_block = np.stack(s1_list, axis=0)  # (t, b, h, w)
        else:
            s1_block = np.zeros((1, 2, y_size, x_size), dtype=np.float32)
        
        return s2_block, s1_block
    
    def _preprocess_block(self, s2_block, s1_block):
        """预处理block数据
        
        Args:
            s2_block: (t, b, h, w) S2数据
            s1_block: (t, b, h, w) S1数据
            
        Returns:
            s2_input: (n, t, b) 标准化后的S2数据
            s1_input: (n, t, b) 标准化后的S1数据
            missing_mask: (n, t) 有效观测掩膜
            attention_mask: (n, t) 注意力掩膜
        """
        t, b, h, w = s2_block.shape
        n = h * w
        
        # S2标准化
        # X_mean和X_std的形状是(4,)，需要reshape为(1, 4, 1, 1)以匹配(t, b, h, w)
        s2_normalized = s2_block.astype(np.float32) / self.scale
        s2_normalized = np.where(
            s2_block != 0,
            (s2_normalized - self.X_mean[None, :, None, None]) / self.X_std[None, :, None, None],
            0
        )
        
        # S1标准化
        if self.args.with_X_aux:
            # 处理NaN
            s1_block = np.nan_to_num(s1_block, nan=0.0)
            # X_aux_mean和X_aux_std的形状是(3,)，需要reshape为(1, 3, 1, 1)
            s1_normalized = (s1_block - self.X_aux_mean[None, :, None, None]) / self.X_aux_std[None, :, None, None]
            s1_normalized = np.where(s1_block != 0, s1_normalized, 0)
        else:
            s1_normalized = s1_block
        
        # Reshape: (t, b, h, w) -> (n, t, b)
        s2_input = s2_normalized.reshape(t, b, -1).transpose(2, 0, 1)
        s1_input = s1_normalized.reshape(t, b, -1).transpose(2, 0, 1)
        
        # 生成missing_mask: 0是缺失，1是有效
        # 判断条件：值>0 且 值!=NoData 且 值<=10000
        valid_mask = (s2_block > 0) & (s2_block != -32768) & (s2_block <= 10000)
        valid_mask = valid_mask.all(axis=1)  # (t, h, w) - 所有波段都有效
        missing_mask = valid_mask.reshape(t, -1).T  # (n, t)
        
        # 生成attention_mask: 0是有效，1是缺失
        # 判断条件：所有波段都是0
        attention_mask = (s2_input == 0).all(axis=-1)  # (n, t)
        
        return s2_input, s1_input, missing_mask.astype(np.float32), attention_mask.astype(np.float32)
    
    def _inference_block(self, s2_input, s1_input, missing_mask, attention_mask):
        """对单个block进行推理
        
        Args:
            s2_input: (n, t, b) S2数据
            s1_input: (n, t, b) S1数据
            missing_mask: (n, t) 有效观测掩膜
            attention_mask: (n, t) 注意力掩膜
            
        Returns:
            output: (n, t_out, b) 重建后的数据
        """
        # 转换为tensor
        date_input = torch.from_numpy(self.s2_doy.astype(np.float32)).to(self.device)
        date_input = date_input.unsqueeze(0).repeat(s2_input.shape[0], 1)
        
        X = torch.from_numpy(s2_input.astype(np.float32)).to(self.device)
        X_aux = torch.from_numpy(s1_input.astype(np.float32)).to(self.device)
        missing_mask_tensor = torch.from_numpy(missing_mask).to(self.device)
        attention_mask_tensor = torch.from_numpy(attention_mask).to(self.device)
        
        dates_aux = torch.from_numpy(self.s1_doy.astype(np.float32)).to(self.device)
        if self.args.with_X_aux:
            dates_aux = dates_aux.unsqueeze(0).repeat(s2_input.shape[0], 1)
        
        # 输出时间点：与输入S2的DOY一致
        date_output = date_input.clone()
        
        # 构建输入
        inputs = {
            "date_input": date_input,
            "X": X,
            "missing_mask": missing_mask_tensor,
            "X_holdout": X,  # 推理时不需要，但模型可能需要
            "indicating_mask": missing_mask_tensor,  # 推理时不需要
            "attention_mask": attention_mask_tensor,
            "X_aux": X_aux,
            "dates_aux": dates_aux,
            "date_output": date_output
        }
        
        # 推理
        with torch.no_grad():
            results = self.model(inputs, stage="test")
            output = results["reconstructed_data"]  # (n, t_out, b)
        
        return output.cpu().numpy()
    
    def _postprocess_block(self, output, block_h, block_w):
        """后处理：转换为int16并reshape回空间维度
        
        Args:
            output: (n, t_out, b) 模型输出（已经反标准化到0-1范围）
            block_h: block高度
            block_w: block宽度
            
        Returns:
            output_spatial: (t_out, b, h, w) int16格式
        """
        # 注意：模型内部已经做了反标准化 (x * std + mean)
        # 这里只需要乘回scale转为int16
        output_int16 = np.round(output * self.scale).astype(np.int16)
        
        # Reshape: (n, t_out, b) -> (t_out, b, h, w)
        output_spatial = output_int16.reshape(block_h, block_w, -1, output_int16.shape[-1])
        output_spatial = output_spatial.transpose(2, 3, 0, 1)  # (t_out, b, h, w)
        
        return output_spatial
    
    def run_inference(self):
        """执行完整推理流程"""
        logger.info("=" * 80)
        logger.info("开始推理...")
        logger.info(f"  Block大小: {self.args.block_size} × {self.args.block_size}")
        logger.info(f"  输出时间点数: {len(self.s2_doy)}")
        logger.info("=" * 80)
        
        # 初始化输出数组
        t_out = len(self.s2_doy)
        b_out = self.n_bands
        output_arrays = {
            doy: np.zeros((b_out, self.height, self.width), dtype=np.int16)
            for doy in self.s2_doy
        }
        
        # 计算block数量
        n_blocks_h = int(np.ceil(self.height / self.args.block_size))
        n_blocks_w = int(np.ceil(self.width / self.args.block_size))
        total_blocks = n_blocks_h * n_blocks_w
        
        logger.info(f"总共需要处理 {total_blocks} 个blocks ({n_blocks_h} × {n_blocks_w})")
        
        # 遍历所有blocks
        block_idx = 0
        for i in tqdm(range(0, self.height, self.args.block_size), desc="行进度"):
            for j in range(0, self.width, self.args.block_size):
                block_idx += 1
                
                # 计算实际block大小（处理边界）
                block_h = min(self.args.block_size, self.height - i)
                block_w = min(self.args.block_size, self.width - j)
                
                # GDAL读取参数：x_off, y_off, x_size, y_size
                x_off = j
                y_off = i
                x_size = block_w
                y_size = block_h
                
                # 读取时序数据
                s2_block, s1_block = self._read_time_series_block(x_off, y_off, x_size, y_size)
                
                # 预处理
                s2_input, s1_input, missing_mask, attention_mask = \
                    self._preprocess_block(s2_block, s1_block)
                
                # 推理
                output = self._inference_block(s2_input, s1_input, missing_mask, attention_mask)
                
                # 后处理
                output_spatial = self._postprocess_block(output, block_h, block_w)
                
                # 写入输出数组
                for t_idx, doy in enumerate(self.s2_doy):
                    output_arrays[doy][:, i:i+block_h, j:j+block_w] = output_spatial[t_idx]
                
                # 释放内存
                del s2_block, s1_block, s2_input, s1_input, missing_mask, attention_mask, output, output_spatial
                
                if block_idx % 10 == 0:
                    gc.collect()
        
        logger.info("✓ 所有blocks推理完成")
        
        # 写入TIF文件
        self._write_output_tif(output_arrays)
        
        logger.info("=" * 80)
        logger.info("推理完成！")
        logger.info("=" * 80)
    
    def _write_output_tif(self, output_arrays):
        """写入输出TIF文件（使用GDAL）
        
        Args:
            output_arrays: {doy: (b, h, w)} 字典
        """
        logger.info(f"写入输出TIF文件到: {self.args.output_folder}")
        check_path(self.args.output_folder)
        
        # 创建GDAL驱动
        driver = gdal.GetDriverByName('GTiff')
        
        for doy in tqdm(self.s2_doy, desc="写入TIF"):
            # DOY转日期
            ymd = doy_to_ymd(year=self.args.year, doy=int(doy))
            ymd_str = ymd.replace('-', '')
            
            # 输出文件名
            output_filename = f"S2_L2A_reconstructed_{ymd_str}.tif"
            output_path = os.path.join(self.args.output_folder, output_filename)
            
            # 创建输出数据集
            out_ds = driver.Create(
                output_path,
                self.width,
                self.height,
                self.n_bands,
                gdal.GDT_Int16,
                options=['COMPRESS=LZW', 'TILED=YES']
            )
            
            # 设置地理变换和投影
            out_ds.SetGeoTransform(self.geotransform)
            out_ds.SetProjection(self.projection)
            
            # 写入数据
            for i in range(self.n_bands):
                band = out_ds.GetRasterBand(i + 1)
                band.WriteArray(output_arrays[doy][i])
                band.FlushCache()
            
            # 关闭数据集
            out_ds = None
        
        logger.info(f"✓ 已写入 {len(self.s2_doy)} 个TIF文件")


def main():
    parser = argparse.ArgumentParser(description="原始分辨率推理脚本")
    
    # 必需参数
    parser.add_argument(
        "--train_hdf_path",
        type=str,
        default="/public/home/xwlin/Data/songyibo/dataset_for_model/36JWT/hdf/anytime.hdf",
        help="训练HDF路径（用于加载标准化参数）"
    )
    parser.add_argument(
        "--saved_model_path",
        type=str,
        default="/public/home/xwlin/Data/songyibo/dataset_for_model/36JWT/work_dir/anytime/2026-03-07_T19-01-55_AnytimeFormer-36JWT-40%-r8-128-wTV/models/best_model.ckpt",
        help="训练好的模型路径"
    )
    parser.add_argument(
        "--s2_tif_folder",
        type=str,
        default="/public/home/xwlin/Data/songyibo/dataset_down_from_GEE/36JWT/S2_remove_cloud",
        help="S2 TIF文件夹路径"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/public/home/xwlin/Data/songyibo/dataset_for_model/36JWT/work_dir/anytime/2026-03-07_T19-01-55_AnytimeFormer-36JWT-40%-r8-128-wTV/inference",
        help="输出文件夹路径"
    )
    
    # 可选参数
    parser.add_argument(
        "--s1_tif_folder",
        type=str,
        default="/public/home/xwlin/Data/songyibo/dataset_down_from_GEE/36JWT/S1_raw",
        help="S1 TIF文件夹路径（如果使用S1）"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="模型配置文件路径"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Window分块大小"
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default="40%",
        help="训练时使用的缺失率（用于加载对应的标准化参数）"
    )
    parser.add_argument(
        "--with_X_aux",
        type=str2bool,
        default=True,
        help="是否使用S1辅助数据"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10000,
        help="影像缩放系数"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="影像年份（用于DOY转日期）"
    )
    
    # 模型参数（从训练配置中复制）
    parser.add_argument("--model_name", type=str, default="AnytimeFormer", help="模型名称")
    parser.add_argument("--with_atten_mask", type=str2bool, default=False, help="是否使用attention mask")
    parser.add_argument("--n_groups", type=int, default=2, help="层组数")
    parser.add_argument("--n_group_inner_layers", type=int, default=1, help="组内层数")
    parser.add_argument("--d_model", type=int, default=128, help="模型隐藏维度")
    parser.add_argument("--d_inner", type=int, default=64, help="前馈层隐藏维度")
    parser.add_argument("--n_head", type=int, default=4, help="注意力头数")
    parser.add_argument("--d_k", type=int, default=32, help="key维度")
    parser.add_argument("--d_v", type=int, default=32, help="value维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout率")
    parser.add_argument("--diagonal_attention_mask", type=str2bool, default=True, help="是否使用对角attention mask")
    parser.add_argument("--d_feature", type=int, default=4, help="S2波段数")
    parser.add_argument("--d_time", type=int, default=61, help="S2时间步数")
    parser.add_argument("--d_time_aux", type=int, default=70, help="S1时间步数")
    parser.add_argument("--d_feature_aux", type=int, default=3, help="S1波段数")
    parser.add_argument("--learn_emb", type=str2bool, default=False, help="是否学习embedding")
    parser.add_argument("--rank", type=int, default=8, help="rank参数")
    parser.add_argument("--with_rec_loss", type=str2bool, default=True, help="是否使用重建损失")
    parser.add_argument("--with_tv", type=str2bool, default=True, help="是否使用TV损失")
    
    args = parser.parse_args()
    
    # 设置anytime_ouput为S2的DOY（推理时输出所有S2时间点）
    # 这个参数会在后面根据实际扫描的S2文件动态设置
    
    # 添加其他必需的训练参数（推理时不使用，但模型初始化需要）
    args.artificial_missing_rate = 0.25
    args.gap_mode = "random"
    args.X_mean = None  # 会在后面设置
    args.X_std = None   # 会在后面设置
    
    # 加载配置文件（如果提供）
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for section in cfg:
            cfg_args = cfg[section]
            for key, value in cfg_args.items():
                if not hasattr(args, key):
                    setattr(args, key, value)
    
    # 设置日志
    log_path = os.path.join(args.output_folder, "inference.log")
    check_path(args.output_folder)
    global logger
    logger = setup_logger(log_path, "w")
    
    # 打印参数
    logger.info("=" * 80)
    logger.info("推理参数:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)
    
    # 设置随机种子
    seed_torch()
    
    # 执行推理
    inferencer = FullResolutionInference(args)
    inferencer.run_inference()


if __name__ == "__main__":
    main()
