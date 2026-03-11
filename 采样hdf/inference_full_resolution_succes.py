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
import threading
import queue

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
            
            # 确保形状正确：应该是(4,)或(1,1,1,4)，统一flatten为(4,)
            self.X_mean = self.X_mean.flatten()
            self.X_std = self.X_std.flatten()
            
            # S1标准化参数（如果使用）
            if self.args.with_X_aux:
                self.X_aux_mean = hf["data"]["X_aux_mean"][:]
                self.X_aux_std = hf["data"]["X_aux_std"][:]
                # 确保形状正确：应该是(3,)
                self.X_aux_mean = self.X_aux_mean.flatten()
                self.X_aux_std = self.X_aux_std.flatten()
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
        
        # 转换为DOY（保持原始 1~365 范围，不做跨年连续化）
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
        load_model(self.model, self.args.saved_model_path, logger)
        # load_model 之后再 to(device)，确保权重加载后整体移到GPU
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 验证模型确实在GPU上
        first_param_device = next(self.model.parameters()).device
        logger.info(f"✓ 模型加载完成，参数设备: {first_param_device}")
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        logger.info(f"  模型加载后显存: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
        if allocated < 0.05:
            logger.warning("  ⚠️  显存接近0，模型可能未正确加载到GPU！请检查 load_model 实现")
    

    def _open_datasets(self):
        """预先打开所有GDAL数据集，避免每个block重复open/close"""
        self._s2_ds = [gdal.Open(p) for p in self.s2_file_paths]
        if self.args.with_X_aux:
            self._s1_ds = [gdal.Open(p) for p in self.s1_file_paths]
        else:
            self._s1_ds = []
        logger.info(f"✓ 预开 {len(self._s2_ds)} 个S2 + {len(self._s1_ds)} 个S1 数据集")

    def _close_datasets(self):
        self._s2_ds = []
        self._s1_ds = []

    def _read_time_series_block(self, x_off, y_off, x_size, y_size):
        """读取指定位置的时序数据（复用预开的GDAL数据集，用ds.ReadAsArray一次读所有波段）"""
        s2_block = np.stack(
            [ds.ReadAsArray(x_off, y_off, x_size, y_size) for ds in self._s2_ds], axis=0
        )  # (t, b, h, w)
        if self.args.with_X_aux:
            s1_block = np.stack(
                [ds.ReadAsArray(x_off, y_off, x_size, y_size) for ds in self._s1_ds], axis=0
            ).astype(np.float32)
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
        t_s2, b_s2, h, w = s2_block.shape
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
            t_s1, b_s1, _, _ = s1_block.shape
            # Keep S1 preprocessing consistent with HDF generation + training:
            # inf/extreme -> NaN, then fill NaN with 0 before standardization.
            s1_block = s1_block.astype(np.float32)
            s1_block[np.isinf(s1_block)] = np.nan
            s1_block[np.abs(s1_block) > 100] = np.nan
            s1_block = np.nan_to_num(s1_block, nan=0.0, posinf=0.0, neginf=0.0)
            # X_aux_mean和X_aux_std的形状是(3,)，需要reshape为(1, 3, 1, 1)
            s1_normalized = (s1_block - self.X_aux_mean[None, :, None, None]) / self.X_aux_std[None, :, None, None]
            # Reshape S1: (t_s1, b_s1, h, w) -> (n, t_s1, b_s1)
            s1_input = s1_normalized.reshape(t_s1, b_s1, -1).transpose(2, 0, 1)
        else:
            s1_input = np.zeros((n, 1, 1), dtype=np.float32)  # 占位符
        
        # Reshape S2: (t_s2, b_s2, h, w) -> (n, t_s2, b_s2)
        s2_input = s2_normalized.reshape(t_s2, b_s2, -1).transpose(2, 0, 1)
        # s2_input_full 保留完整数据作为 X_holdout（含云像素的标准化值）
        s2_input_full = s2_input.copy()
        
        # 生成missing_mask: 0是缺失，1是有效
        # 判断条件：值>0 且 值!=NoData 且 值<=10000
        valid_mask = (s2_block > 0) & (s2_block != -32768) & (s2_block <= 10000)
        valid_mask = valid_mask.all(axis=1)  # (t_s2, h, w) - 所有波段都有效
        missing_mask = valid_mask.reshape(t_s2, -1).T  # (n, t_s2)
        # 扩展到波段维度: (n, t_s2) -> (n, t_s2, b_s2)
        missing_mask = np.repeat(missing_mask[:, :, None], b_s2, axis=2)
        
        # 把缺失位置（missing_mask=0）在 s2_input 中置0，作为模型输入 X
        # 作者做法：X_hat[missing_mask==0] = 0
        # missing_mask shape: (n, t, b)，s2_input shape: (n, t, b)
        s2_input = np.where(missing_mask > 0, s2_input, 0.0)
        
        # 生成attention_mask: 0是有效，1是缺失
        # 作者用 X_hat（置0后）判断，我们也用置0后的 s2_input
        attention_mask = (s2_input == 0).all(axis=-1)  # (n, t_s2)
        
        # 保护：如果某像素所有时间步都被mask，softmax产生NaN
        # 策略：强制保留第一个时间步为有效，给模型一个基准点
        all_masked = attention_mask.all(axis=1)  # (n,) 全程无观测的像素
        if all_masked.any():
            attention_mask[all_masked, 0] = 0  # 强制第一个时间步为"有效"
        
        # 二次保护：attention_mask全为True的情况（数值上可能因浮点精度残留）
        # 确保每个像素至少有一个有效时间步
        mask_sum = attention_mask.sum(axis=1)  # (n,) 每个像素被mask的时间步数
        t_total = attention_mask.shape[1]
        fully_masked = mask_sum >= t_total  # 所有时间步都是mask
        if fully_masked.any():
            attention_mask[fully_masked, 0] = 0
        
        # 关键保护：对角线mask会把"自己对自己"的attention也mask掉
        # 如果某像素有效观测数 <= 1，仅有的有效key会被对角线mask掉 → softmax全-1e9 → NaN
        # 解决：有效观测 <= 1 的像素，把 attention_mask 全置0（让模型纯重建）
        valid_count = (attention_mask == 0).sum(axis=1)  # (n,) 每个像素有效时间步数
        too_few_valid = valid_count <= 1
        if too_few_valid.any():
            attention_mask[too_few_valid] = 0  # 全置0，模型纯重建，不依赖原始观测
        
        return s2_input, s2_input_full, s1_input, missing_mask.astype(np.float32), attention_mask.astype(np.float32)
    
    def _inference_block(self, s2_input, s2_input_full, s1_input, missing_mask, attention_mask):
        """对单个block进行推理
        
        Args:
            s2_input: (n, t, b) S2数据
            s1_input: (n, t, b) S1数据
            missing_mask: (n, t) 有效观测掩膜
            attention_mask: (n, t) 注意力掩膜
            
        Returns:
            output: (n, t_out, b) 重建后的数据
        """
        n = s2_input.shape[0]
        
        # batch_size 从配置文件读取；OOM时永久减半，后续batch直接用小值
        batch_size = getattr(self.args, "inference_batch_size", 32768)
        n_batches = (n + batch_size - 1) // batch_size
        _warned_oom = [False]  # 用列表以便内层函数修改
        
        # 预分配输出数组，避免 np.concatenate 导致的内存峰值（拼接时需要2倍内存）
        t_out = len(self.s2_doy)
        b_out = s2_input.shape[-1]
        output = np.empty((n, t_out, b_out), dtype=np.float32)
        
        # 提前把DOY tensor移到GPU，在整个block内复用，不在loop内重复创建+销毁
        s2_doy_1d = torch.from_numpy(self.s2_doy.astype(np.float32)).to(self.device)
        s1_doy_1d = torch.from_numpy(self.s1_doy.astype(np.float32)).to(self.device)
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n)
            cur_batch = end_idx - start_idx
            
            # 获取当前批次数据（numpy切片，零拷贝）
            s2_batch = s2_input[start_idx:end_idx]
            s1_batch = s1_input[start_idx:end_idx]
            missing_mask_batch = missing_mask[start_idx:end_idx]
            attention_mask_batch = attention_mask[start_idx:end_idx]
            
            # 转换为tensor
            # s2_batch 是缺失位置置0后的输入X，s2_full_batch 是完整数据X_holdout
            s2_full_batch = s2_input_full[start_idx:end_idx]
            X = torch.from_numpy(s2_batch.astype(np.float32)).to(self.device)
            X_holdout = torch.from_numpy(s2_full_batch.astype(np.float32)).to(self.device)
            X_aux = torch.from_numpy(s1_batch.astype(np.float32)).to(self.device)
            missing_mask_tensor = torch.from_numpy(missing_mask_batch.astype(np.float32)).to(self.device)
            attention_mask_tensor = torch.from_numpy(attention_mask_batch.astype(np.float32)).to(self.device)
            
            date_input = s2_doy_1d.unsqueeze(0).expand(cur_batch, -1).contiguous()
            dates_aux = s1_doy_1d.unsqueeze(0).expand(cur_batch, -1).contiguous() if self.args.with_X_aux else s1_doy_1d
            date_output = date_input.clone()
            
            inputs = {
                "date_input": date_input,
                "X": X,
                "missing_mask": missing_mask_tensor,
                "X_holdout": X_holdout,
                "indicating_mask": missing_mask_tensor,
                "attention_mask": attention_mask_tensor,
                "X_aux": X_aux,
                "dates_aux": dates_aux,
                "date_output": date_output
            }
            
            # 推理：OOM时自动将batch切成两半重试，直到成功
            with torch.no_grad():
                if i == 0:
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    res   = torch.cuda.memory_reserved()  / 1024**3
                    logger.info(f"  推理前(block首batch): allocated={alloc:.2f}GB reserved={res:.2f}GB")
                
                try:
                    if i == 0 and not hasattr(self, '_input_logged'):
                        self._input_logged = True
                        logger.info(f"  [诊断] date_input值: {inputs['date_input'][0].tolist()}")
                        logger.info(f"  [诊断] attention_mask全1的像素数: {(inputs['attention_mask'].all(dim=1)).sum().item()}")
                        logger.info(f"  [诊断] X的NaN数: {inputs['X'].isnan().sum().item()}")
                        logger.info(f"  [诊断] X_aux的NaN数: {inputs['X_aux'].isnan().sum().item()}")

                    results = self.model(inputs, stage="anytime")
                    if i == 0 and not hasattr(self, '_result_logged'):
                        self._result_logged = True
                        imp = results["imputed_data"]
                        rec = results["reconstructed_data"]
                        nan_count = imp.isnan().sum().item()
                        total = imp.numel()
                        imp_clean = imp.nan_to_num(0)
                        logger.info(f"  [诊断] imputed_data: min={imp_clean.min():.4f} max={imp_clean.max():.4f} mean={imp_clean.mean():.4f} NaN={nan_count}/{total} ({nan_count/total*100:.1f}%)")
                        valid_px = (imp_clean.abs().sum(dim=1).sum(dim=1) > 0).nonzero(as_tuple=True)[0]
                        if len(valid_px) > 0:
                            px = valid_px[0].item()
                            logger.info(f"  [诊断] 有值像素[{px}] t=0: {imp_clean[px,0,:].tolist()}")
                    imputed = results["imputed_data"]
                    rec = results["reconstructed_data"]
                    invalid_imputed = ~torch.isfinite(imputed)
                    if invalid_imputed.any():
                        invalid_rec = ~torch.isfinite(rec)
                        fallback = torch.where(invalid_rec, results["X_holdout"], rec)
                        imputed = torch.where(invalid_imputed, fallback, imputed)
                        logger.warning(
                            f"  [诊断] imputed_data存在非有限值，已用reconstructed/X_holdout回退 "
                            f"({invalid_imputed.sum().item()} / {imputed.numel()})"
                        )
                    output[start_idx:end_idx] = imputed.cpu().numpy()
                    del results
                    inputs = inputs  # keep ref for del below
                except RuntimeError as oom_err:
                    if "out of memory" not in str(oom_err).lower():
                        raise
                    # OOM：永久缩小 batch_size，重新规划剩余 batches
                    del inputs
                    inputs = None
                    torch.cuda.empty_cache()
                    new_bs = batch_size // 2
                    if not _warned_oom[0]:
                        logger.warning(
                            f"  OOM(batch_size={batch_size}): 永久缩小为 {new_bs}，"
                            f"后续所有batch均使用新大小"
                        )
                        _warned_oom[0] = True
                    batch_size = new_bs
                    # 把当前batch用新大小切开重跑
                    for sub_s in range(start_idx, end_idx, batch_size):
                        sub_e = min(sub_s + batch_size, end_idx)
                        sub_len = sub_e - sub_s
                        sub_inputs = {
                            "date_input":      s2_doy_1d.unsqueeze(0).expand(sub_len,-1).contiguous(),
                            "X":               X[sub_s - start_idx : sub_e - start_idx],
                            "missing_mask":    missing_mask_tensor[sub_s - start_idx : sub_e - start_idx],
                            "X_holdout":       X[sub_s - start_idx : sub_e - start_idx],
                            "indicating_mask": missing_mask_tensor[sub_s - start_idx : sub_e - start_idx],
                            "attention_mask":  attention_mask_tensor[sub_s - start_idx : sub_e - start_idx],
                            "X_aux":           X_aux[sub_s - start_idx : sub_e - start_idx],
                            "dates_aux":       s1_doy_1d.unsqueeze(0).expand(sub_len,-1).contiguous() if self.args.with_X_aux else s1_doy_1d,
                            "date_output":     s2_doy_1d.unsqueeze(0).expand(sub_len,-1).contiguous().clone(),
                        }
                        sub_res = self.model(sub_inputs, stage="anytime")
                        sub_imputed = sub_res["imputed_data"]
                        sub_rec = sub_res["reconstructed_data"]
                        sub_invalid_imputed = ~torch.isfinite(sub_imputed)
                        if sub_invalid_imputed.any():
                            sub_invalid_rec = ~torch.isfinite(sub_rec)
                            sub_fallback = torch.where(sub_invalid_rec, sub_res["X_holdout"], sub_rec)
                            sub_imputed = torch.where(sub_invalid_imputed, sub_fallback, sub_imputed)
                        output[sub_s:sub_e] = sub_imputed.cpu().numpy()
                        del sub_inputs, sub_res
                    # 更新后续循环的 n_batches（让 i 跳过已处理的像素）
                    n_batches = (n + batch_size - 1) // batch_size

            del X, X_aux, missing_mask_tensor, attention_mask_tensor
            if inputs is not None:
                del inputs
        
        # 整个block推理结束后统一清理一次
        del s2_doy_1d, s1_doy_1d
        torch.cuda.empty_cache()
        
        return output
    
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
        # 诊断：打印第一个block的输出范围
        if not hasattr(self, '_postprocess_logged'):
            self._postprocess_logged = True
            finite_mask = np.isfinite(output)
            if finite_mask.any():
                finite_vals = output[finite_mask]
                logger.info(
                    f"  [诊断] output原始范围(有限值): "
                    f"min={finite_vals.min():.4f}, max={finite_vals.max():.4f}, mean={finite_vals.mean():.4f}"
                )
            else:
                logger.warning("  [诊断] output全是非有限值(NaN/Inf)")
            logger.info(f"  [诊断] output形状: {output.shape}, scale={self.scale}")
        # NaN兜底：全程被云覆盖的像素模型输出NaN，替换为0
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        output_int16 = np.round(output * self.scale).astype(np.int16)
        if not hasattr(self, '_postprocess_int16_logged'):
            self._postprocess_int16_logged = True
            logger.info(f"  [诊断] int16范围: min={output_int16.min()}, max={output_int16.max()}, mean={output_int16.mean():.2f}")
        
        # Reshape: (n, t_out, b) -> (t_out, b, h, w)
        output_spatial = output_int16.reshape(block_h, block_w, -1, output_int16.shape[-1])
        output_spatial = output_spatial.transpose(2, 3, 0, 1)  # (t_out, b, h, w)
        
        return output_spatial
    
    def _io_worker(self, block_queue, done_event, block_coords):
        """IO预读线程：读取block数据放入队列，与GPU推理并行"""
        for (row_i, col_j) in block_coords:
            block_h = min(self.args.block_size, self.height - row_i)
            block_w = min(self.args.block_size, self.width  - col_j)
            s2_block, s1_block = self._read_time_series_block(col_j, row_i, block_w, block_h)
            s2_input, s2_input_full, s1_input, missing_mask, attention_mask =                 self._preprocess_block(s2_block, s1_block)
            del s2_block, s1_block
            block_queue.put((row_i, col_j, block_h, block_w,
                             s2_input, s2_input_full, s1_input, missing_mask, attention_mask))
        done_event.set()

    def run_inference(self):
        """执行完整推理流程
        优化：预开数据集 + IO/GPU流水线 + 即时写出（不积累全图）
        """
        logger.info("=" * 80)
        logger.info("开始推理...")
        logger.info(f"  Block大小: {self.args.block_size} × {self.args.block_size}")
        logger.info(f"  输出时间点数: {len(self.s2_doy)}")
        logger.info(f"  inference_batch_size: {getattr(self.args, 'inference_batch_size', 32768)}")
        logger.info("=" * 80)

        b_out = self.n_bands
        block_coords = [
            (i, j)
            for i in range(0, self.height, self.args.block_size)
            for j in range(0, self.width,  self.args.block_size)
        ]
        total_blocks = len(block_coords)
        n_blocks_h = int(np.ceil(self.height / self.args.block_size))
        n_blocks_w = int(np.ceil(self.width  / self.args.block_size))
        logger.info(f"总共需要处理 {total_blocks} 个blocks ({n_blocks_h} × {n_blocks_w})")
        
        # 测试模式：只跑前N个block，验证输出正确性
        test_blocks = getattr(self.args, 'test_blocks', 0)
        if test_blocks > 0:
            block_coords = block_coords[:test_blocks]
            total_blocks = len(block_coords)
            logger.info(f"⚠️  测试模式: 只跑前 {total_blocks} 个blocks")

        # 预开所有GDAL数据集（消除每block的open/close IO开销）
        logger.info("预开所有GDAL数据集...")
        self._open_datasets()

        # 预创建所有输出TIF（保持打开，逐block写入，不在内存积累全图）
        check_path(self.args.output_folder)
        driver = gdal.GetDriverByName('GTiff')
        out_ds_dict = {}
        # 用原始日期字符串（YYYYMMDD）作为文件名，避免跨年DOY转换错误
        # s2_dates[i] 对应 s2_doy[i]，直接用 date 整数作为文件名
        for doy, date in zip(self.s2_doy, self.s2_dates):
            out_path = os.path.join(self.args.output_folder,
                                    f"S2_L2A_reconstructed_{date}.tif")
            ds_out = driver.Create(
                out_path, self.width, self.height, b_out, gdal.GDT_Int16,
                options=['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
            )
            ds_out.SetGeoTransform(self.geotransform)
            ds_out.SetProjection(self.projection)
            out_ds_dict[doy] = ds_out
        logger.info(f"✓ 已预创建 {len(out_ds_dict)} 个输出TIF文件")

        # 启动IO预读线程（最多预读3个block，防止内存爆炸）
        block_queue = queue.Queue(maxsize=3)
        done_event  = threading.Event()
        io_thread   = threading.Thread(
            target=self._io_worker,
            args=(block_queue, done_event, block_coords),
            daemon=True
        )
        io_thread.start()
        logger.info("✓ IO预读线程已启动（IO/GPU流水线）")

        # GPU推理主循环
        pbar = tqdm(total=total_blocks, desc="Block推理进度")
        block_idx = 0

        while not (done_event.is_set() and block_queue.empty()):
            try:
                item = block_queue.get(timeout=60)
            except queue.Empty:
                continue

            row_i, col_j, block_h, block_w,                 s2_input, s2_input_full, s1_input, missing_mask, attention_mask = item

            output = self._inference_block(s2_input, s2_input_full, s1_input, missing_mask, attention_mask)
            del s2_input, s2_input_full, s1_input, missing_mask, attention_mask

            output_spatial = self._postprocess_block(output, block_h, block_w)
            del output

            # 即时写入TIF
            for t_idx, doy in enumerate(self.s2_doy):
                for b_idx in range(b_out):
                    out_ds_dict[doy].GetRasterBand(b_idx + 1).WriteArray(
                        output_spatial[t_idx, b_idx], xoff=col_j, yoff=row_i
                    )
            del output_spatial

            block_idx += 1
            pbar.update(1)
            if block_idx % 20 == 0:
                gc.collect()
                pbar.set_postfix({"done": f"{block_idx}/{total_blocks}"})

        pbar.close()
        io_thread.join()

        logger.info("✓ 所有blocks推理完成，正在关闭输出文件...")
        for doy, ds_out in out_ds_dict.items():
            for b_idx in range(1, b_out + 1):
                ds_out.GetRasterBand(b_idx).FlushCache()
            ds_out = None
        self._close_datasets()

        logger.info(f"✓ 已写入 {len(self.s2_doy)} 个TIF文件到: {self.args.output_folder}")
        logger.info("=" * 80)
        logger.info("推理完成！")
        logger.info("=" * 80)


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
    
    # 设置CUDA内存分配策略，减少碎片化
    # max_split_size_mb 限制单次分配的最大块大小，缓解 reserved >> allocated 问题
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:512"
    )
    
    # 设置anytime_ouput为S2的DOY（推理时输出所有S2时间点）
    # 这个参数会在后面根据实际扫描的S2文件动态设置
    
    # 添加其他必需的训练参数（推理时不使用，但模型初始化需要）
    args.artificial_missing_rate = 0.25
    args.gap_mode = "random"
    args.mode = "test_anytime"  # 推理模式
    args.debug_mode = False
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