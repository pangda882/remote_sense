# -*- coding: utf-8 -*-
"""
shape_logger.py
----------------------------------
一个轻量级的 PyTorch tensor 形状记录模块。
支持分布式环境自动识别主进程，仅主进程写文件。

使用方式：
    from shape_logger import log_shape

    log_shape("input_tensor", x)
"""

import os
import time
import torch
import logging

# ---------------- 基础 logger（先放个 NullHandler，避免未初始化时告警） ----------------
logger = logging.getLogger("shape_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# ---------------- 分布式主进程判断 ----------------
def _is_main_process():
    r = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    try:
        return r is None or int(r) == 0
    except Exception:
        return True

# ---------------- 懒初始化：第一次真正写日志时才创建目录/文件 ----------------
_shape_logger_ready = False

def _ensure_shape_logger():
    global _shape_logger_ready
    if _shape_logger_ready:
        return

    if not _is_main_process():
        # 非主进程不写文件
        _shape_logger_ready = True
        return

    # 日志目录：优先环境变量 SHAPE_LOG_DIR，否则 ../../logs
    default_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../logs"))
    log_dir = os.environ.get("SHAPE_LOG_DIR", default_dir)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"shape_debug_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # 清空旧 handler
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _shape_logger_ready = True
    logger.info(f"Shape logger initialized: {log_path}")

# ---------------- 主函数 ----------------
_logged_shapes = set()

def log_shape(name, tensor,prefix=""):
    """记录 tensor 的 shape/dtype/device（只主进程执行一次）"""
    _ensure_shape_logger()
    if not _is_main_process():
        return

    if isinstance(tensor, torch.Tensor):
        key = (name, tuple(tensor.shape))
        if key in _logged_shapes:
            return
        _logged_shapes.add(key)
        logger.info(f"{prefix}/{name}: shape={tuple(tensor.shape)}")
    elif isinstance(tensor, (list, tuple)):
        for i, t in enumerate(tensor):
            log_shape(f"{name}[{i}]", t)
    else:
        logger.info(f"{name}: type={type(tensor)}")
