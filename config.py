"""
config.py
=========
ตั้งค่า Path และ Parameters สำหรับ Hippocampus Preprocessing Pipeline
ผู้ใช้แก้ไขเฉพาะไฟล์นี้ก่อนรัน
"""

import os

# =============================================================================
# [จำเป็น] ตั้งค่า Path ไปยัง FastSurfer
# =============================================================================
# ชี้ไปยังโฟลเดอร์ที่มี FastSurferCNN/ อยู่ข้างใน
FASTSURFER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FastSurfer")

# =============================================================================
# [ไม่ต้องแก้] Output Directory
# =============================================================================
# ผลลัพธ์สุดท้าย (binary mask พร้อมเข้า SPHARM-PDM)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputpreprocess")

# ผลลัพธ์ระหว่างทาง (FastSurfer segmentation output)
FASTSURFER_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastsurfer_temp")

# =============================================================================
# [ไม่ต้องแก้] Preprocessing Parameters
# =============================================================================
# โหมดการสกัด: strict, moderate, generous, smooth
# แนะนำ "moderate" สำหรับ SPHARM-PDM
EXTRACTION_MODE = "moderate"

# จำนวนรอบ closing (สำหรับ moderate mode)
CLOSE_ITERATIONS = 1

# =============================================================================
# [ไม่ต้องแก้] FastSurfer Parameters
# =============================================================================
# ขนาด voxel: "min" = ใช้ความละเอียดต่ำสุดของภาพต้นฉบับ
VOX_SIZE = "min"

# Batch size สำหรับ CNN inference
BATCH_SIZE = 1

# Device: "auto" = ใช้ GPU ถ้ามี, "cpu" = บังคับใช้ CPU
DEVICE = "auto"

# จำนวน threads
THREADS = 1
