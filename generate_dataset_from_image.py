import cv2
import os
import numpy as np
from ultralytics import YOLO

# --- 1. ตัดแบ่งภาพตาราง 12 ช่องอัตโนมัติ ---
CHART_FILE = "naruto_chart.webp"
OUTPUT_DIR = "naruto_dataset"

# รายชื่อท่าเรียงตามลำดับ แถวละ 3 ท่า (จากซ้ายไปขวา บนลงล่าง)
SEAL_ORDER = [
    "Snake",  "Rat",    "Sheep",
    "Monkey", "Ox",     "Horse",
    "Dragon", "Dog",    "Boar",
    "Bird",   "Rabbit", "Tiger"
]

if not os.path.exists(CHART_FILE):
    print(f"❌ ไม่พบไฟล์ภาพ {CHART_FILE} กรุณาเซฟรูปและตั้งชื่อให้ถูกต้องก่อนรันสคริปต์นี้")
    exit()

img = cv2.imread(CHART_FILE)
img_h, img_w, _ = img.shape

# คำนวณขนาดบล็อกแต่ละช่อง (ตาราง 4 แถว 3 คอลัมน์)
rows = 4
cols = 3
block_h = img_h // rows
block_w = img_w // cols

print("✂️ กำลังตัดแบ่งรูปภาพและทำ Data Augmentation สำหรับ Python 3.14...")

#ฟังก์ชันจำลองเพิ่มมิติภาพ (Augmentation)
def augment_image(sub_img, save_dir, prefix, count=100):
    for i in range(count):
        # สุ่มหมุนภาพเล็กน้อย (-15 ถึง 15 องศา)
        angle = np.random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((block_w/2, block_h/2), angle, 1.0)
        aug_img = cv2.warpAffine(sub_img, M, (block_w, block_h), borderValue=(255,255,255))
        
        # สุ่มปรับความสว่าง/มืด มืดลง 20% หรือสว่างขึ้น 20% (เลียนแบบแสงในห้องกล้อง)
        value = np.random.randint(-40, 40)
        aug_img = cv2.addWeighted(aug_img, 1.0, np.zeros(aug_img.shape, aug_img.dtype), 0, value)
        
        # สุ่มเบลอภาพเล็กน้อย (เลียนแบบ Motion Blur ตอนเคลื่อนไหวมือ)
        if np.random.rand() > 0.5:
            aug_img = cv2.GaussianBlur(aug_img, (3, 3), 0)
            
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_{i}.jpg"), aug_img)

idx = 0
for r in range(rows):
    for c in range(cols):
        seal_name = SEAL_ORDER[idx]
        seal_dir = os.path.join(OUTPUT_DIR, seal_name)
        os.makedirs(seal_dir, exist_ok=True)
        
        # พิกัดสำหรับครอปภาพแต่ละช่อง
        y1 = r * block_h
        y2 = (r + 1) * block_h
        x1 = c * block_w
        x2 = (c + 1) * block_w
        
        cropped = img[y1:y2, x1:x2]
        
        # สร้างภาพจำลองขึ้นมาท่าละ 120 รูป
        augment_image(cropped, seal_dir, seal_name, count=120)
        idx += 1

print("==================================================")
print(" ✅ สร้าง Dataset จำลองสำเร็จเสร็จสิ้น!")
print(" รูปภาพทั้งหมดถูกจัดเตรียมไว้ในโฟลเดอร์: naruto_dataset")
print(" ขั้นตอนต่อไป: สามารถรันไฟล์ `train_naruto.py` เพื่อใช้เทรนต่อได้ทันทีครับ")
print("==================================================")