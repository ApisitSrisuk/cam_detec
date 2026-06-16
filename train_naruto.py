from ultralytics import YOLO
import os

def main():
    # 1. ระบุพิกัดโฟลเดอร์รูปภาพที่เราทำไว้ (naruto_dataset) ให้เป็น Absolute Path ป้องกันปัญหาหาไฟล์ไม่เจอ
    dataset_path = os.path.abspath("naruto_dataset")
    
    print("==================================================")
    print(f"🚀 กำลังเริ่มต้นเทรนโมเดลจำแนกท่าประสานอินนารูโตะ")
    print(f"📂 คลังข้อมูลรูปภาพ: {dataset_path}")
    print("==================================================")
    
    # 2. โหลดโมเดลตั้งต้นของ YOLO สำหรับงานจำแนกภาพ (Classification)
    # ตัว 'yolov8n-cls.pt' เป็นโมเดลขนาดเล็ก (Nano) รันได้เร็วและเสถียรบน Python 3.14
    model = YOLO("yolov8n-cls.pt")  

    # 3. สั่งเริ่มกระบวนการเทรน (Training Process)
    # - data: โฟลเดอร์รูปภาพที่จัดหมวดหมู่แยกตามชื่อท่าแล้ว
    # - epochs: จำนวนรอบที่ต้องการให้ AI วิ่งเรียนรู้ซ้ำๆ (แนะนำ 30 รอบ กำลังดีสำหรับจุดเริ่มต้น)
    # - imgsz: ขนาดภาพที่ส่งเข้าโมเดล (มาตรฐานระบบคือ 640x640 พิกเซล)
    # - workers: จำนวน Thread ในการดึงข้อมูล (ปรับเป็น 2 เพื่อเซฟทรัพยากรบน Windows)
    model.train(
        data=dataset_path, 
        epochs=30, 
        imgsz=640,
        workers=2
    )
    
    print("\n==============================================")
    print(" 🎉 เทรนโมเดลสำเร็จเสร็จสิ้น! ")
    print(" 💾 สมอง AI ชิ้นเอกของคุณถูกเซฟไว้ที่พิกัด:")
    print("    runs/classify/train/weights/best.pt")
    print("==============================================")

if __name__ == "__main__":
    main()