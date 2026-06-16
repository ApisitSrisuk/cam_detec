import cv2
from ultralytics import YOLO

# 1. โหลดโมเดลที่เราพึ่งเทรนเสร็จสดๆ ร้อนๆ
# ระบบจะชี้ไปที่โฟลเดอร์ runs ที่เกิดขึ้นหลังจากรัน train_naruto.py
model_path = "runs/classify/train/weights/best.pt"
model = YOLO(model_path)

# 2. เปิดใช้งานกล้อง WebCam
cap = cv2.VideoCapture(0)

print("==================================================")
print(" 🔮 ระบบตรวจจับคาถาประสานอินนารูโตะ Real-time เริ่มทำงาน!")
print(" -> ลองทำท่าตามรูป naruto_chart.webp หน้ากล้องได้เลย")
print(" -> กดปุ่ม 'ESC' บนคีย์บอร์ดเพื่อปิดโปรแกรม")
print("==================================================")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("ไม่สามารถเปิดกล้องได้")
        break

    # พลิกภาพให้เหมือนกระจกเงา จะได้ไม่งงเวลาขยับมือ
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ส่งภาพจากกล้องไปให้โมเดลทายผล (ปิด verbose ไม่ให้ log บรรทัดเยอะเกินไป)
    results = model(frame, verbose=False)
    
    for r in results:
        probs = r.probs
        top_class_index = probs.top1
        top_class_name = r.names[top_class_index]
        confidence = float(probs.top1conf)

        # --- แสดงผลเมื่อโมเดลมั่นใจมากกว่า 65% ---
        if confidence > 0.65:
            # กำหนดข้อความผลลัพธ์
            display_text = f"Seal: {top_class_name} ({confidence*100:.1f}%)"
            
            # วาดพื้นหลังกล่องข้อความเพื่อให้เห็นชัดๆ
            cv2.rectangle(frame, (20, 30), (500, 100), (0, 0, 0), -1)
            # แสดงชื่อท่าประสานอินตัวใหญ่ๆ สีส้มคาถานารูโตะ
            cv2.putText(frame, display_text, (40, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 140, 255), 3)
            
            # 💡 [ไอเดีย Backend/Automation ต่อยอด]
            # คุณสามารถเอาตัวแปร top_class_name ไปเช็คเงื่อนไขทำระบบคอมโบคาถาต่อได้เลยครับ
            # เช่น if top_class_name == "Tiger": พ่นไฟออกหน้าจอ

    # แสดงหน้าต่างกล้อง
    cv2.imshow('Naruto Hand Seals Real-time Inference', frame)

    # กดปุ่ม ESC เพื่อปิดโปรแกรม
    if cv2.waitKey(1) & 0xFF == 27:
        break

# คืนหน่วยความจำกล้อง
cap.release()
cv2.destroyAllWindows()