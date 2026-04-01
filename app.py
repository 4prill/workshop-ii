from flask import Flask, render_template, Response, request
from evaluate_web import generate_frames
from ultralytics import YOLO
import socket

app = Flask(__name__)

# โหลดโมเดล 1 ครั้งเมื่อ Start Server (ลดเวลาโหลดซ้ำ)
print("Loading YOLOv8 Pose model...")
yolo_model = YOLO("yolov8n-pose.pt")
print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    try:
        weight_kg = float(request.args.get('weight', 15.0))
        user_h_cm = float(request.args.get('height', 170.0))
        duration_mode = int(request.args.get('duration_mode', 0))
        
        dur_str_map = {0: "< 1 hr", 1: "1-2 hrs", 2: "> 2 hrs"}
        duration_str = dur_str_map.get(duration_mode, "< 1 hr")
        
        return Response(generate_frames(
            source=0, 
            user_h_cm=user_h_cm, 
            weight_kg=weight_kg, 
            duration_mode=duration_mode, 
            duration_str=duration_str,
            yolo_model=yolo_model
        ), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error starting video feed: {e}")
        return "Video Feed Error", 500

if __name__ == '__main__':
    # อัตโนมัติค้นหา IP Address ของเครื่องนี้บนระบบ LAN เพื่อให้แสดงผลลัพธ์ URL ที่คลิก/ก็อปปี้ได้ง่าย
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*60)
    print("🚀 ระบบประเมิน Lifting Assessment พร้อมให้บริการแล้ว!")
    print(f"👉 สำหรับเข้าใช้งานในเครื่องนี้ (Local): http://127.0.0.1:5000")
    print(f"📱 สำหรับเข้าใช้งานจากมือถือ/แท็บเล็ต (LAN): http://{local_ip}:5000")
    print("="*60 + "\n")
    
    # รันเซิร์ฟเวอร์บน 0.0.0.0 เพื่อให้เครื่องอื่นในเครือข่ายเข้าถึงได้
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
