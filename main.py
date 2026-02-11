import cv2
cv2.setNumThreads(0) # Fix Segfault

import numpy as np
import os
import time
import datetime
from threading import Thread, Lock
from ultralytics import YOLO
from config import CameraConfig
from visualizer import Visualizer
from processor import FrameProcessor

# --- CẤU HÌNH ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
MODEL_ITEM_PATH = r"best_ck.pt"
MODEL_SLOT_PATH = r"best.pt"

RTSP_URLS = [
    "rtsp://admin",
    "rtsp://admin:",
    "rtsp://admin:",
    "rtsp://admin:",
]

PROC_W, PROC_H = 640, 480 
DASHBOARD_WIDTH = 350 

class SafeCameraStream:
    def __init__(self, rtsp_url, cam_id):
        self.url = rtsp_url
        self.cam_id = cam_id
        self.frame = None
        self.stopped = False
        self.lock = Lock()
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened(): print(f"❌ Lỗi: {cam_id}")
        else:
            ret, frame = self.cap.read()
            if ret: self.frame = frame

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened(): break
            ret, frame = self.cap.read()
            if ret:
                with self.lock: self.frame = frame
            else:
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url)

    def read(self):
        with self.lock: return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

class SystemFlowManager:
    def __init__(self):
        self.timer_start = None
        self.final_verdict = None 
        self.state = "IDLE" 

    def update(self, configs, cam4_detected):
        if cam4_detected:
            self.state = "RUNNING"
            self.timer_start = None
            self.final_verdict = None
            return None
        else:
            if self.state == "RUNNING":
                self.state = "COUNTDOWN"
                self.timer_start = time.time()
                print("🏁 Cam 4 mất tín hiệu -> Bắt đầu đếm ngược 10s...")
            
            elif self.state == "IDLE": return None

            if self.state == "COUNTDOWN":
                elapsed = time.time() - self.timer_start
                remaining = 10.0 - elapsed
                
                if remaining <= 0:
                    self.state = "SHOW_RESULT"
                    checklist_ok = True
                    for cfg in configs:
                        stats = cfg.get_item_counts()
                        for item_info in stats.values():
                            if not item_info['done']: checklist_ok = False
                    
                    self.final_verdict = "PASS" if checklist_ok else "FAIL"
                    self.timer_start = time.time() 
                    return "FINISHED"
                return remaining 

            elif self.state == "SHOW_RESULT":
                elapsed = time.time() - self.timer_start
                if elapsed > 5.0:
                    print("🔄 Kết thúc hiển thị -> Reset Hệ Thống")
                    return "RESET_NOW"
                return "SHOWING"
        return None

def main():
    if not os.path.exists(MODEL_ITEM_PATH): return
    print(f"🚀 HỆ THỐNG FULL OPTION - RECORDING ENABLED")

    model_items = YOLO(MODEL_ITEM_PATH)
    model_slots = YOLO(MODEL_SLOT_PATH)

    streams = []
    cam_names = ["cam_1", "cam_2", "cam_3", "cam_4"]
    
    print("⏳ Đang khởi tạo Camera...")
    for i, url in enumerate(RTSP_URLS):
        print(f"   -> Cam {i+1}...")
        s = SafeCameraStream(url, cam_names[i]).start()
        streams.append(s)
        time.sleep(0.5)

    configs = [CameraConfig(name) for name in cam_names]
    processors = [FrameProcessor(cfg) for cfg in configs]
    visualizer = Visualizer()
    flow_manager = SystemFlowManager()

    total_w = (PROC_W * 2) + DASHBOARD_WIDTH
    total_h = PROC_H * 2
    main_canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    if not os.path.exists("recordings"): os.makedirs("recordings")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"recordings/session_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_video = cv2.VideoWriter(video_filename, fourcc, 20.0, (total_w, total_h))
    
    print(f"🎥 Đang ghi hình: {video_filename}")

    try:
        while True:
            batch_frames = []
            valid_indices = []
            
            for i, stream in enumerate(streams):
                frame = stream.read()
                if frame is not None:
                    try:
                        resized = cv2.resize(frame, (PROC_W, PROC_H))
                        batch_frames.append(resized)
                        valid_indices.append(i)
                    except: batch_frames.append(np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8))
                else:
                    black = np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8)
                    cv2.putText(black, "NO SIGNAL", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    batch_frames.append(black)

            if batch_frames:
                res_slots = model_slots.predict(batch_frames, conf=0.5, verbose=False, stream=False)
                res_items = model_items.predict(batch_frames, conf=0.45, verbose=False, stream=False)

            cam4_detected = False

            for i in range(4):
                dx, dy = (i % 2) * PROC_W, (i // 2) * PROC_H
                roi = main_canvas[dy:dy+PROC_H, dx:dx+PROC_W]
                np.copyto(roi, batch_frames[i])

                if i in valid_indices:
                    detected = processors[i].process(res_slots[i], res_items[i])
                    if i == 3: cam4_detected = detected 

                    for slot in configs[i].slots.values():
                        visualizer.draw_slot_obb(roi, slot)
                    if res_items[i].boxes:
                        for b, c, cl in zip(res_items[i].boxes.xyxy.cpu().numpy(), res_items[i].boxes.conf.cpu().numpy(), res_items[i].boxes.cls.cpu().numpy()):
                            visualizer.draw_item_box(roi, b, res_items[i].names[int(cl)], c)
                    visualizer.draw_camera_info(roi, configs[i])

            status = flow_manager.update(configs, cam4_detected)
            
            if status == "RESET_NOW":
                for cfg in configs: cfg.force_reset()
                flow_manager.state = "IDLE"
            
            blink = int(time.time() * 4) % 2 == 0
            
            if flow_manager.state == "COUNTDOWN" and isinstance(status, float):
                cv2.putText(main_canvas, f"FINAL CHECK: {status:.1f}s", (PROC_W+50, PROC_H+100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

            elif flow_manager.state == "SHOW_RESULT":
                if flow_manager.final_verdict == "PASS" and blink:
                    cv2.putText(main_canvas, "OKE - DONE", (total_w//2-200, total_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10)
                    cv2.rectangle(main_canvas, (0,0), (total_w, total_h), (0,255,0), 20)
                
                elif flow_manager.final_verdict == "FAIL" and blink:
                    cv2.putText(main_canvas, "WRONG / MISSING", (total_w//2-350, total_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
                    cv2.rectangle(main_canvas, (0,0), (total_w, total_h), (0,0,255), 20)

            dashboard_roi = main_canvas[:, -DASHBOARD_WIDTH:]
            dashboard_roi[:] = (20, 20, 20)
            if flow_manager.state == "SHOW_RESULT" and flow_manager.final_verdict == "FAIL" and blink:
                dashboard_roi[:] = (0, 0, 100)
                
            visualizer.draw_dashboard_on_roi(dashboard_roi, configs)
            visualizer.draw_fps(main_canvas)

            if out_video is not None:
                out_video.write(main_canvas)

            cv2.imshow("Smart Packing System", main_canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        print("🛑 Đang dừng hệ thống...")
        for s in streams: s.stop()
        if 'out_video' in locals() and out_video is not None: out_video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
