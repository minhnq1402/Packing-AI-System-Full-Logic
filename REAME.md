# 📦 Smart Packing System (Hệ thống Kiểm Tra Đóng Gói Thông Minh)

Hệ thống giám sát và kiểm tra quy trình đóng gói công nghiệp thời gian thực (Real-time QA/QC) sử dụng 4 Camera IP, trí tuệ nhân tạo (YOLOv8) và các thuật toán thị giác máy tính nâng cao.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-yellow)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green)
![Hardware](https://img.shields.io/badge/Hardware-RTX3090-red)

## 📖 Giới thiệu

Dự án này được thiết kế để chạy trên Server (Linux/Windows) với GPU mạnh mẽ (RTX 3090), giúp kiểm soát chất lượng đóng gói theo dây chuyền thủ công. Hệ thống đảm bảo:
1.  **Đúng vật phẩm:** Nhận diện chính xác từng loại linh kiện (Đèn, Bo mạch, Dây, Túi, Sạc...).
2.  **Đúng vị trí (Slot):** Đặt đúng vào khay định hình.
3.  **Đúng quy cách:** Kiểm tra độ sâu (Depth Check) dựa trên diện tích để phát hiện vật chưa ấn chặt/bị nổi.
4.  **Đúng quy trình:** Cảnh báo nếu vật của công đoạn sau xuất hiện ở công đoạn trước.

## ✨ Tính năng nổi bật

* **Multi-Stream Processing:** Xử lý song song 4 luồng Camera RTSP với cơ chế `Thread-Safe` và `Locking` để ngăn chặn xung đột bộ nhớ & Segmentation Fault.
* **Dual AI Models:** Kết hợp 2 model YOLOv8:
    * `best_slot.pt`: Nhận diện vị trí khay/lỗ (OBB).
    * `best_ck.pt`: Nhận diện vật phẩm (Item).
* **Slot Recovery & Geometry:** Thuật toán khôi phục vị trí Slot khi bị tay che khuất dựa trên Ma trận biến đổi (Affine Transformation).
* **Smart Latching Logic:**
    * **3-Second Rule:** Vật đặt đúng 3 giây sẽ tự động **SAVE** trạng thái vào Checklist.
    * **Independent State:** Dù sau đó vật bị che hoặc lấy ra, Dashboard vẫn ghi nhận đã hoàn thành.
* **Area/Size Validation:** Tính năng loại bỏ trường hợp vật "nổi" (chưa lắp khít) bằng cách so sánh ngưỡng diện tích (Area Threshold).
* **Workflow Manager:**
    * Tự động phát hiện chuyển tầng (Reset sau 15s nếu mất tín hiệu).
    * Đếm ngược **10s** chốt đơn khi quy trình hoàn tất (Trigger bởi Cam 4).
* **Auto Recording:** Tự động quay màn hình và lưu video `.mp4` vào thư mục `recordings/` với tên file theo thời gian thực.

## 🛠️ Cài đặt & Yêu cầu

### Phần cứng
* PC/Server có GPU NVIDIA (VRAM > 6GB).
* 4 Camera IP hỗ trợ giao thức RTSP.

### Cài đặt thư viện
```bash
pip install opencv-python numpy ultralytics
⚙️ Cấu hình
Toàn bộ cấu hình nằm trong file config.py và main.py:

Chỉnh sửa RTSP URL trong main.py:

Python
RTSP_URLS = [
    "rtsp://admin:pass@ip_cam1...",
    "rtsp://admin:pass@ip_cam2...",
    ...
]
Cấu hình Ngưỡng diện tích (Size Check) trong config.py:

Python
ITEM_SIZE_LIMITS = {
    "Den_nho": 6000,   # Nếu diện tích > 6000 -> Báo lỗi vật đang nổi
    "Board": 15000,
    ...
}
Quy tắc đóng gói trong config.py:

Python
PACKING_RULES = {
    "cam_1": {1: "Den_nho", 2: "Den_nho"},
    ...
}
🚀 Chạy chương trình
Bash
python3 main.py
📂 Cấu trúc thư mục
Plaintext
packing_system/
│
├── main.py             # Chương trình chính (Camera, Threading, Logic Flow)
├── config.py           # Cấu hình luật, timer 3s, area limit
├── processor.py        # Xử lý logic AI, check size, check vật lạ
├── visualizer.py       # Giao diện: Vẽ Box, Dashboard, Đếm ngược
├── slot_recovery.py    # Thuật toán khôi phục slot bị che
├── utils.py            # Hàm hình học (IoU, Sort)
│
├── recordings/         # Video lưu tự động (Tự tạo khi chạy)
└── README.md           # Tài liệu này
⚠️ Lưu ý vận hành
Hệ thống sử dụng cv2.setNumThreads(0) để tránh xung đột giữa OpenCV và PyTorch trên Linux. Không xóa dòng này.

Để dừng chương trình an toàn và lưu video, nhấn phím 'q' trên cửa sổ hiển thị.

## 🎥 Demo Hoạt Động

Dưới đây là video quay lại quá trình hệ thống hoạt động thực tế với 4 Camera:

https://github.com/user/project/assets/12345678/video-id-goc.mp4

_Video demo quy trình đóng gói, kiểm tra lỗi và tự động reset._