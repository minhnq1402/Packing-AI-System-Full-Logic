import numpy as np
import cv2

class GeometryUtils:
    def is_item_in_slot(self, item_box, slot_obb, threshold=0.45):
        """
        Kiểm tra item_box có nằm trong slot_obb không bằng IoU
        """
        # Tạo mask cho Slot
        mask_slot = np.zeros((1000, 1000), dtype=np.uint8) # Kích thước giả lập đủ lớn
        # Scale về tọa độ tương đối hoặc dùng polygon
        # Cách đơn giản: Tính giao nhau của 2 hình chữ nhật (vì item là box thẳng)
        
        # Chuyển slot_obb thành Rect bao quanh để tính nhanh
        x,y,w,h = cv2.boundingRect(slot_obb)
        slot_rect = [x, y, x+w, y+h]
        item_rect = [int(item_box[0]), int(item_box[1]), int(item_box[2]), int(item_box[3])]
        
        # Tính Intersection
        xA = max(slot_rect[0], item_rect[0])
        yA = max(slot_rect[1], item_rect[1])
        xB = min(slot_rect[2], item_rect[2])
        yB = min(slot_rect[3], item_rect[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        itemArea = (item_rect[2] - item_rect[0] + 1) * (item_rect[3] - item_rect[1] + 1)
        
        # Tỷ lệ diện tích giao / diện tích item
        ratio = interArea / float(itemArea + 1e-6)
        return ratio > threshold

    def identify_slots_logic(self, slot_centers):
        """
        Sắp xếp các slot tìm thấy để gán ID (S1..S5)
        Logic đơn giản: Sắp xếp theo Y rồi theo X
        """
        if not slot_centers: return {}
        
        # Chuyển về list có index ban đầu
        centers_with_idx = []
        for i, c in enumerate(slot_centers):
            centers_with_idx.append([c[0], c[1], i]) # x, y, original_index
            
        # Sắp xếp theo Y tăng dần (trên xuống dưới), sau đó X tăng dần (trái qua phải)
        # Tùy thực tế camera xoay ngang/dọc mà chỉnh lại logic sort này
        centers_with_idx.sort(key=lambda k: (k[1], k[0])) 
        
        result = {}
        # Gán ID từ 1 đến N
        for new_id, val in enumerate(centers_with_idx, 1):
            original_index = val[2]
            # Trả về: {ID_Local: Tọa độ tâm}
            result[new_id] = np.array([val[0], val[1]], dtype=int)
            
        return result