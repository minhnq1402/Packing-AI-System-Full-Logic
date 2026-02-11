import cv2
import numpy as np

class SlotRecovery:
    def __init__(self):
        # Lưu tọa độ 5 slot chuẩn (Reference)
        # Dạng: {1: [x,y], ..., 5: [x,y]}
        self.ref_slots = None 

    def update_reference(self, current_slots):
        self.ref_slots = current_slots.copy()

    def recover(self, detected_slots):
        if self.ref_slots is None: return detected_slots

        # Tìm điểm chung
        common_ids = set(detected_slots.keys()) & set(self.ref_slots.keys())
        if len(common_ids) < 2: return detected_slots 

        src_pts = [] 
        dst_pts = []
        for sid in common_ids:
            src_pts.append(self.ref_slots[sid])
            dst_pts.append(detected_slots[sid])

        src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)

        # Tính ma trận biến đổi
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None: return detected_slots

        # Khôi phục điểm thiếu
        recovered_slots = detected_slots.copy()
        missing_ids = set(self.ref_slots.keys()) - set(detected_slots.keys())

        if not missing_ids: return recovered_slots

        missing_pts_ref = []
        for mid in missing_ids:
            missing_pts_ref.append(self.ref_slots[mid])
        
        missing_pts_ref = np.array(missing_pts_ref, dtype=np.float32).reshape(-1, 1, 2)
        recovered_pts = cv2.transform(missing_pts_ref, M)

        for i, mid in enumerate(missing_ids):
            pos = recovered_pts[i][0]
            recovered_slots[mid] = pos.astype(int)

        return recovered_slots