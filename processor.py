import numpy as np
from slot_recovery import SlotRecovery
from utils import GeometryUtils

class FrameProcessor:
    def __init__(self, cam_config, conf_threshold=0.5):
        self.cam_config = cam_config
        self.conf_threshold = conf_threshold
        self.recovery = SlotRecovery()
        self.geo_utils = GeometryUtils()

    def process(self, results_slot, results_item):
        """
        Xử lý logic 1 camera: Slot -> Check Item -> Check Size -> Forbidden Item
        """
        # --- BƯỚC 1: XỬ LÝ SLOT (Tìm khay) ---
        slots_found = []
        slot_centers = []
        
        if hasattr(results_slot, 'obb') and results_slot.obb is not None:
            obbs = results_slot.obb.xyxyxyxy.cpu().numpy()
            confs = results_slot.obb.conf.cpu().numpy()
            for obb, conf in zip(obbs, confs):
                if conf < self.conf_threshold: continue
                center = np.mean(obb, axis=0)
                slot_centers.append(center)
                slots_found.append(obb)

        is_tray_detected = len(slot_centers) >= 3

        # --- BƯỚC 2: ĐỊNH DANH & CẬP NHẬT VỊ TRÍ ---
        geometry_ids = self.geo_utils.identify_slots_logic(slot_centers)
        if geometry_ids:
            if len(geometry_ids) == 5:
                self.recovery.update_reference(geometry_ids)
            elif len(geometry_ids) >= 2:
                geometry_ids = self.recovery.recover(geometry_ids)

            for local_id, center_pos in geometry_ids.items():
                slot_obj = self.cam_config.get_slot_by_local_id(local_id)
                if slot_obj:
                    if slots_found:
                        closest_obb = min(slots_found, key=lambda obb: np.linalg.norm(np.mean(obb, axis=0) - center_pos))
                        slot_obj.update_position(closest_obb, center_pos)

        # --- BƯỚC 3: XỬ LÝ ITEM & DIỆN TÍCH ---
        items_boxes = []
        items_classes = []
        items_areas = [] 
        
        self.cam_config.forbidden_item_detected = None 

        if hasattr(results_item, 'boxes') and results_item.boxes:
            boxes = results_item.boxes.xyxy.cpu().numpy()
            confs = results_item.boxes.conf.cpu().numpy()
            clss = results_item.boxes.cls.cpu().numpy()
            
            detected_classes_set = set()

            for box, conf, cls in zip(boxes, confs, clss):
                if conf < 0.45: continue 
                
                cls_name = results_item.names[int(cls)]
                
                # Tính diện tích (W * H)
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                
                items_boxes.append(box)
                items_classes.append(cls_name)
                items_areas.append(area)
                detected_classes_set.add(cls_name)

            # Check vật sai quy trình
            if is_tray_detected:
                forbidden_items = detected_classes_set - self.cam_config.allowed_classes
                if forbidden_items:
                    self.cam_config.forbidden_item_detected = list(forbidden_items)[0]
        
        # --- BƯỚC 4: CHECK VA CHẠM & KÍCH THƯỚC ---
        for s_id, slot in self.cam_config.slots.items():
            if slot.obb_points is None: continue 

            is_occupied = False
            for box, cls_name, area in zip(items_boxes, items_classes, items_areas):
                if self.geo_utils.is_item_in_slot(box, slot.obb_points, threshold=0.45):
                    
                    # Check Size (Nổi hay chìm)
                    if not self.cam_config.is_size_valid(cls_name, area):
                        # print(f"⚠️ S{s_id} NỔI: {cls_name} Area={int(area)}") # Uncomment để debug
                        continue 

                    is_occupied = True
                    if cls_name == slot.expected_item:
                        slot.set_state("oke", cls_name)
                    else:
                        slot.set_state("wrong", cls_name)
                    break 
            
            if not is_occupied:
                slot.set_state("empty")

        # --- BƯỚC 5: UPDATE TRẠNG THÁI ---
        self.cam_config.update_camera_state()
        
        return is_tray_detected