import numpy as np
import cv2

def unrotate_image(image):
    mask = (~np.isnan(image)).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found.")

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.array(box)

    def edge_angle(p1, p2):
        delta = p2 - p1
        return np.degrees(np.arctan2(delta[1], delta[0]))

    angles = []
    for i in range(4):
        angle = edge_angle(box[i], box[(i + 1) % 4])
        length = np.linalg.norm(box[i] - box[(i + 1) % 4])
        angles.append((angle, length))

    angle, length = max(angles, key=lambda x: x[1])
    
    if angle >= 2:
    
        h, w = image.shape
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        image_filled = np.nan_to_num(image, nan=0)
        mask = (~np.isnan(image)).astype(np.uint8)

        rotated_img = cv2.warpAffine(image_filled, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        rotated_mask = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST)

        rotated_img[rotated_mask == 0] = np.nan

        return rotated_img, angle
    
    else:
        return image, angle