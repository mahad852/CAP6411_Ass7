from segment_anything import SamPredictor
from utils import compute_iou_between_gt_and_sam
from utils import get_image_center_coords
import os
import numpy as np
from PIL import Image

def compute_iou_for_leaf_using_sam(path, predictor: SamPredictor):
    leaves = os.path.join(path, 'Ara2012')
    tobaco = os.path.join(path, 'Tobaco')

    ious = []

    def get_file_name(is_leaves, image_num):
        prefix = 'ara2012_plant' if is_leaves else 'tobaco_plant'
        num_str = '0' * (3 - len(str(image_num))) + str(image_num)
        
        return prefix + num_str
        
    for i in range(1, 121):
        image_name = get_file_name(True, i)

        image_path = os.path.join(leaves, image_name + '_rgb.png')
        masks_path = os.path.join(leaves, image_name + '_label.png')
        centers_path = os.path.join(leaves, image_name +'_centers.png')

        if not os.path.exists(image_path):
            continue
        
        image = np.array(Image.open(image_path).convert('RGB'))
        predictor.set_image(image)

        mask_gt = Image.open(masks_path)

        centers_image = Image.open(centers_path)
        center_coords = get_image_center_coords(centers_image, mask_gt)

        for ((x, y), label) in center_coords:
            points = np.array([
                [y, x]
            ])
            masks, _, _ = predictor.predict(
                point_coords=points,
                point_labels=np.array([int(label[1:], 16)]),
                multimask_output=True,
            )

            ious.append(max(list(map(lambda mask: compute_iou_between_gt_and_sam(mask_gt, mask, label), masks))))  
    print('total mean IoU:', sum(ious)/len(ious), 'total IoU:', sum(ious))
