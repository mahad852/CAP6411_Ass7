from segment_anything import SamPredictor, sam_model_registry
from utils import get_mask_sets_from_segmented_image, compute_iou_between_gt_and_sam
import os
import numpy as np
from PIL import Image

def compute_iou_for_gtea_using_sam(path, predictor: SamPredictor):
    ious = []

    images_dir = os.path.join(path, 'Images')
    masks_dir = os.path.join(path, 'Masks')


    for image_name in os.listdir(images_dir):
        mask_name = image_name.split('.')[0] + '.png'
        masks_path = os.path.join(masks_dir, mask_name)
        image_path = os.path.join(images_dir, image_name)

        image = np.array(Image.open(image_path).convert('RGB'))
        predictor.set_image(image)

        mask_gt = Image.open(masks_path)
        mask_sets = get_mask_sets_from_segmented_image(mask_gt)

 
        for _, label in enumerate(mask_sets.keys()):
            x, y = list(mask_sets[label])[int(len(mask_sets[label])/2)]
            points = np.array([[y, x]])
            masks, _, _ = predictor.predict(
                point_coords=points,
                point_labels=np.array([int(label[1:], 16)]),
                multimask_output=True,
            )
            
            max_iou = 0

            for _, mask in enumerate(masks):
                iou = compute_iou_between_gt_and_sam(mask_gt, mask, label)
                if iou > max_iou:
                    max_iou = iou

            ious.append(max_iou)
            with open('iou_gtea.txt', 'a+') as f:
                f.write('IoU: ' + str(max_iou) + '\n')

    print('mean IoU for leaf dataset:', sum(ious)/len(ious), 'IoUs sum:', sum(ious))
