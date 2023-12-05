
from segment_anything import SamPredictor
import os
import numpy as np
from PIL import Image
import cv2
from pycocotools.coco import COCO
import json

def get_mask(coco: COCO, ann_id):
    return coco.annToMask(coco.loadAnns(ann_id)[0])

def compute_iou_for_woodscape_using_sam(path, predictor: SamPredictor):
    ious = []

    val_dir = os.path.join(path, 'rgb_images')
    instances_dir = os.path.join(path, 'instance_annotations')
    
    image_map = {}

    for img_fname in os.listdir(val_dir):
        image_map[img_fname] = os.path.join(instances_dir, img_fname.split('.')[0] + '.json')
    
    for img_fname in image_map.keys():
        img_path = os.path.join(val_dir, img_fname)

        image = np.array(Image.open(img_path).convert('RGB'))

        predictor.set_image(image)
        
        with open(image_map[img_fname]) as f:
            annotations = json.load(f)

        annotations = annotations[img_fname.split('.')[0] + '.json']['annotation']
        
        for ann_i, ann_obj in enumerate(annotations):
            polygon = ann_obj['segmentation']
            mask_gt = create_mask_from_polygon(polygon, image.shape[:-1])
            
            points = np.transpose((mask_gt == 1).nonzero())
            point = points[int(len(points)/2)]
            point = [point[1], point[0]]

            masks, _, _ = predictor.predict(
                point_coords=np.array([point]),
                multimask_output=True,
                point_labels=np.array([0])
            )

            mask_gt = mask_gt == 1
            
            max_iou = 0
            for mask in masks:
                intersection = np.logical_and(mask_gt, mask)
                union = np.logical_or(mask_gt, mask)
                
                iou = np.ndarray.flatten(intersection).sum()/np.ndarray.flatten(union).sum()
                
                max_iou = iou if iou > max_iou else max_iou

            ious.append(max_iou)
            with open('iou_woodscape.txt', 'a+') as f:
                f.writelines(['IoU: ' + str(max_iou)])
    
    print('mean IoU for woodscape dataset:', sum(ious)/len(ious), 'IoUs sum:', sum(ious))

def create_mask_from_polygon(segmentation, image_shape):
    # Parse the coordinates into a format suitable for OpenCV
    polygon = np.array(segmentation, np.int32).reshape((-1, 1, 2))
    
    # Create a blank mask with the same dimensions as the image
    mask = np.zeros(image_shape, np.uint8)
    
    # Draw and fill the polygon on the mask
    cv2.fillPoly(mask, [polygon], 1)
    
    return mask