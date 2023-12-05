
from segment_anything import SamPredictor
import os
import numpy as np
from PIL import Image
from pycocotools import mask as pycoco_mask
import json

def compute_iou_for_ishape_using_sam(path, predictor: SamPredictor):
    ious = []

    for sub_dirs in os.listdir(path):
        if not os.path.isdir(os.path.join(path, sub_dirs)):
            continue

        val_dir = os.path.join(path, sub_dirs, 'val')
        annotations_obj = {}

        with open(os.path.join(val_dir, 'coco_format-mask_encoding=rle-instances_2017.json')) as f:
            annotations_obj = json.load(f)
        
        images = {}

        for img_obj in annotations_obj['images']:
            images[img_obj['id']] = {}
            images[img_obj['id']]['path'] = os.path.join(val_dir, 'image', img_obj['file_name'])
            images[img_obj['id']]['seg'] = []

        for ann_obj in annotations_obj['annotations']:
            point = [ann_obj['bbox'][1] + int(ann_obj['bbox'][3]/2), ann_obj['bbox'][0] + int(ann_obj['bbox'][2]/2)]
            
            images[ann_obj['image_id']]['seg'].append({
                'mask' : pycoco_mask.decode(ann_obj['segmentation']), 
                'point' : point
            })
        
        for img_id in images.keys():
            if not os.path.exists(images[img_id]['path']):
                continue

            image = np.array(Image.open(images[img_id]['path']).convert('RGB'))
            predictor.set_image(image)

            for seg_obj in images[img_id]['seg']:
                mask_gt = seg_obj['mask']
                point = seg_obj['point']

                masks, _, _ = predictor.predict(
                    point_coords=np.array([[point[1], point[0]]]),
                    multimask_output=True,
                    point_labels=np.array([0])
                )

                max_iou = 0
                for mask in masks:
                    intersection = np.logical_and(mask_gt, mask)
                    union = np.logical_or(mask_gt, mask)
                    
                    iou = np.ndarray.flatten(intersection).sum()/np.ndarray.flatten(union).sum()
                    
                    max_iou = iou if iou > max_iou else max_iou

                ious.append(max_iou)

                with open('iou_ishape.txt', 'a+') as f:
                    f.writelines(['IoU: ' + str(max_iou)])
    
    print('mean IoU for leaf dataset:', sum(ious)/len(ious), 'IoUs sum:', sum(ious))