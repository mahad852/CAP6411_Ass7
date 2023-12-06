
from segment_anything import SamPredictor
import os
import numpy as np
from PIL import Image
from pycocotools import mask
from pycocotools.coco import COCO

def get_image_map(coco: COCO):
    annsIds = coco.getAnnIds()
    images_map = {}
    
    for ann in annsIds:
        img_file_name = coco.loadImgs(coco.loadAnns(ann)[0]['image_id'])[0]['file_name']
        point = [coco.loadAnns(ann)[0]['bbox'][1] + int(coco.loadAnns(ann)[0]['bbox'][3]/2), coco.loadAnns(ann)[0]['bbox'][0] + int(coco.loadAnns(ann)[0]['bbox'][2]/2)]
        if img_file_name not in images_map:
            images_map[img_file_name] = []
        images_map[img_file_name].append({'ann' : ann, 'point' : point})
    
    return images_map

def get_mask(coco: COCO, ann_id):
    return coco.annToMask(coco.loadAnns(ann_id)[0])

def compute_iou_for_trash_using_sam(path, predictor: SamPredictor):
    ious = []

    coco = COCO(os.path.join(path, 'instances_val_trashcan.json'))
    val_dir = os.path.join(path, 'val')
    
    image_map = get_image_map(coco)
    
    for img_name in image_map.keys():
        img_path = os.path.join(val_dir, img_name)

        image = np.array(Image.open(img_path).convert('RGB'))
        masks = [get_mask(coco, ann_obj['ann']) for ann_obj in image_map[img_name]]
        points = [ann_obj['point'] for ann_obj in image_map[img_name]]

        predictor.set_image(image)

        for m_index, mask_gt in enumerate(masks):
            point = [points[m_index][1], points[m_index][0]]
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
            with open('iou_trash.txt', 'a+') as f:
                f.write('IoU: ' + str(max_iou) + '\n')
    
    print('mean IoU for trash dataset:', sum(ious)/len(ious), 'IoUs sum:', sum(ious))