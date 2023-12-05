from segment_anything import SamPredictor, sam_model_registry

from test_ade20k import compute_iou_for_ade20k_using_sam
from test_leaf import compute_iou_for_leaf_using_sam
from test_ishape import compute_iou_for_ishape_using_sam
from test_gtea import compute_iou_for_gtea_using_sam
from test_trash import compute_iou_for_trash_using_sam
from test_woodscape import compute_iou_for_woodscape_using_sam

import os


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)


ade_path = os.path.join('..', 'ade20k', 'ADE20K_2021_17_01', 'images', 'ADE', 'validation')
plant_path = os.path.join('..', 'Plant_Phenotyping_Datasets', 'Plant')
ishape_path = os.path.join('..', 'ishape')
gtea_path = os.path.join('..', 'GeorgiaTech', 'GTEA')
trash_path = os.path.join('..', 'dataset', 'instance_version')
woodscape_path = os.path.join('..', 'woodscape')


print('Testing SAM on ADE20k...')
compute_iou_for_ade20k_using_sam(ade_path, predictor)
print()

print('Testing SAM on leaf...')
compute_iou_for_leaf_using_sam(plant_path, predictor)
print()

print('Testing SAM on iShape...')
compute_iou_for_ishape_using_sam(ishape_path, predictor)
print()

print('Testing SAM on GTEA...')
compute_iou_for_gtea_using_sam(gtea_path, predictor)

print('Testing SAM on Trash Can...')
compute_iou_for_trash_using_sam(trash_path, predictor)

print('Testing SAM on WoodScape Can...')
compute_iou_for_woodscape_using_sam(woodscape_path, predictor)