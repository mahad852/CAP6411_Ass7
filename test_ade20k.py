from segment_anything import SamPredictor, sam_model_registry
from utils import get_mask_sets_from_segmented_image, compute_iou_between_gt_and_sam
from utils import color_image_with_mask
import os
import json
import numpy as np
from PIL import Image

def compute_iou_for_ade20k_using_sam(path, predictor: SamPredictor):
    sub_dirs = [os.path.join(path, sub_dir) for sub_dir in os.listdir(path)]

    ious = []
    masks_processed = 0
    
    for sub_dir in sub_dirs:
        if not os.path.isdir(sub_dir):
            continue

        for object in os.listdir(sub_dir):
            if not os.path.isdir(os.path.join(sub_dir, object)):
                continue

            for file in os.listdir(os.path.join(sub_dir, object)):
                if 'json' not in file:
                    continue
                
                with open(os.path.join(sub_dir, object, file)) as f:
                    annotations_json = json.load(f)

                image_name = annotations_json['annotation']['filename']
                image_path = os.path.join(sub_dir, object, image_name)
                image = np.array(Image.open(image_path).convert('RGB'))

                predictor.set_image(image)
                
                masks_image_name = image_name.split('.')[0] + '_seg.png'
                mask_path = os.path.join(sub_dir, object, masks_image_name)
                mask_gt = Image.open(mask_path)
                mask_sets = get_mask_sets_from_segmented_image(mask_gt)

                for label in mask_sets.keys():
                    x, y = list(mask_sets[label])[int(len(mask_sets[label])/2)]
                    points = np.array([
                        [y, x]
                    ])
                    masks, _, _ = predictor.predict(
                        point_coords=points,
                        point_labels=np.array([int(label[1:], 16)]),
                        multimask_output=True,
                    )
                    max_iou = max(list(map(lambda mask: compute_iou_between_gt_and_sam(mask_gt, mask, label), masks)))
                    ious.append(max_iou)

                    masks_processed += 1
                    
                    if masks_processed % 20 == 0:
                        print('ADE20K, masks processed:', masks_processed)

                    with open('iou_ade20k.txt', 'a+') as f:
                         f.write('IoU: ' + str(max_iou) + '\n')
    
    print('total mean IoU:', sum(ious)/len(ious), 'Total IoUs sum:', sum(ious))


def generate_masks_for_ade20k_using_sam(path, num_images = 10):
    sub_dirs = [os.path.join(path, sub_dir) for sub_dir in os.listdir(path)]

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    total_images = 0

    for sub_dir in sub_dirs:
        if not os.path.isdir(sub_dir):
            continue

        for object in os.listdir(sub_dir):
            if not os.path.isdir(os.path.join(sub_dir, object)):
                continue

            for file in os.listdir(os.path.join(sub_dir, object)):
                total_images += 1
                
                if total_images > num_images:
                    return

                if 'json' not in file:
                    continue
                
                with open(os.path.join(sub_dir, object, file)) as f:
                    annotations_json = json.load(f)

                image_name = annotations_json['annotation']['filename']
                image_path = os.path.join(sub_dir, object, image_name)
                image = np.array(Image.open(image_path).convert('RGB'))

                predictor.set_image(image)
                
                masks_image_name = image_name.split('.')[0] + '_seg.png'
                mask_path = os.path.join(sub_dir, object, masks_image_name)
                mask_gt = Image.open(mask_path)
                mask_sets = get_mask_sets_from_segmented_image(mask_gt)

                color_image = np.array(Image.new('RGB', (image.shape[1], image.shape[0])).convert('RGB'))

                for l_index, label in enumerate(mask_sets.keys()):
                    x, y = list(mask_sets[label])[int(len(mask_sets[label])/2)]
                    points = np.array([
                        [y, x]
                    ])
                    masks, _, _ = predictor.predict(
                        point_coords=points,
                        point_labels=np.array([int(label[1:], 16)]),
                        multimask_output=True,
                    )
                    
                    max_iou = 0
                    max_index = 0

                    for i, mask in enumerate(masks):
                        iou = compute_iou_between_gt_and_sam(mask_gt, mask, label)

                        if iou > max_iou:
                            max_index = i
                            max_iou = iou

                    mask_color_image = np.array(Image.new('RGB', (image.shape[1], image.shape[0])).convert('RGB'))
                    
                    color_image_with_mask(mask_color_image, masks[max_index], label)
                    Image.fromarray(mask_color_image.astype('uint8'), 'RGB').save('sample_' + object + '_' + str(l_index) + '.png')

                    print('label:', label, max_iou, np.array(mask_gt.convert('RGB')).shape, image.shape)
                    
                    color_image_with_mask(color_image, masks[max_index], label)
                
                
                Image.fromarray(color_image.astype('uint8'), 'RGB').save('sample_' + object + '.png')
                Image.fromarray(image.astype('uint8'), 'RGB').save('original_' + object + '.png')
