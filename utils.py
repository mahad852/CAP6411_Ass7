import numpy as np

from PIL import Image, ImageColor
from matplotlib.colors import rgb2hex


def get_predicted_mask_set(predicted_mask) -> set:
    if len(predicted_mask.shape) == 3:
        predicted_mask = predicted_mask[0]
    
    p_set = set()

    for h in range(len(predicted_mask)):
        for w in range(len(predicted_mask[h])):
            if predicted_mask[h][w] == 1:
                p_set.add([h, w])
    
    return p_set


def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)


def get_image_center_coords(image: Image, masks: Image):
    image_centers = np.asarray(image.convert('RGB'))
    masks_array = np.asarray(masks.convert('RGB'))

    centers = []

    for h in range(len(image_centers)):
        for w in range(len(image_centers[h])):
            if sum(image_centers[h][w]) != 0:
                color = rgb2hex(masks_array[h][w][0], masks_array[h][w][1], masks_array[h][w][2])
                centers.append(((h, w), color))
    
    return centers

def get_mask_sets_from_segmented_image(image: Image) -> dict[str, set]:
    image_array = np.asarray(image.convert('RGB'))
    i_sets = {}
    for h in range(len(image_array)):
        for w in range(len(image_array[h])):
            if np.all(image_array[h][w] == 0):
                continue

            color = rgb2hex(image_array[h][w][0], image_array[h][w][1], image_array[h][w][2])
            if color not in i_sets:
                i_sets[color] = set()

            i_sets[color].add((h, w))
    
    return i_sets


def compute_iou_between_gt_and_sam(image: Image, mask, label):
    image_array = np.asarray(image.convert('RGB'))
    intersection = 0
    union = 0

    for h in range(len(image_array)):
        for w in range(len(image_array[h])):
            color = rgb2hex(image_array[h][w][0], image_array[h][w][1], image_array[h][w][2])
            
            if color == label or mask[h][w]:
                union += 1

            if color == label and mask[h][w]:
                intersection += 1

    return intersection/union

def color_image_with_mask(image, mask, color):
    color_rgb = ImageColor.getcolor(color, 'RGB')
    for h in range(len(image)):
        for w in range(len(image[h])):
            if mask[h][w]:
                image[h][w] = color_rgb
    
    return image