import cv2
from PIL import Image
import numpy as np

from src.core.ClassificationModels import MotorBikeModels


def crop_bb(original_image: Image, bb_boxes: list):
    # Original size of the image
    original_width, original_height = original_image.size

    crop_images = []

    for bb in bb_boxes:
        x, y, w, h = bb  # Here, they are coordinates in pixel

        # Convert to the coordinates in the original image
        x_min = int(round(x - (w / 2)))
        y_min = int(round(y - (h / 2)))
        x_max = x_min + int(round(w))
        y_max = y_min + int(round(h))

        # Crop the image
        crop_image = original_image.crop((x_min, y_min, x_max, y_max))
        crop_images.append(crop_image)

    return crop_images


def draw_bb(thickness: int, font_scale: float, original_image: Image, bb_boxes: list, classes: list):
    # Convert the image to numpy array
    original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    for bb, cls in zip(bb_boxes, classes):
        x, y, w, h = bb
        x_min = int(round(x - (w / 2)))
        y_min = int(round(y - (h / 2)))
        x_max = x_min + int(round(w))
        y_max = y_min + int(round(h))

        # Class 0 will be in blue, class 1 will be in green, else transparent
        if cls == 0:
            color = (255, 0, 0)
            label = 'xe so'
        elif cls == 1:
            color = (0, 255, 0)
            label = 'xe ga'
        else:
            color = (0, 0, 0)
            label = ''

        # Draw the bounding box
        cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), color, thickness)

        # Draw the label with the bounding box
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_min = max(y_min, label_size[1])
        cv2.rectangle(original_image, (x_min, y_min - label_size[1]),
                      (x_min + label_size[0], y_min + base_line),
                      color, cv2.FILLED)
        cv2.putText(original_image, label, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

    return Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    original_image = Image.open(r'C:\Users\QUANPC\Documents\GitHub\Motorcycle-Detection-Streamlit\demo\original.jpg')
    bounding_boxes = [[741.3970947265625, 402.2313232421875, 48.309814453125, 67.0274658203125],
                      [530.93017578125, 441.2001647949219, 45.9754638671875, 59.4688720703125]
                      ]
    crop_images = crop_bb(original_image, bounding_boxes)

    # Peform prediction on the cropped images
    model = MotorBikeModels(
        model='resnet18',
        weight='src/configs/weights/classify/resnet18.ckpt',
    )

    classes = []
    for img in crop_images:
        classes.append(model.infer(img))

    # Draw the bounding boxes
    original_image = draw_bb(2, 0.5, original_image, bounding_boxes, classes)
    original_image.save('demo/processed.jpg')
