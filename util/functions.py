import cv2
import numpy as np
from nms_wrapper import nms


def resize(image, min_side=800, max_side=1400):
    rows, cols, cns = image.shape
    smallest_side = min(rows, cols)
    scale = min_side / smallest_side
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side
    image = cv2.resize(image, (int(round((cols * scale))), int(round((rows * scale)))))

    rows, cols, cns = image.shape

    pad_w = (-rows) % 32
    pad_h = (-cols) % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)

    return new_image, scale


def vis_detections(im, class_det, w=None):
    for det in class_det:
        bbox = det[:4]
        score = det[4]
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (127, 255, 0), 1)
        cv2.putText(im, '{:.3f}'.format(score), (int(bbox[0]), int(bbox[1] - 9)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), thickness=1, lineType=8)

    if w is not None:
        cv2.imwrite(w, im)


def parse_det_offset(pos, height, offset, size, score=0.1, down=4, nms_thresh=0.3):
    pos = np.squeeze(pos)
    height = np.squeeze(height)
    offset_y = offset[0, 0, :, :]
    offset_x = offset[0, 1, :, :]
    y_c, x_c = np.where(pos > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = 0.41 * h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = pos[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, size[1]), min(y1 + h, size[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, nms_thresh, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs