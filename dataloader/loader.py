import sys
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from load_data import get_citypersons


class CityPersons(Dataset):
    def __init__(self, path, train, config, preloaded=False, transform=None):
        self.dataset = get_citypersons(root_dir=path, type=train)
        self.dataset_len = len(self.dataset)
        if config.train_random:
            random.shuffle(self.dataset)
        self.config = config
        self.transform = transform
        self.preprocess = RandomResizeFix(size=config.size_train, scale=(0.4, 1.5))
        self.preloaded = preloaded
        if self.preloaded:
            self.img_cache = []
            for i, data in enumerate(self.dataset):
                self.img_cache.append(Image.open(data['filepath']))
                print('%d/%d\r' % (i+1, self.dataset_len)),
                sys.stdout.flush()

    def __getitem__(self, item):
        img_data = self.dataset[item]
        if self.preloaded:
            img = self.img_cache[item]
        else:
            img = Image.open(img_data['filepath'])
        gts = img_data['bboxes'].copy()
        igs = img_data['ignoreareas'].copy()
        
        x_img, gts, igs = self.preprocess(img, gts, igs)
        
        y_center, y_height, y_offset = self.calc_gt_center(gts, igs, radius=2, stride=self.config.down)

        x_img = self.transform(x_img)

        return x_img, [y_center, y_height, y_offset]

    def __len__(self):
        return self.dataset_len
        
    def calc_gt_center(self, gts, igs, radius=2, stride=4):
        def gaussian(kernel):
            sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
            s = 2*(sigma**2)
            dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
            return np.reshape(dx, (-1, 1))
        scale_map = np.zeros((2, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        offset_map = np.zeros((3, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        pos_map = np.zeros((3, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        pos_map[1, :, :,] = 1 # channel 1: 1-value mask, ignore area will be set to 0
        if len(igs) > 0:
            igs = igs / stride
            for ind in range(len(igs)):
                x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
                pos_map[1, y1:y2, x1:x2] = 0
        if len(gts) > 0:
            gts = gts / stride
            for ind in range(len(gts)):
                x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
                c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
                dx = gaussian(x2-x1)
                dy = gaussian(y2-y1)
                gau_map = np.multiply(dy, np.transpose(dx))
                pos_map[0, y1:y2, x1:x2] = np.maximum(pos_map[0, y1:y2, x1:x2], gau_map)  # gauss map
                pos_map[1, y1:y2, x1:x2] = 1  # 1-mask map
                pos_map[2, c_y, c_x] = 1  # center map

                scale_map[0, c_y-radius:c_y+radius+1, c_x-radius:c_x+radius+1] = np.log(gts[ind, 3] - gts[ind, 1])  # log value of height
                scale_map[1, c_y-radius:c_y+radius+1, c_x-radius:c_x+radius+1] = 1  # 1-mask

                offset_map[0, c_y, c_x] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5  # height-Y offset
                offset_map[1, c_y, c_x] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5  # width-X offset
                offset_map[2, c_y, c_x] = 1  # 1-mask

        return pos_map, scale_map, offset_map


class RandomResizeFix(object):
    """
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.4, 1.5), interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.scale = scale

    def __call__(self, img, gts, igs):
        # resize image
        w, h = img.size
        ratio = np.random.uniform(self.scale[0], self.scale[1])
        n_w, n_h = int(ratio * w), int(ratio * h)
        img = img.resize((n_w, n_h), self.interpolation)
        gts = gts.copy()
        igs = igs.copy()
        
        # resize label
        if len(gts) > 0:
            gts = np.asarray(gts, dtype=float)
            gts *= ratio

        if len(igs) > 0:
            igs = np.asarray(igs, dtype=float)
            igs *= ratio
        
        # random flip
        w, h = img.size
        if np.random.randint(0, 2) == 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(gts) > 0:
                gts[:, [0, 2]] = w - gts[:, [2, 0]]
            if len(igs) > 0:
                igs[:, [0, 2]] = w - igs[:, [2, 0]]

        if h >= self.size[0]:
            # random crop
            img, gts, igs = self.random_crop(img, gts, igs, self.size, limit=16)
        else:
            # random pad
            img, gts, igs = self.random_pave(img, gts, igs, self.size, limit=16)

        return img, gts, igs

    @staticmethod
    def random_crop(img, gts, igs, size, limit=8):
        w, h = img.size
        crop_h, crop_w = size

        if len(gts) > 0:
            sel_id = np.random.randint(0, len(gts))
            sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
            sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
        else:
            sel_center_x = int(np.random.randint(0, w - crop_w + 1) + crop_w * 0.5)
            sel_center_y = int(np.random.randint(0, h - crop_h + 1) + crop_h * 0.5)

        crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
        crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
        diff_x = max(crop_x1 + crop_w - w, int(0))
        crop_x1 -= diff_x
        diff_y = max(crop_y1 + crop_h - h, int(0))
        crop_y1 -= diff_y
        cropped_img = img.crop((crop_x1, crop_y1, crop_x1 + crop_w, crop_y1 + crop_h))
        # crop detections
        if len(igs) > 0:
            igs[:, 0:4:2] -= crop_x1
            igs[:, 1:4:2] -= crop_y1
            igs[:, 0:4:2] = np.clip(igs[:, 0:4:2], 0, crop_w)
            igs[:, 1:4:2] = np.clip(igs[:, 1:4:2], 0, crop_h)
            keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & ((igs[:, 3] - igs[:, 1]) >= 8)
            igs = igs[keep_inds]
        if len(gts) > 0:
            before_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
            gts[:, 0:4:2] -= crop_x1
            gts[:, 1:4:2] -= crop_y1
            gts[:, 0:4:2] = np.clip(gts[:, 0:4:2], 0, crop_w)
            gts[:, 1:4:2] = np.clip(gts[:, 1:4:2], 0, crop_h)

            after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

            keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit) & (after_area >= 0.5 * before_area)
            gts = gts[keep_inds]

        return cropped_img, gts, igs

    @staticmethod
    def random_pave(img, gts, igs, size, limit=8):
        img = np.asarray(img)
        h, w = img.shape[0:2]
        pave_h, pave_w = size
        # paved_image = np.zeros((pave_h, pave_w, 3), dtype=image.dtype)
        paved_image = np.ones((pave_h, pave_w, 3), dtype=img.dtype) * np.mean(img, dtype=int)
        pave_x = int(np.random.randint(0, pave_w - w + 1))
        pave_y = int(np.random.randint(0, pave_h - h + 1))
        paved_image[pave_y:pave_y + h, pave_x:pave_x + w] = img
        # pave detections
        if len(igs) > 0:
            igs[:, 0:4:2] += pave_x
            igs[:, 1:4:2] += pave_y
            keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & ((igs[:, 3] - igs[:, 1]) >= 8)
            igs = igs[keep_inds]

        if len(gts) > 0:
            gts[:, 0:4:2] += pave_x
            gts[:, 1:4:2] += pave_y
            keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit)
            gts = gts[keep_inds]

        return Image.fromarray(paved_image), gts, igs

    
