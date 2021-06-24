"""MSCOCO Semantic Segmentation pretraining for VOC."""
import os
import glob
import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np

import torch
import torch.utils.data as data

# from .utils import try_import_pycocotools
# from ..segbase import SegmentationDataset


class COCOSegmentation(data.Dataset):
    """COCO Semantic Segmentation Dataset for VOC Pre-training.

    Parameters
    ----------
    root : string
        Path to COCO folder. Default is '$(HOME)/mxnet/datasets/coco'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image

    Examples
    --------
    >>> from mxnet.gluon.data.vision import transforms
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = gluoncv.data.COCOSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = gluon.data.DataLoader(
    >>>     trainset, 4, shuffle=True, last_batch='rollover',
    >>>     num_workers=4)
    """
    # CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
    #             1, 64, 20, 63, 7, 72]
    CAT_LIST = [0, 1]
    # NUM_CLASS = 21
    NUM_CLASS = 2
    # CLASSES = ("background", "airplane", "bicycle", "bird", "boat", "bottle",
    #            "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    #            "motorcycle", "person", "potted-plant", "sheep", "sofa", "train",
    #            "tv")
    CLASSES = ("background", "fire hydrant")

    def __init__(self, root,
                 split='train', mode=None, transform=None, base_size=520, crop_size=480, **kwargs):
        super(COCOSegmentation, self).__init__()
        self.mode = mode if mode is not None else split
        self.split = split
        self.base_size = base_size
        self.crop_size = crop_size
        # lazy import pycocotools
        # try_import_pycocotools()
        # from pycocotools.coco import COCO
        # from pycocotools import mask
        self.root = os.path.abspath(root)
        if split == 'train':
            print('train set')
            self.anns = glob.glob(os.path.join(root, 'annotations/train2017/*.png'))
            # ids_file = os.path.join(root, 'annotations/train_ids.mx')
            # self.root = os.path.join(root, 'train2017')
        else:
            print('val set')
            self.anns = glob.glob(os.path.join(root, 'annotations/val2017/*.png'))
            # ids_file = os.path.join(root, 'annotations/val_ids.mx')
            # self.root = os.path.join(root, 'val2017')
        # self.coco = COCO(ann_file)
        # self.coco_mask = mask
        # if os.path.exists(ids_file):
        #     with open(ids_file, 'rb') as f:
        #         self.ids = pickle.load(f)
        # else:
        #     ids = list(self.coco.imgs.keys())
            # self.ids = self._preprocess(ids, ids_file)
        self.transform = transform

    def __getitem__(self, index):
        annot_path = self.anns[index]
        image_path = annot_path.replace('annotations', 'images').replace('.png', '.jpg')

        img = Image.open(image_path).convert('RGB')
        mask = Image.open(annot_path)
        # coco = self.coco
        # img_id = self.ids[index]
        # img_metadata = coco.loadImgs(img_id)[0]
        # path = img_metadata['file_name']
        # img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        # mask = Image.fromarray(self._gen_seg_mask(
        #     cocotarget, img_metadata['height'], img_metadata['width']))
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def __len__(self):
        return len(self.ids)

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def __len__(self):
        return len(self.anns)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0