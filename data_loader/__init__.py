from .cityscapes import CitySegmentation
from .coco import COCOSegmentation

datasets = {
    'citys': CitySegmentation,
    'coco': COCOSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
