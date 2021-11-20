from .cityscapes import CitySegmentation
from .coco import COCOSegmentation
from .person import PersonSegmentation

datasets = {
    'citys': CitySegmentation,
    'coco': COCOSegmentation,
    'person': PersonSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
