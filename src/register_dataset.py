from detectron2.data.datasets import register_coco_instances
from src.config import Config

config = Config()

def register_dataset():

    register_coco_instances(
        "crack_train",
        {},
        "/content/train_fixed.coco.json",
        f"{config.DATASET_PATH}/train/"
    )

    register_coco_instances(
        "crack_valid",
        {},
        "/content/valid_fixed.coco.json",
        f"{config.DATASET_PATH}/valid/"
)

