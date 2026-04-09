import random
import cv2
from google.colab.patches import cv2_imshow
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from src.config import Config

config = Config()


def sanity_check():

    for d in random.sample(config.dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=config.metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)

        cv2_imshow(out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
