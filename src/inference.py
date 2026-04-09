from detectron2.engine import DefaultPredictor
import matplotlib.pyplot as plt
from trainer import cfg
import os
from src.config import Config
import cv2 
import Visualizer

config = Config()

image_path = "/content/test.jpg"
image = cv2.imread(image_path)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)


outputs = predictor(image)

v = Visualizer(image[:, :, ::-1], metadata=config.metadata, scale=0.8)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

plt.imshow(out.get_image())
plt.axis("off")