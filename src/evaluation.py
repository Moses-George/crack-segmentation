from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
import pandas as pd
from trainer import cfg, trainer
import json
import matplotlib.pyplot as plt

metrics_file = "/content/output/metrics.json"

evaluator = COCOEvaluator("crack_valid", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "crack_valid")

trainer.test(cfg, trainer.model, evaluators=[evaluator])



data = []
with open(metrics_file, "r") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Plot training loss

plt.figure()
plt.plot(df["iteration"], df["total_loss"], label="Total Loss")

if "loss_cls" in df:
    plt.plot(df["iteration"], df["loss_cls"], label="Cls Loss")

if "loss_box_reg" in df:
    plt.plot(df["iteration"], df["loss_box_reg"], label="Box Loss")

if "loss_mask" in df:
    plt.plot(df["iteration"], df["loss_mask"], label="Mask Loss")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss Curve")
plt.show()

# Plot evaluation metrics
if "bbox/AP" in df.columns:
    eval_df = df[df["bbox/AP"].notna()]

    plt.figure()
    plt.plot(eval_df["iteration"], eval_df["bbox/AP"], label="AP")
    plt.plot(eval_df["iteration"], eval_df["bbox/AP50"], label="AP50")

    plt.legend()
    plt.title("Detection Performance")
    plt.show()
else:
    print("⚠️ No evaluation metrics found. Did you set cfg.TEST.EVAL_PERIOD?")


# Segmentation metrics (if available)
if "segm/AP" in df:
    segm_df = df[df["segm/AP"].notna()]

    plt.figure()
    plt.plot(segm_df["iteration"], segm_df["segm/AP"], label="Mask AP")

    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Segmentation Performance")
    plt.show()