import json
from src.config import Config

config = Config()

datasets = [
    ("train/_annotations.coco.json", "/content/train_fixed.coco.json"),
    ("valid/_annotations.coco.json", "/content/valid_fixed.coco.json"),
]

def clean_data_annotation ():

    for input_path, output_path in datasets:
        full_input_path = f"{config.DATASET_PATH}/{input_path}"

        with open(full_input_path) as f:
            data = json.load(f)

        # Keep only ONE category
        data["categories"] = [{"id": 1, "name": "crack"}]

        # Fix all annotations
        for ann in data["annotations"]:
            ann["category_id"] = 1

        # Save
        with open(output_path, "w") as f:
            json.dump(data, f)

        print(f"✅ Fixed: {input_path} → {output_path}")