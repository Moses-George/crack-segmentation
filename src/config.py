from detectron2.data import DatasetCatalog, MetadataCatalog

class Config ():
    DATASET_PATH= "/content/datasets"
    dataset_dicts = DatasetCatalog.get("crack_train")
    metadata = MetadataCatalog.get("crack_train")
