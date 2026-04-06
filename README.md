# Structural Crack Detection & Segmentation using Mask-RCNN and Detecron2

**A Computer Vision Approach for Infrastructure Monitoring**

---

## 📌 Project Summary

This project presents a deep learning-based approach to automated pavement crack detection and segmentation using state-of-the-art object detection and instance segmentation models.  
The work is motivated by the need to improve infrastructure inspection, where traditional manual methods are slow, subjective, and difficult to scale. By leveraging computer vision, this project demonstrates how data-driven techniques can support efficient and consistent pavement condition assessment.

---

## 🎯 Research Motivation

cracks on buildins are early indicators of structural deterioration. Accurate detection and characterization are essential for:

- Preventive maintenance planning
- Pavement Condition Index (PCI) estimation
- Infrastructure lifecycle management  

However, cracks are:

- Thin and irregular
- Highly variable in appearance
- Difficult to detect using conventional methods

This project explores how deep learning can address these challenges.

---

## 🧠 Methodology

The system is built using the **Detectron2** framework and implements:

### 🔹 Models

- **Faster R-CNN** → Crack detection (bounding boxes)
- **Mask R-CNN** → Crack segmentation (pixel-level masks)

### 🔹 Workflow

- Dataset preparation (COCO format from Roboflow)
- Data cleaning and class normalization
- Model configuration and transfer learning from COCO-pretrained weights
- Training with periodic evaluation
- Performance visualization and analysis

---

## ⚙️ Technical Stack

- Python
- PyTorch
- Detectron2
- OpenCV
- Pandas & Matplotlib
- Google Colab (for training and experimentation)

---

## 📊 Dataset

- Annotated crack images (COCO format)
- Single-class problem: **crack** detection
- Supports both:
  - Bounding boxes (detection)
  - Polygon masks (segmentation)
- ⚠️ Special attention was given to dataset consistency, including class ID normalization and annotation validation.

[Dataset Source](https://universe.roboflow.com/roboflow-jvuqo/creacks-eapny/dataset/8)

---

## 📈 Evaluation Metrics

The model is evaluated using COCO metrics:

**Bounding Box Metrics**

- AP (Average Precision)
- AP50, AP75

**Segmentation Metrics**

- Mask AP

### Example result:

| Metric       | Score |
| ------------ | ----- |
| BBox AP      | 57.36  |
| BBox AP50    | 80.62  |
| Segmentation | ----- |


---


## 🔍 Inference Example

The trained model can:

- Detect crack regions in images
- Generate bounding boxes and segmentation masks
- Support further analysis (e.g., crack length and severity)

---
