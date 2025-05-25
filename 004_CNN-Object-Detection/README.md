# 🧠 CNN Object Detection

This folder contains Jupyter Notebook implementations of popular **CNN-based Object Detection algorithms**, ranging from classical region-based methods to modern real-time detectors like YOLO. It also includes practical usage with **Roboflow** for dataset handling and model deployment.

---

## 📁 Contents

### 🔹 Region-Based CNNs

| Algorithm       | Description                                                                                                            |
| --------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **RCNN**        | Region-based CNN using selective search for proposals, CNN for feature extraction, SVM for classification.             |
| **Fast RCNN**   | Faster and more efficient version of RCNN using ROI pooling and a single model for both classification and regression. |
| **Faster RCNN** | Introduced Region Proposal Network (RPN) for learning region proposals directly from data.                             |

---

### 🔹 YOLO (You Only Look Once)

YOLO is a family of real-time object detection models with different versions:

| Version               | Highlights                                                                          |
| --------------------- | ----------------------------------------------------------------------------------- |
| **YOLOv1**            | First unified model for object detection using a single CNN.                        |
| **YOLOv2 (YOLO9000)** | Improved accuracy, multi-scale training, and better bounding box priors.            |
| **YOLOv3**            | Introduced feature pyramid networks and improved performance on small objects.      |
| **YOLOv4**            | Combined multiple tricks like mish activation, mosaic data augmentation, and more.  |
| **YOLOv5**            | PyTorch-based, modular, and widely used in the industry (community-developed).      |
| **YOLOv6**            | Optimized for industrial applications with improved speed and accuracy.             |
| **YOLOv7**            | State-of-the-art performance with architectural improvements and extended features. |

---

### 🔹 Roboflow Integration

- Hands-on experience using **Roboflow** for:

  - Dataset preprocessing and annotation
  - Custom object detection dataset creation
  - Model training and evaluation pipeline

---

## 📌 Notes

- All implementations are done in **Jupyter Notebooks** for better visualization and experimentation.
- This folder is part of my **Deep Learning journey** to master object detection from research to real-world applications.

---

## 📚 References

- Papers: RCNN, Fast RCNN, Faster RCNN, YOLOv1–v7
- Roboflow: [https://roboflow.com/](https://roboflow.com/)
- YOLOv5 Repo: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
