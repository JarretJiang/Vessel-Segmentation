# Vessel Segmentation with Convolutional Neural Networks

## Author
- **Name: Yuxuan(Jarret) Jiang
- **Email: jiangyx96@gmail.com  
- **GitHub: https://github.com/JarretJiang/Vessel-Segmentation.git
- **Date: September 2025  

---

## Project Description
This repository contains the implementation of a convolutional neural network (CNN) for vessel (carotid artery) cross-sectional segmentation.  
The project includes training scripts, pretrained model parameters, and example vessel videos.

---

## Python Files
- **Training.py**
  Main script for training the neural network. It loads the dataset, initializes the model, and saves the best-performing parameters.  

- **Prediction.py**
  Uses the trained model parameters and the neural network to perform vessel area segmentation on a specified video.

- **echo.py**
  Defines a class that reads the annotation (label) file and generates the corresponding segmentation masks.  

- **init1.py**
  A collection of helper functions for auxiliary tasks, such as reading and saving video files.

---

## Pretrained Model and Example Data
The trained model parameters and three example vessel videos are available on Google Drive:
[Google Drive Link]
https://drive.google.com/drive/folders/1pdBDtPfCtE1m5XU82qvUlBO-rGurly_S?usp=sharing

	Contents:
- 'best.pt'
	Trained model parameters (weights) for vessel segmentation.  
- 'c0001.avi', 'c0002.avi', 'c0003.avi'
	Example vessel videos for testing and demonstration.  