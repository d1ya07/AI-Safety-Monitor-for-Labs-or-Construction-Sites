# 🦺 AI Safety Monitor for Labs or Construction Sites

## 🔍 Project Overview  
This project is an AI-based helmet detection system designed to improve safety in laboratories and construction sites. Using a Convolutional Neural Network (CNN), it classifies images to determine whether a person is wearing a helmet or not. The system helps automate safety monitoring using image processing and machine learning.

---

## ❓ Problem Statement  
Manual monitoring of safety gear usage is often inefficient and error-prone. In high-risk environments like construction sites and labs, even small lapses in helmet usage can lead to serious injuries. There's a need for an automated solution that can identify and report non-compliance instantly.

---

## 🎯 Objective  
To develop a deep learning-based image classification system that can:
- Detect if individuals in an image are **wearing helmets**
- Identify instances where helmets are **not worn**
- Assist in ensuring compliance with safety protocols

---

## 🧠 Libraries & Technologies Used  
- **TensorFlow** – For building and training the CNN model  
- **OpenCV-Python** – For reading and preprocessing images  
- **NumPy** – For numerical operations  
- **Matplotlib** – For visualizing accuracy and loss graphs  
- **Scikit-learn** – For splitting the dataset and performance evaluation

---

## 📁 Dataset  
- The dataset consists of labeled images divided into two folders:
  - `with_helmet/`
  - `without_helmet/`
- Images were resized to uniform dimensions and normalized.
- Data was split into training and testing sets using `train_test_split` from scikit-learn.

---

## 📈 Results  
- **Accuracy Achieved:** ~82% on test data  
- The model performs well in distinguishing between helmeted and non-helmeted individuals  
- Visualizations include training/validation accuracy and loss graphs
