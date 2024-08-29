# Final_Project_MRI
**Advanced Deep Learning for Automated Brain Tumor Detection and Classification via MRI Images**

This project aims to classify brain tumors from MRI images using various machine learning and deep learning models. The models implemented in this project include:
  Convolutional Neural Networks (CNN)
  CNN-based VGG-16
  K-Nearest Neighbors (KNN)
  Random Forest
  
The goal is to as follows
  Create and apply machine learning models, such as Convolutional Neural Networks (CNN), VGG-16, k-Nearest Neighbors (KNN), and Random Forest, designed specifically for identifying brain tumors.
  Optimize the performance of each model by fine-tuning them to accurately detect and classify brain tumors, ensuring their suitability for medical imaging tasks.
	Assess the accuracy, speed, and computational efficacy of each model by employing an extensive dataset of MRI scans, offering a thorough evaluation of their capabilities.
	Perform a comprehensive comparative analysis of the four models, evaluating their performance metrics to determine their individual strengths and weaknesses.
 
**Features**
  Data Preprocessing: Efficient loading, augmentation, and preprocessing of MRI images.
  Model Training: Training scripts for CNN, VGG-16, KNN, and Random Forest models.
  Model Evaluation: Detailed evaluation metrics including accuracy, confusion matrix, precision, recall, and F1-score.
  Prediction: Predicts tumor types for new MRI images and visualizes results.
  
**Dependencies**
To run this project, you need the following libraries installed google-colab, tensorflow, keras, numpy, pandas, matplotlib, seaborn, PIL, scikit-learn, joblib 

**Project structure**
 brain_tumor_classification.ipynb  :Google Colab file of my project code
 models/
    cnn_model.keras               : Saved CNN model
    cnn_vgg16_model.keras         : Saved VGG-16 model
    knn_model.pkl                 : Saved KNN model
    random_forest_model.pkl       : Saved Random Forest model
 data
    Training                     : Directory containing training MRI images
    Testing                     : Directory containing testing MRI images
 ImagePrediction                  : Directory containing new images for prediction
 README.md                         : This README file

**Results**
The models achieve varying degrees of accuracy, with deep learning models like CNN and VGG-16 generally outperforming traditional machine learning algorithms like KNN and Random Forest in terms of accuracy and precision.
![Screenshot 2024-08-29 150249](https://github.com/user-attachments/assets/d739c436-5ee6-4907-bde7-7b70ae659b03)
![Screenshot 2024-08-29 150321](https://github.com/user-attachments/assets/ae2a8b0a-4e26-47b3-8af8-2877f0a409ff)


**License**
The dataset is available on Kaggle and it is a open source.

**Acknowledgement**
Thanks to Alyssa Drake for her guidance throughout the project.


