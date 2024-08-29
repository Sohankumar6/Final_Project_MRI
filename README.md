# Final_Project_MRI
Advanced Deep Learning for Automated Brain Tumor Detection and Classification via MRI Images

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
 brain_tumor_classification.ipynb  : Main Jupyter Notebook containing the project code
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

**Mount Google Drive**
from google.colab import drive
drive.mount('/content/drive')   this code is to mount my google drive to access dataset.

**Difining Paths**
directory_path = '/content/drive/MyDrive/Final project'
testing_path = directory_path + '/Testing'
training_path = directory_path + '/Training'  setting the base directory path.

**Loading and Preprocessing Data**
training_count, training_image_paths, training_labels = load_images(training_path)
testing_count, testing_image_paths, testing_labels = load_images(testing_path)  Using the load_images function to load the images, and preprocess them for model training

**Model Training**
cnn_history = cnn_model.fit(
    train_data_preprocessed,
    epochs=epoch,
    validation_data=test_data_preprocessed,
    callbacks=[model_lr_reduction, model_checkpoint, lr_scheduler],
    verbose=True
) this is a training a CNN model code, similarly CNN-VGG-16, KNN, Random Forest models are trained.

**Evaluating Models**
training_loss, training_accuracy = cnn_model.evaluate(train_data_preprocessed)
print(f"Training accuracy: {training_accuracy*100:.4f}%") After training, evaluating the models on the test set

**Predictions**
display_sample_predictions(cnn_model, prediction_dir, inv_class_namings, figsize=(13, 12)) Predicting using trained models

**Visualize Results**
display_confusion_matrix(true_label_list,
                      predicted_label_list,
                      class_namings,
                      metrics=True)  Visualize the performance of the models using confusion matrices and learning curves

**Results**
The models achieve varying degrees of accuracy, with deep learning models like CNN and VGG-16 generally outperforming traditional machine learning algorithms like KNN and Random Forest in terms of accuracy and precision.

**Future Work**
	Gathering more extensive and varied datasets to enhance model performance and generalization on unseen data.
	Investigating cutting-edge methods of data augmentation, like the generation of synthetic data using Generative Adversarial Networks (GANs).
	Utilizing transfer learning with architectures such as ResNet, EfficientNet, or Vision Transformers that goes beyond VGG-16.
 	Enhancing classification accuracy by combining CNNs with additional machine learning methods for decision-level fusion.
	Enhancing generalization and lowering variance by combining several models through ensemble learning techniques.

