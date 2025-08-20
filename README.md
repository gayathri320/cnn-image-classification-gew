# CNN Image Classification: Glasses, Earrings, Watch

## 📌 Overview
This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify images into three categories: **Glasses**, **Earrings**, and **Watch**.  

Since the dataset is small (~35 images per class), we applied **data preprocessing, augmentation, and normalization** to improve generalization and model performance.

---

## 📂 Repository Contents
- `dataset_raw/` → Original dataset (Glasses, Earrings, Watch images)  
- `dataset_augmented_train/` → Augmented dataset generated during preprocessing  
- `codes/` → 
  - `cnn_notebook.ipynb` (detailed Colab notebook with preprocessing, training, and evaluation)  
  - `train_cnn.py` (Python script version of the notebook)  
- `models/` → Saved trained CNN model (`cnn_model.keras`)  
- `results/` → Confusion matrix, training curves, classification report  
- `report/` → Detailed project report (PDF/DOCX)  
- `README.md` → Project documentation  

---

## 📊 Dataset & Preprocessing
- Dataset contains ~35 images per class (**Glasses**, **Earrings**, **Watch**)  
- Preprocessing steps applied:
  1. File cleaning (remove non-images, normalize filenames)  
  2. Train/Validation split (**80% / 20%**)  
  3. **Data Augmentation**: rotation, shift, zoom, shear, horizontal flip  
  4. **Normalization**: pixel values scaled from `[0,255]` → `[0,1]`  

---

## 🧠 CNN Model Architecture
- **Conv2D (32 filters, ReLU)** → MaxPooling  
- **Conv2D (64 filters, ELU)** → MaxPooling  
- **Conv2D (128 filters, ReLU)** → MaxPooling  
- Flatten → Dense(128, ELU) → Dropout(0.5)  
- Dense(3, Softmax)  

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  
**Evaluation Metric:** Accuracy  

---

## 📈 Results
- **Training Accuracy:** ~100%  
- **Validation Accuracy:** ~86%  
- Evaluation includes:
  - Confusion Matrix (`results/confusion_matrix.png`)  
  - Training Accuracy & Loss Curves (`results/training_accuracy.png`, `results/training_loss.png`)  
  - Classification Report (`results/classification_report.txt`)  

---

## 🚀 How to Run
### Option 1: Run in Google Colab
1. Open `codes/cnn_notebook.ipynb` in Colab  
2. Upload dataset ZIP when prompted  
3. Run all cells  

### Option 2: Run Locally
```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/cnn-image-classification-gew.git
cd cnn-image-classification-gew

# Install dependencies
pip install -r requirements.txt

# Run training script
python codes/train_cnn.py

