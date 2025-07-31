
# 🏥 Hospital Image Diagnosis Web App

A beginner-friendly deep learning project for detecting **Pneumonia** from Chest X-rays and **Brain Tumors** from MRI scans using Convolutional Neural Networks (CNNs), built and deployed with **Streamlit**.

## 🔍 Overview

This project aims to simplify medical image analysis using basic deep learning models. Users can upload Chest X-rays or Brain MRI images to get predictions on whether the input indicates Pneumonia or a Brain Tumor. This project is built with a clear user interface using **Streamlit**, and includes animations for an engaging UX.

> ⚠️ **Disclaimer:** This app is built for learning purposes and **should not be used for real-world medical diagnosis**.

---

## 🛠️ Tech Stack

| Component         | Tool / Library             |
|------------------|----------------------------|
| Language          | Python                     |
| Deep Learning     | TensorFlow / Keras         |
| UI Framework      | Streamlit                  |
| Animation         | Lottie + Streamlit-Lottie  |
| Deployment        | Streamlit Cloud            |
| Data Handling     | NumPy, PIL                 |
| Dataset Handling  | Git clone in Google Colab  |

---

## 📂 Dataset

- **Chest X-Ray Dataset:** Contains X-ray images labeled as Normal or Pneumonia.
  **link:**
- **Brain MRI Dataset:** Contains MRI scans labeled as Tumor or Normal.

Due to size constraints, datasets were accessed using:

```bash
!git clone https://github.com/<your-dataset-repo>.git
```

in **Google Colab** for efficient training and experimentation.

---

## 🧠 Model Architecture

Two simple **CNN** models were built and trained independently:

### 🫁 Chest X-Ray Model
- 3 Conv2D layers with MaxPooling
- Flatten + Dense layers
- Final layer: Sigmoid (binary classification)

### 🧠 Brain Tumor Model
- 4 Conv2D layers with MaxPooling
- 2 Dense hidden layers
- Final layer: Sigmoid

Both models use binary cross-entropy loss and Adam optimizer.

---

## 🧪 Workflow

1. **Data Preprocessing:** Resize images to 64x64, normalize, handle RGB/grayscale.
2. **Model Building:** Constructed in Keras with consistent architecture.
3. **Training:** Conducted in Colab using GPU.
4. **Saving:** Saved as `.h5` using `model.save()` and used `load_weights()` for prediction.
5. **Deployment:** Streamlit app with tabs for Home, Chest X-Ray, and Brain Tumor prediction.
6. **UI Enhancements:** Integrated Lottie animations for a user-friendly UI.

---

## 🚀 Deployment

✅ **Live App**  
👉 [hospital-diagnosis-rksta.streamlit.app](https://hospital-diagnosis-rksta.streamlit.app)

✅ **GitHub Repository**  
👉 [github.com/rakshitha-varma/Hospital-Image-Diagnosis](https://github.com/rakshitha-varma/Hospital-Image-Diagnosis)

---

## 🖼️ Screenshots

| Home Page | Chest X-ray Prediction | Brain Tumor Prediction |
|-----------|------------------------|--------------------------|
| ![home](screenshots/home.png) | ![chest](screenshots/chest_xray.png) | ![brain](screenshots/brain_mri.png) |

---

## 💡 Learning Highlights

- Using CNNs for medical image analysis
- Image preprocessing for model input
- Building responsive UI with Streamlit and Lottie
- Streamlit deployment and hosting live apps
- Practical ML deployment workflow

---

## 🙏 Acknowledgements

Wrapping up this week’s journey with deep gratitude.  
I am sincerely grateful to my mentor for providing unwavering support and thoughtful guidance throughout this journey. Your insights have been instrumental in helping me stay focused, accelerate my learning, and approach challenges with greater clarity and confidence.

---

## 💬 Feedback

I'd love to hear suggestions or ideas to improve this further. Feel free to connect!

---

## 📌 Tags

`#AIProjects` `#DeepLearning` `#Streamlit` `#ComputerVision` `#MedicalAI` `#BeginnerFriendly` `#MLDeployment` `#WomenInTech`
