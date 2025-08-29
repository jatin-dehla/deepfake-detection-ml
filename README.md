# Deepfake Detection using ResNeXt + LSTM

This project implements a **deepfake video detection pipeline** using **ResNeXt50** for frame-level feature extraction and **LSTM** for temporal modeling.  
You will be able to preprocess the dataset, train a PyTorch model of your own, and predict on new unseen data using your trained model.

---

### Note
We recommend using [Google Colab](https://colab.research.google.com/) for running the notebooks in this project.

---

## Dataset
Some of the datasets used in this project are listed below:
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data)

---

## Preprocessing
Steps performed before training:
- Load the dataset  
- Split each video into frames  
- Detect and crop the face from each frame  
- Save the cropped frames/videos for training  

Notebook: `notebooks/preprocessing.ipynb`

---

## Model and Training
- Load preprocessed videos and labels (from CSV)  
- Build a PyTorch model using transfer learning with **ResNeXt50 + LSTM**  
- Split the data into training and testing sets  
- Train the model  
- Test and evaluate performance  
- Save the trained model in `.pt` format  

Notebook: `notebooks/deepfake_starter_kit.ipynb`

---

## Prediction
- Load the saved PyTorch model (`.pt`)  
- Predict whether a new unseen video is **real** or **fake**, based on trained weights  

Notebook: `notebooks/predict.ipynb`

---

## Helpers
Code in the helper notebooks and scripts may assist with preprocessing tasks, such as:
- Converting JSON label files into CSV (`scripts/label_json_to_csv.py`)  
- Copying files from one directory to another (`notebooks/dataset_split_real_fake.ipynb`)  
- Handling class imbalance (`notebooks/balance_data.ipynb`)  
- Removing audio-altered files from DFDC dataset (`notebooks/remove_audio_altered_files.ipynb`)  

---

## Helpful Links

### Preprocessed Data
- [Celeb-DF Fake processed videos](https://drive.google.com/drive/folders/1SxCb_Wr7N4Wsc-uvjUl0i-6PpwYmwN65?usp=sharing)  
- [Celeb-DF Real processed videos](https://drive.google.com/drive/folders/1g97v9JoD3pCKA2TxHe8ZLRe4buX2siCQ?usp=sharing)  
- [FaceForensics++ Real and Fake processed videos](https://drive.google.com/drive/folders/1VIIWRLs6VBXRYKODgeOU7i6votLPPxT0?usp=sharing)  
- [DFDC Fake processed videos](https://drive.google.com/drive/folders/1yz3DBeFJvZ_QzWsyY7EwBNm7fx4MiOfF?usp=sharing)  
- [DFDC Real processed videos](https://drive.google.com/drive/folders/1wN3ZOd0WihthEeH__Lmj_ENhoXJN6U11?usp=sharing)  

**Note:** Labels for all the above preprocessed data are under `/label/Gobal_metadata.csv`.

---

### Trained Models
You can also directly use pre-trained models:
- [Download trained models](https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-?usp=sharing)  

Then simply run the prediction notebook (`predict.ipynb`) for inference.

---

## Usage Order
1. Preprocess dataset → `preprocessing.ipynb`  
2. (Optional) Balance dataset → `balance_data.ipynb`  
3. Train model → `deepfake_starter_kit.ipynb`  
4. Save trained weights → `models/`  
5. Predict new videos → `predict.ipynb`  

---

## System Architecture
![System Architecture](System%20Architecture.png)

---

## Acknowledgements
- Datasets: [FaceForensics++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics), [DFDC](https://www.kaggle.com/c/deepfake-detection-challenge/data)  
- Model inspired by deepfake detection research using **CNN + LSTM** architectures.  

---

***If you need any help regarding the project, please contact us. We will be happy to help.***  
Maintainer: **Jatin Choudhary (IIT Hyderabad)**
