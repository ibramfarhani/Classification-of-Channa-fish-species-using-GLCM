# %%
import pickle
import cv2
import os
import numpy as np
import json
from skimage.feature import graycomatrix, graycoprops  # Import tambahan untuk GLCM

# %%
IMAGE_SIZE = (256, 256)
DATASETS_DIR = "app/training/datasets"
TRAINING_DIR = f"{DATASETS_DIR}/training"
OUTPUT_DIR = f"{DATASETS_DIR}/pengecekan"


# %%
def preProcessingData(image):
    resized_image = cv2.resize(image, IMAGE_SIZE)  # ubah ukuran gambar menjadi 500x500
    greyscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # grayscale
    trensholding_image = cv2.adaptiveThreshold(
        greyscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 1
    )
    filter_canny_image = cv2.Canny(trensholding_image, 25, 255)
    contours, _ = cv2.findContours(
        filter_canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    max_contour = max(contours, key=cv2.contourArea)

    moments = cv2.moments(max_contour)

    hu_moments = cv2.HuMoments(moments).flatten()

    # Ekstraksi fitur GLCM
    glcm = graycomatrix(greyscale_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    glcm_features = np.array([contrast, dissimilarity, homogeneity, ASM, energy, correlation])

    return np.hstack((hu_moments, glcm_features))  # Menggabungkan fitur Hu Moments dan GLCM


# %%
model = None
labels = []
with open("app/models/knn_model.model", "rb") as knn_model:
    model = pickle.load(knn_model)

with open("app/models/labels.json") as label:
    labels = json.load(label)
    labels = labels

# %%
# load gambar dan lakukan preprocessing
print("===== Memulai Memproses Gambar untuk Pengecekan =====")
root, dirs, _ = next(os.walk(TRAINING_DIR))
container_datas = []
container_labels = []
no = 0
for i, dir in enumerate(dirs):
    folder_jenis = f"{TRAINING_DIR}/{dir}"
    files = next(os.walk(folder_jenis))
    print("===== memproses Folder = ", dir, "=====")
    os.makedirs(f"{OUTPUT_DIR}/{dir}", exist_ok=True)
    for o, file in enumerate(files[2]):
        no += 1
        input = cv2.imread(f"{folder_jenis}/{file}")
        output = preProcessingData(input)
        predict_image = output.reshape(1, -1)
        predict_result = model.predict(predict_image)
        result = labels[predict_result[0]]
        print(f"({file}){dir} === {result}")
        if result == dir:
            cv2.imwrite(f"{OUTPUT_DIR}/{dir}/{file}", input)
        else:
            os.makedirs(f"{OUTPUT_DIR}/{dir}/{result}", exist_ok=True)
            cv2.imwrite(f"{OUTPUT_DIR}/{dir}/{result}/{file}", input)

print("===== Selesai Memproses Gambar untuk Training =====")
