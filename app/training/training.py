# %%
# import library yang diperlukan
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import json
import xlsxwriter
from skimage.feature import graycomatrix, graycoprops  # Import tambahan untuk GLCM

# %%
# Konfigurasi project ukuran gambar dan folder datasets
IMAGE_SIZE = (256, 256)
DATASETS_DIR = "app/training/datasets"
TRAINING_DIR = f"{DATASETS_DIR}/training"
TESTING_DIR = f"{DATASETS_DIR}/testing"

# %%
# fungsi preProcessing gambar
def preProcessingData(image):
    resized_image = cv2.resize(
        image, IMAGE_SIZE
    )  # ubah ukuran gambar menjadi (256, 256)
    greyscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # grayscale
    trensholding_image = cv2.adaptiveThreshold(
        greyscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 1
    )  # trensholding gambar
    filter_canny_image = cv2.Canny(trensholding_image, 25, 255)  # filter canny

    # fungsi invariant moment
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

# persiapkan variable penampung untuk class dan data x_train,y_train,x_test,y_test
classes = []
x_train = []
y_train = []
x_test = []
y_test = []

# %%
# setup data header untuk excel extraksi fitur
extraksi_fitur = []
label_fitur = ["Nomor", "Jenis", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "Contrast", "Dissimilarity", "Homogeneity", "ASM", "Energy", "Correlation"]
extraksi_fitur.append(label_fitur)

print("===== Memulai Memproses Gambar untuk Training =====")
root, dirs, _ = next(os.walk(TRAINING_DIR))
container_datas = []
container_labels = []
no = 0
for i, dir in enumerate(dirs):
    folder_jenis = f"{TRAINING_DIR}/{dir}"
    files = next(os.walk(folder_jenis))
    print("===== memproses Folder = ", dir, "=====")
    data = []
    labels = []
    for o, file in enumerate(files[2]):
        no += 1
        print(f"memproses gambar {dir} =  {o+1}/{len(files[2])}")
        input = cv2.imread(f"{folder_jenis}/{file}")
        output = preProcessingData(input)
        data.append(output)
        labels.append(i)
        excel_data = [no, dir] + output.tolist()
        extraksi_fitur.append(excel_data)
    container_datas.append(data)
    container_labels.append(labels)
    classes.append(dir)
print("===== Selesai Memproses Gambar untuk Training =====")
x_train = np.concatenate(container_datas)
y_train = np.concatenate(container_labels)

# %%
# digunakan untuk mengambil datasets testing dan melakukan preprocessing setelah itu hasil dari data gambar disimpan di variabel x_train dan label disimpan di y_test
print("===== Memulai Memproses Gambar untuk Testing =====")
root, dirs, _ = next(os.walk(TESTING_DIR))
container_datas = []
container_labels = []
for i, dir in enumerate(dirs):
    folder_jenis = f"{TESTING_DIR}/{dir}"
    files = next(os.walk(folder_jenis))
    print("===== memproses Folder = ", dir, "=====")
    data = []
    labels = []
    for o, file in enumerate(files[2]):
        print(f"memproses gambar {dir} =  {o+1}/{len(files[2])}")
        input = cv2.imread(f"{folder_jenis}/{file}")
        output = preProcessingData(input)
        data.append(output)
        labels.append(i)
    container_datas.append(data)
    container_labels.append(labels)
print("===== Selesai Memproses Gambar untuk Testing =====")
x_test = np.concatenate(container_datas)
y_test = np.concatenate(container_labels)

# %%
# inisialisasi model KKN dengan nilai N adalah 5 dan training dengan menggunakan data yang sudah dipersiapkan sebelumnya
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

# %%
# testing model yang telah dilatih menggunakan data testing
y_pred = model.predict(x_test)

# %%
# generate clasification report
print(classification_report(y_test, y_pred, target_names=classes, zero_division=1))

# %%
# simpan model dan label untuk digunakan di aplikasi web
print("===== Menyimpan Model =====")
with open("app/models/knn_model.model", "wb") as knn_model:
    pickle.dump(model, knn_model)

print("===== Menyimpan Label/Jenis Ikan Channa =====")
with open("app/models/labels.json", "w", encoding="utf-8") as f:
    json.dump(classes, f, ensure_ascii=False, indent=4)

# %%
# simpan data extraksi fitur dalam bentuk excel
print("===== Menyimpan Excel  =====")
workbook = xlsxwriter.Workbook("app/models/hasil.xlsx")
worksheet = workbook.add_worksheet("extraksi fitur")
for baris, row in enumerate(extraksi_fitur):
    for kolom, col in enumerate(row):
        worksheet.write(baris, kolom, col)
workbook.close()

# %%
print("===== Selesai =====")

# %%
# Generate Confusional Matrix
fig, ax = plt.subplots(figsize=(8, 8))
cmatrix = confusion_matrix(y_test, y_pred)
sns.color_palette("hls", 8)
sns.heatmap(
    cmatrix,
    cmap="rocket",
    annot=True,
    fmt=".4g",
    linewidths=1,
    linecolor="white",
)

ax.set_title("Confusion Matrix", fontsize=20, pad=24)
ax.set_xticklabels(labels=classes, fontsize=18)
ax.set_yticklabels(labels=classes, fontsize=18)

plt.xlabel("(y) Prediksi", fontsize=16, color="black", labelpad=24)
plt.ylabel("(y) Sebenarnya", fontsize=16, color="black", labelpad=24)
plt.show()

# %%
# cetak akurasi dari model yang telah di testing menggunakan data testing
print(f"Akurasi = {accuracy_score(y_test, y_pred)}")
