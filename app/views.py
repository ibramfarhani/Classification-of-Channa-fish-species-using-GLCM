import config
import cv2
import numpy
from flask import Flask, current_app, render_template, request, send_file
from rembg import remove
from utils import encode_from_cv2, generate_excel, NumpyEncoder
import json
from skimage.feature import graycomatrix, graycoprops

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("identifikasi.html")


@app.route("/download/excel", methods=["GET"])
def download_excel():
    return send_file("temp/matrix.xlsx", as_attachment=True)


@app.route("/api/identifikasi", methods=["POST"])
def identifikasi():
    response = {
        "success": False,
    }
    gambar = request.files.get("gambar")
    if not gambar:
        response.update({"error": "Silahkan pilih gambar terlebih dahulu"})
        return response
    elif gambar.mimetype not in config.ALLOWED_MIMETYPE:
        response.update({"error": "Gambar Harus Berformat jpg,jpeg,png."})
        return response
    try:
        gambar = gambar.read()
        file_bytes = numpy.fromstring(gambar, numpy.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        resized_image = cv2.resize(img, (256, 256))
        # extract RGB
        resized_image_blue_channel = resized_image[:, :, 0]
        resized_image_green_channel = resized_image[:, :, 1]
        resized_image_red_channel = resized_image[:, :, 2]

        removed_bg = remove(resized_image)
        # extract RGB
        removed_bg_blue_channel = removed_bg[:, :, 0]
        removed_bg_green_channel = removed_bg[:, :, 1]
        removed_bg_red_channel = removed_bg[:, :, 2]

        greyscale = cv2.cvtColor(removed_bg, cv2.COLOR_BGR2GRAY)
        trensholding = cv2.adaptiveThreshold(
            greyscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 1
        )

        filter_canny = cv2.Canny(trensholding, 25, 255)
        contours, _ = cv2.findContours(
            filter_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        max_contour = max(contours, key=cv2.contourArea)

        moments = cv2.moments(max_contour)

        hu_moments = cv2.HuMoments(moments)

        feature = hu_moments.flatten()

        # Calculate GLCM features
        # You need to specify the distances and angles for graycomatrix
        # Here I'm using a distance of 1 pixel and an angle of 0 degrees
        glcm = graycomatrix(greyscale, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        ASM = graycoprops(glcm, 'ASM')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        # Combine features
        glcm_features = numpy.array([contrast, dissimilarity, homogeneity, ASM, energy, correlation])
        feature = numpy.hstack((hu_moments.flatten(), glcm_features))

        # prediksi
        model = current_app.model
        labels = current_app.labels
        predict_image = feature.reshape(1, -1)
        predict_result = model.predict(predict_image)
        result = labels[predict_result[0]]
        feature = feature.tolist()
        extraksi_fitur = []
        label_fitur = [ "M1", "M2", "M3", "M4", "M5", "M6", "M7", "Contrast", "Dissimilarity", "Homogeneity", "ASM", "Energy", "Correlation"]
        extraksi_fitur.append(label_fitur)
        extraksi_fitur.append(feature)
        excel_data = {
            "Resize 256x256 (RED)": resized_image_red_channel,
            "Resize 256x256 (GREEN)": resized_image_green_channel,
            "Resize 256x256 (BLUE)": resized_image_blue_channel,
            "Remove Backgorund (RED)": removed_bg_red_channel,
            "Remove Backgorund (GREEN)": removed_bg_green_channel,
            "Remove Backfeaturegorund (BLUE)": removed_bg_blue_channel,
            "Greyscale": greyscale,
            "Trensholding": trensholding,
            # "Edge Detector (Canny)": filter_canny,
            "Ektraksi Fitur": extraksi_fitur,
        }
        generate_excel(excel_data)
        print(feature)
        response.update(
            {
                "success": True,
                "gambar": {
                    "removed_bg": encode_from_cv2(removed_bg),
                    "greyscale": encode_from_cv2(greyscale),
                    "trensholding": encode_from_cv2(trensholding),
                    "edge_detector": encode_from_cv2(filter_canny),
                },
                # "matrix": {
                #     "edge_detector": json.dumps(filter_canny, cls=NumpyEncoder),
                # },
                "ektraksi_fitur": extraksi_fitur,
                "jenis": result,
            }
        )
        return response
    except Exception as e:
        print("Error [identifikasi()]: {}".format(e))
        response.update({"error": "Terjadi Kesalahan Sistem"})
        return response