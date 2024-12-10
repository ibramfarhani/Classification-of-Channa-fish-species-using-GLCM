import base64
import json

import cv2
import numpy as np
import xlsxwriter


def encode_from_cv2(img):
    bin = cv2.imencode(".jpg", img)[1]
    return str(base64.b64encode(bin))[2:-1]


def write_worksheet(workbook, name, matrix):
    worksheet = workbook.add_worksheet(name)
    for baris, row in enumerate(matrix):
        for kolom, col in enumerate(row):
            worksheet.write(baris, kolom, col)


def generate_excel(data):
    workbook = xlsxwriter.Workbook("app/temp/matrix.xlsx")
    for key, value in data.items():
        write_worksheet(workbook, key, value)
    workbook.close()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
