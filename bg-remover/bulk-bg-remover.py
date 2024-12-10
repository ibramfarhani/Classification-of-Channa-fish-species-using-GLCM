import os

from PIL import Image
from rembg import remove

DATASETS_DIR = "bg-remover/datasets"
DATASETS_OUTPUT_DIR = "bg-remover/results"
root, dirs, _ = next(os.walk(DATASETS_DIR))
print("Jumlah Jenis Ikan Cupang :", len(dirs))
for dir in dirs:
    folder_jenis = f"{DATASETS_DIR}/{dir}"
    files = next(os.walk(folder_jenis))
    print("===== memproses folder = ", dir, "=====")
    for i, file in enumerate(files[2]):
        print(f"sedang menghapus background {dir} =  {i+1}/{len(files[2])}")
        input = Image.open(f"{folder_jenis}/{file}")
        output = remove(input)
        os.makedirs(f"{DATASETS_OUTPUT_DIR}/{dir}", exist_ok=True)
        rename_file = file.split(".")
        rename_file[-1] = "png"
        output_name = ".".join(rename_file)
        output.save(f"{DATASETS_OUTPUT_DIR}/{dir}/{output_name}")
print("=====SELESAI=====")
