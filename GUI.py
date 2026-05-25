import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"D:\iti\tesseract.exe"
)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile
import os
import time

from ocr_passport import ocr_passport

app = FastAPI(
    title="Passport OCR API",
    description="OCR and MRZ extraction for passports",
    version="1.0"
)

# ==========================================
# OUTPUT FOLDERS
# ==========================================
SAVE_DIR = "Output/Images"
os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# READ IMAGE
# ==========================================
def read_image(file: UploadFile):

    try:

        contents = file.file.read()

        nparr = np.frombuffer(contents, np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image")

        return img

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )


# ==========================================
# SAVE TEMP IMAGE
# ==========================================
def save_temp_image(img):

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    cv2.imwrite(temp.name, img)

    return temp.name


# ==========================================
# API
# ==========================================
@app.post("/process-passport")
async def process_passport(
    file: UploadFile = File(...)
):

    start_time = time.time()

    try:

        # ==============================
        # READ IMAGE
        # ==============================
        img = read_image(file)

        # ==============================
        # SAVE ORIGINAL
        # ==============================
        image_path = save_temp_image(img)

        # ==============================
        # OCR
        # ==============================
        extracted_fields, formatted_text = ocr_passport(
            image_path
        )

        processing_time = round(
            time.time() - start_time,
            2
        )

        # ==============================
        # ERROR
        # ==============================
        if not extracted_fields:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": formatted_text,
                    "processing_time_sec": processing_time
                }
            )

        # ==============================
        # SUCCESS
        # ==============================
        return {

            "success": True,

            "processing_time_sec": processing_time,

            "extracted_data": extracted_fields,

            "formatted_text": formatted_text
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )