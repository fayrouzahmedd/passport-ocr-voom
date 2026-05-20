import sys
import os
import json
from datetime import datetime, date
import numpy as np
import pytesseract
from passporteye import read_mrz
#import face_recognition
import cv2
from deep_translator import GoogleTranslator

# Set the Tesseract OCR executable path.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image, image_path, output_folder):

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Slight upscale for better MRZ reading
    gray_image = cv2.resize(
        gray_image,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    # Light blur only
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Mild sharpening
    sharpened_image = cv2.addWeighted(
        gray_image,
        1.5,
        blurred_image,
        -0.5,
        0
    )

    # Save sharpened image
    preprocessed_image_path = os.path.join(
        output_folder,
        os.path.splitext(os.path.basename(image_path))[0]
        + "_preprocessed_image.jpg"
    )

    cv2.imwrite(preprocessed_image_path, sharpened_image)

    return preprocessed_image_path

def preprocess_image_light(image, image_path, output_folder):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    sharpened_image = cv2.addWeighted(
        gray_image,
        2,
        blurred_image,
        -1,
        0
    )

    preprocessed_image_path = os.path.join(
        output_folder,
        os.path.splitext(os.path.basename(image_path))[0]
        + "_light_preprocessed.jpg"
    )

    cv2.imwrite(preprocessed_image_path, sharpened_image)

    return preprocessed_image_path

def find_issuing_date(image_path, extracted_fields, face_recognition_message):
    # Preprocess image for OCR with Tesseract.
    image = cv2.imread(image_path, 0)

    # Apply Otsu thresholding.
    _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Perform OCR using Tesseract.
    extracted_text = pytesseract.image_to_string(thresh_image, lang='eng')

    # Find dates in format DD/MM/YYYY.
    import re

    date_pattern = r"\b(\d{2}/\d{2}/\d{4})\b"
    dates = re.findall(date_pattern, extracted_text)

    # Remove known dates (Expiry Date and Date of Birth).
    known_dates = [extracted_fields['expiry_date'], extracted_fields['date_of_birth']]
    dates = [date for date in dates if date not in known_dates]

    # # Print extracted text.
    # print("Extracted Text:")
    # print(extracted_text)

    # Print extracted dates.
    print("\nExtracted Date of Issue which is not in the MRZ:")
    if len(dates) != 0:
        for date in dates:
            print(date)
    else:
        print("No Issuing Date in DD/MM/YYYY format.")

    # Add extracted date of issue to fields.
    extracted_fields['date_of_issue'] = dates[0]

    # Update the formatted text.
    formatted_text = f"Name: {extracted_fields['name']}\n" \
                     f"Expiry Date: {extracted_fields['expiry_date']}\n" \
                     f"Date of Birth: {extracted_fields['date_of_birth']}\n" \
                     f"Passport Number: {extracted_fields['passport_number']}\n" \
                     f"Nationality: {extracted_fields['nationality']}\n"
    formatted_text += face_recognition_message

    return extracted_fields, formatted_text


# def extract_passport_portrait(input_path):
#     # Create the Portraits folder in the Output folder if it doesn't exist.    
#     portrait_output_folder = os.path.join(sys.path[0], "Output", "Portraits")
#     os.makedirs(portrait_output_folder, exist_ok=True)

#     # Load the input image.
#     image = face_recognition.load_image_file(input_path)

#     # Use face_recognition library for face detection.
#     face_locations = face_recognition.face_locations(image)

#     if len(face_locations) == 0:
#         return "\nNo faces found in the image."

#     # Assume the first detected face corresponds to the person's portrait.
#     top, right, bottom, left = face_locations[0]

#     # Calculate the new coordinates to increase the portrait area by 1.6 times.
#     width = right - left
#     height = bottom - top
#     new_width = int(width * 1.4)
#     new_height = int(height * 1.8)

#     # Calculate the adjustment to keep the portrait centered.
#     width_diff = new_width - width
#     height_diff = new_height - height
#     left -= width_diff // 2
#     right += width_diff - width_diff // 2
#     top -= height_diff // 2
#     bottom += height_diff - height_diff // 2

#     # Adjust the top and bottom coordinates to move the portrait higher by 0.15 of the height.
#     height = bottom - top
#     height_adjustment = int(height * 0.05)
#     top -= height_adjustment
#     bottom -= height_adjustment

#     # Ensure the coordinates are within the image bounds.
#     left = max(left, 0)
#     right = min(right, image.shape[1])
#     top = max(top, 0)
#     bottom = min(bottom, image.shape[0])

#     # Extract the person's portrait from the image.
#     portrait = image[top:bottom, left:right]

#     # Convert color space from BGR to RGB.
#     portrait_rgb = cv2.cvtColor(portrait, cv2.COLOR_BGR2RGB)

#     # Save the portrait image to the Output folder.
#     output_path = os.path.join(portrait_output_folder, 
#                                os.path.splitext(os.path.basename(input_path))[0] + "_portrait.jpg")
#     cv2.imwrite(output_path, portrait_rgb)
    
#     return "\nPortrait extracted and saved successfully."


def clean_mrz_text(text):

    if not text:
        return ""

    # Replace MRZ separators with spaces
    text = text.replace("<", " ")

    # Common OCR mistakes
    text = text.replace("CCC", " ")
    text = text.replace("CC", " ")

    # Remove repeated spaces
    text = " ".join(text.split())

    return text.strip().upper()
def translate_to_arabic(text):

    try:

        translated = GoogleTranslator(
            source='auto',
            target='ar'
        ).translate(text)

        return translated

    except Exception:

        return text
NATIONALITY_CODES = {

    "EGY": {
        "en": "Egyptian",
        "ar": "مصري"
    },

    "KWT": {
        "en": "Kuwaiti",
        "ar": "كويتي"
    },

    "SAU": {
        "en": "Saudi",
        "ar": "سعودي"
    },

    "ARE": {
        "en": "Emirati",
        "ar": "إماراتي"
    },

    "QAT": {
        "en": "Qatari",
        "ar": "قطري"
    },

    "BHR": {
        "en": "Bahraini",
        "ar": "بحريني"
    },

    "OMN": {
        "en": "Omani",
        "ar": "عماني"
    },

    "JOR": {
        "en": "Jordanian",
        "ar": "أردني"
    },

    "LBN": {
        "en": "Lebanese",
        "ar": "لبناني"
    },

    "SYR": {
        "en": "Syrian",
        "ar": "سوري"
    },

    "IRQ": {
        "en": "Iraqi",
        "ar": "عراقي"
    },

    "YEM": {
        "en": "Yemeni",
        "ar": "يمني"
    },

    "USA": {
        "en": "American",
        "ar": "أمريكي"
    },

    "GBR": {
        "en": "British",
        "ar": "بريطاني"
    },

    "FRA": {
        "en": "French",
        "ar": "فرنسي"
    },

    "DEU": {
        "en": "German",
        "ar": "ألماني"
    },

    "ITA": {
        "en": "Italian",
        "ar": "إيطالي"
    },

    "ESP": {
        "en": "Spanish",
        "ar": "إسباني"
    },

    "TUR": {
        "en": "Turkish",
        "ar": "تركي"
    },

    "RUS": {
        "en": "Russian",
        "ar": "روسي"
    },

    "CHN": {
        "en": "Chinese",
        "ar": "صيني"
    },

    "JPN": {
        "en": "Japanese",
        "ar": "ياباني"
    },

    "IND": {
        "en": "Indian",
        "ar": "هندي"
    },

    "PAK": {
        "en": "Pakistani",
        "ar": "باكستاني"
    },

    "CAN": {
        "en": "Canadian",
        "ar": "كندي"
    },

    "CMR": {
        "en": "Cameroonian",
        "ar": "كاميروني"
    },

    "AUS": {
        "en": "Australian",
        "ar": "أسترالي"
    }
}

def extract_fields(mrz_data):

    fields = {
        "name": "",
        "name_ar": "",
        "expiry_date": "",
        "date_of_birth": "",
        #"date_of_issue": "",
        "passport_number": "",
        "nationality": "",
        "nationality_ar": "",
        #"sex": "",
        #"sex_ar": ""
    }

    # ==============================
    # NAME
    # ==============================
    surname = mrz_data.get("surname", "")
    names = mrz_data.get("names", "")

    full_name = f"{names} {surname}"

    # Replace MRZ separators
    full_name = full_name.replace("<", " ")

    # Remove repeated spaces
    full_name = " ".join(full_name.split())

    # Common OCR mistakes
    full_name = full_name.replace("CCC", " ")
    full_name = full_name.replace("CC", " ")

    # Clean spaces again
    full_name = " ".join(full_name.split())

    fields["name"] = full_name.upper()

    # ==============================
    # NAME ARABIC TRANSLATION
    # ==============================
    try:

        from deep_translator import GoogleTranslator

        fields["name_ar"] = GoogleTranslator(
            source='auto',
            target='ar'
        ).translate(fields["name"])

    except Exception:

        fields["name_ar"] = fields["name"]

    # ==============================
    # PASSPORT NUMBER
    # ==============================
    passport_number = mrz_data.get("number", "")

    passport_number = passport_number.replace("O", "0")
    passport_number = passport_number.replace("o", "0")
    passport_number = passport_number.replace("<", "")

    fields["passport_number"] = passport_number

    # ==============================
    # NATIONALITY
    # ==============================
    nationality_code = mrz_data.get("country", "")

    nationality_code = nationality_code.replace("<", "")
    nationality_code = nationality_code.upper()

    nationality_info = NATIONALITY_CODES.get(
        nationality_code,
        {
            "en": "Unknown",
            "ar": "غير معروف"
        }
    )

    nationality_en = nationality_info["en"]
    nationality_ar = nationality_info["ar"]

    fields["nationality"] = (
        f"{nationality_code} - {nationality_en}"
    )

    fields["nationality_ar"] = nationality_ar

    # # ==============================
    # # SEX
    # # ==============================
    # sex = mrz_data.get("sex", "").upper()

    # sex = sex.replace("<", "")

    # if sex not in ["M", "F"]:
    #     sex = "UNKNOWN"

    # fields["sex"] = sex

    # # Arabic sex translation
    # if sex == "M":
    #     sex_ar = "ذكر"

    # elif sex == "F":
    #     sex_ar = "أنثى"

    # else:
    #     sex_ar = "غير معروف"

    # fields["sex_ar"] = sex_ar

    # ==============================
    # EXPIRY DATE
    # ==============================
    expiry_date = mrz_data.get("expiration_date", "")

    expiry_date = expiry_date.replace("O", "0")
    expiry_date = expiry_date.replace("o", "0")
    expiry_date = expiry_date.replace("<", "")

    if len(expiry_date) == 6:

        expiry_date = datetime.strptime(
            expiry_date,
            "%y%m%d"
        ).strftime("%d/%m/%Y")

    fields["expiry_date"] = expiry_date

    # ==============================
    # DATE OF BIRTH
    # ==============================
    dob = mrz_data.get("date_of_birth", "")

    dob = dob.replace("O", "0")
    dob = dob.replace("o", "0")
    dob = dob.replace("<", "")

    if len(dob) == 6:

        dob = datetime.strptime(
            dob,
            "%y%m%d"
        ).strftime("%d/%m/%Y")

        # Fix future-century issue
        if datetime.strptime(
            dob,
            "%d/%m/%Y"
        ).date() > date.today():

            dob = dob[:-4] + str(int(dob[-4:]) - 100)

    fields["date_of_birth"] = dob

    # ==============================
    # FORMAT OUTPUT
    # ==============================
    formatted_text = (

        f"Name: {fields['name']}\n"
        f"الاسم: {fields['name_ar']}\n\n"

        f"Expiry Date: {fields['expiry_date']}\n"
        f"تاريخ الانتهاء: {fields['expiry_date']}\n\n"

        f"Date of Birth: {fields['date_of_birth']}\n"
        f"تاريخ الميلاد: {fields['date_of_birth']}\n\n"

        f"Passport Number: {fields['passport_number']}\n"
        f"رقم جواز السفر: {fields['passport_number']}\n\n"

        # f"Sex: {fields['sex']}\n"
        # f"النوع: {fields['sex_ar']}\n\n"

        f"Nationality: {fields['nationality']}\n"
        f"الجنسية: {fields['nationality_ar']}\n"
    )

    return fields, formatted_text

def ocr_passport(image_path):

    output_folder = os.path.join(sys.path[0], "Output")
    os.makedirs(output_folder, exist_ok=True)

    json_output_folder = os.path.join(
        output_folder,
        "JSON_Files"
    )
    os.makedirs(json_output_folder, exist_ok=True)

    face_recognition_message = ""

    # ==============================
    # LOAD IMAGE
    # ==============================
    image = cv2.imread(image_path)

    if image is None:
        return {}, "Could not read passport image."

    # ==============================
    # TRY STRONG PREPROCESSING
    # ==============================
    preprocessed_image_path = preprocess_image(
        image,
        image_path,
        output_folder
    )

    mrz = read_mrz(preprocessed_image_path)

    # ==============================
    # FALLBACK TO LIGHT PREPROCESSING
    # ==============================
    if mrz is None:

        light_preprocessed_path = preprocess_image_light(
            image,
            image_path,
            output_folder
        )

        mrz = read_mrz(light_preprocessed_path)

    # ==============================
    # FINAL FAILURE
    # ==============================
    if mrz is None:
        return {}, "Could not detect MRZ in passport image."

    # ==============================
    # EXTRACT MRZ DATA
    # ==============================
    mrz_data = mrz.to_dict()

    extracted_fields, formatted_text = extract_fields(
        mrz_data
    )

    formatted_text += face_recognition_message

    # ==============================
    # FIND ISSUING DATE
    # ==============================
    try:

        extracted_fields, formatted_text = find_issuing_date(
            image_path,
            extracted_fields,
            face_recognition_message
        )

    except IndexError:
        pass

    except Exception:
        pass

    # ==============================
    # SAVE JSON OUTPUT
    # ==============================
    output_file = os.path.join(
        json_output_folder,
        os.path.splitext(
            os.path.basename(image_path)
        )[0] + ".json"
    )

    with open(output_file, "w") as file:

        json.dump(
            extracted_fields,
            file,
            indent=4
        )

    return extracted_fields, formatted_text