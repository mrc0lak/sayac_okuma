import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import cv2
import numpy as np
import io
import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

app = FastAPI(title="Water Meter Reader AI")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(BASE_DIR / "results"))).resolve()

print("Qwen2-VL modeli yükleniyor, please wait...")
# Modeli float16 formatında yükleyerek VRAM kullanımını yarıya indiriyoruz
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("Model başarıyla yüklendi!")

def get_unrolled_ring(image_bytes):
    """
    OpenCV ile sayacın yuvarlak dış sarı çemberini bulur, 
    keser ve okunabilir yatay, düz bir şerit haline getirir.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Görsel dosyası OpenCV tarafından okunamadı.")

    h, w = img.shape[:2]
    
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * 0.4)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, 1, h // 4,
        param1=50, param2=35, 
        minRadius=int(min(w, h) * 0.25), 
        maxRadius=int(min(w, h) * 0.5)
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, radius = circles[0][0]
        
    min_r = int(radius * 0.8)
    max_r = int(radius * 1.25)
    
    # Kutupsal dönüşüm: Dairesel metni düz bir cetvele çevir
    unrolled = cv2.warpPolar(img, (max_r, 1600), (cx, cy), max_r, cv2.WARP_POLAR_LINEAR)
    ring_strip = unrolled[:, min_r:max_r]
    # Yazının yatay okunabilmesi için şeridi döndür
    ring_strip = cv2.rotate(ring_strip, cv2.ROTATE_90_CLOCKWISE)
    
    return Image.fromarray(cv2.cvtColor(ring_strip, cv2.COLOR_BGR2RGB))


def analyze_image(contents: bytes):
    """Tek bir sayaç görselini analiz eder."""
    if not contents:
        raise ValueError("Boş bir görsel dosyası gönderildi.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Ham Görsel (m3 okuması için)
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    pil_image.thumbnail((1024, 1024))

    # 2. Düzleştirilmiş Şerit Görseli (Seri No okuması için)
    ring_image = get_unrolled_ring(contents)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {
                    "type": "text",
                    "text": "Task: Perform OCR. Read the 5-digit number on the mechanical counter. Output strictly valid JSON."
                },
                {"type": "image", "image": ring_image},
                {
                    "type": "text",
                    "text": "This is a flattened strip of the water meter's outer brass ring. Read the 8-digit serial number engraved on it. Output strictly valid JSON."
                },
                {
                    "type": "text",
                    "text": "Return ONLY as valid JSON format: {\"m3_degeri\": \"...\", \"seri_no\": \"KASKI ...\"}"
                }
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print("\n" + "=" * 30)
    print("GEOMETRİK DÖNÜŞÜM SONRASI MODEL ÇIKTISI:")
    print(output_text)
    print("=" * 30 + "\n")

    json_match = re.search(r'\{.*?\}', output_text.replace('\n', ' '), re.IGNORECASE)
    if not json_match:
        raise ValueError("Model JSON formatında bir çıktı üretmedi.")

    result_dict = json.loads(json_match.group())
    ham_seri = str(result_dict.get("seri_no", "Okunamadı"))
    seri_match = re.search(r'\d{7,8}', ham_seri)
    temiz_seri = "KASKI " + seri_match.group() if seri_match else ham_seri

    return {
        "m3_degeri": str(result_dict.get("m3_degeri", "Okunamadı")),
        "seri_no": temiz_seri
    }


class ResultItem(BaseModel):
    file_name: str
    status: str
    m3_degeri: str
    seri_no: str
    message: Optional[str] = None


class SaveResultsRequest(BaseModel):
    results: List[ResultItem]

@app.post("/api/analyze")
async def analyze_meter(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = analyze_image(contents)
        return JSONResponse(content={"status": "success", **result})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/save-results")
async def save_results(payload: SaveResultsRequest):
    """Ekrandaki sıralanmış sonuçları sunucuda JSON olarak saklar."""
    try:
        if not payload.results:
            return JSONResponse(
                content={"status": "error", "message": "Kaydedilecek sonuç bulunamadı."},
                status_code=400
            )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now().astimezone()
        file_name = f"sayac_sonuclari_{created_at.strftime('%Y%m%d_%H%M%S_%f')}.json"
        output_path = RESULTS_DIR / file_name
        temp_path = RESULTS_DIR / f".{file_name}.tmp"

        result_rows = [
            item.model_dump() if hasattr(item, "model_dump") else item.dict()
            for item in payload.results
        ]
        document = {
            "created_at": created_at.isoformat(timespec="seconds"),
            "total_count": len(result_rows),
            "successful_count": sum(item["status"] == "success" for item in result_rows),
            "results": result_rows
        }

        with temp_path.open("w", encoding="utf-8") as json_file:
            json.dump(document, json_file, ensure_ascii=False, indent=2)
        os.replace(temp_path, output_path)

        return JSONResponse(content={
            "status": "success",
            "file_name": file_name,
            "relative_path": str(Path(RESULTS_DIR.name) / file_name)
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    print(f"Uyarı: Statik dosya klasörü bulunamadı: {STATIC_DIR}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)