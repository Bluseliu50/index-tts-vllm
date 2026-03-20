import os
import asyncio
import io
import traceback
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import time
import soundfile as sf
from typing import List, Optional, Union

from loguru import logger
logger.add("logs/api_server_v2.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

from indextts.infer_vllm_v2 import IndexTTS2

tts = None


def load_project_tts_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    config_path = os.path.join(project_root, "config.json")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return {}
    tts_cfg = config.get("tts", {}) if isinstance(config, dict) else {}
    return tts_cfg if isinstance(tts_cfg, dict) else {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
import tempfile
import shutil

PROJECT_TTS_CONFIG = load_project_tts_config()
API_KEY = (
    str(PROJECT_TTS_CONFIG.get("api_key") or "").strip()
    or os.environ.get("INDEX_TTS_API_KEY", "")
    or "transvideo-sk-9a8a-bf47e304c379"
)
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == f"Bearer {API_KEY}":
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")

@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(
    text: str = Form(...),
    spk_audio: UploadFile = File(...),
    emo_control_method: int = Form(0),
    emo_audio: Optional[UploadFile] = File(None),
    emo_weight: float = Form(1.0),
    emo_vec_str: Optional[str] = Form(None),
    emo_text: Optional[str] = Form(None),
    emo_random: bool = Form(False),
    max_text_tokens_per_sentence: int = Form(120),
    api_key: str = Security(get_api_key)
):
    temp_dir = tempfile.mkdtemp()
    try:
        # Parse emo_vec from JSON string if provided
        emo_vec = [0.0] * 8
        if emo_vec_str:
            import json
            emo_vec = json.loads(emo_vec_str)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
            
        spk_audio_path = os.path.join(temp_dir, spk_audio.filename)
        with open(spk_audio_path, "wb") as buffer:
            shutil.copyfileobj(spk_audio.file, buffer)

        emo_ref_path = None
        if emo_audio is not None and emo_control_method == 1:
            emo_ref_path = os.path.join(temp_dir, "emo_" + emo_audio.filename)
            with open(emo_ref_path, "wb") as buffer:
                shutil.copyfileobj(emo_audio.file, buffer)

        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        # logger.info(f"Emo control mode:{emo_control_method}, vec:{vec}")
        sr, wav = await tts.infer(spk_audio_prompt=spk_audio_path, text=text,
                        output_path=None,
                        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                        emo_vector=vec,
                        use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                        max_text_tokens_per_sentence=int(max_text_tokens_per_sentence))
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    local_host = str(PROJECT_TTS_CONFIG.get("local_host", "0.0.0.0")).strip() or "0.0.0.0"
    try:
        local_port = int(PROJECT_TTS_CONFIG.get("local_port", 6006))
    except (TypeError, ValueError):
        local_port = 6006
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=local_host)
    parser.add_argument("--port", type=int, default=local_port)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM", help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)
