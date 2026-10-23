
import os
import json
import urllib.request
import urllib.error
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

load_dotenv()

app = FastAPI(
    title="PixelCraft AI API",
    description="AI image generation using Cloudflare Workers AI",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

TEXT_MODEL = "@cf/black-forest-labs/flux-1-schnell"


def cloudflare_image(model: str, payload: dict) -> bytes:
    """Send a prompt to Cloudflare Workers AI and return image bytes."""

    if not ACCOUNT_ID or not API_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Cloudflare credentials are missing from backend/.env",
        )

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{ACCOUNT_ID}/ai/run/{model}"
    )

    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            content = response.read()
            content_type = response.headers.get("Content-Type", "")

        # Some models return image bytes directly.
        if content_type.startswith("image/"):
            return content

        # Some responses contain a base64-encoded image.
        data = json.loads(content.decode("utf-8"))
        result = data.get("result", data)

        if isinstance(result, dict):
            image_data = result.get("image") or result.get("image_b64")

            if image_data:
                import base64
                return base64.b64decode(image_data)

        raise HTTPException(
            status_code=502,
            detail="Cloudflare returned no image.",
        )

    except urllib.error.HTTPError as exc:
        status = exc.code
        provider_error = exc.read().decode("utf-8", errors="replace")

        # Log the provider error without logging credentials.
        print(f"Cloudflare HTTP {status}: {provider_error[:2000]}")

        if status in (401, 403):
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Cloudflare rejected the request (HTTP {status}). "
                    "Check the backend terminal."
                ),
            ) from exc

        if status == 429:
            raise HTTPException(
                status_code=429,
                detail="Cloudflare rate or usage limit reached.",
            ) from exc

        raise HTTPException(
            status_code=502,
            detail=f"Cloudflare AI request failed (HTTP {status}).",
        ) from exc

    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Cloudflare AI.",
        ) from exc

    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Cloudflare returned an invalid image response.",
        ) from exc


@app.get("/")
def root():
    return {
        "service": "PixelCraft AI API",
        "provider": "Cloudflare Workers AI",
        "version": "2.1.0",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "PixelCraft AI API",
        "provider": "Cloudflare Workers AI",
        "cloudflare_configured": bool(ACCOUNT_ID and API_TOKEN),
        "text_to_image_model": TEXT_MODEL,
    }


@app.post("/generate")
def generate(prompt: str = Form(...)):
    """Generate an image from a text prompt."""

    prompt = prompt.strip()

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Prompt is required.",
        )

    if len(prompt) > 2000:
        raise HTTPException(
            status_code=413,
            detail="Prompt must be 2000 characters or fewer.",
        )

    image_bytes = cloudflare_image(
        TEXT_MODEL,
        {"prompt": prompt},
    )

    # Validate the generated image before returning it.
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()

    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Cloudflare response was not a valid image.",
        ) from exc

    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": (
                'inline; filename="pixelcraft-generated.jpg"'
            )
        },
    )
