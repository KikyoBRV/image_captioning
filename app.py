from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import requests
import os
from io import BytesIO
from starlette.requests import Request

# Load environment variables from .env file
load_dotenv()

# Create FastAPI app instance
app = FastAPI()

# Set up the template directory
templates = Jinja2Templates(directory="templates")

# API details for Hugging Face
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
headers = {"Authorization": f"Bearer {API_KEY}"}


# Function to query the Hugging Face API for captioning
def query(image_data):
    response = requests.post(API_URL, headers=headers, data=image_data)
    if response.status_code == 200:
        result = response.json()
        # The response is a list; extract the first item's caption
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No caption generated.")
        else:
            return "No caption generated."
    else:
        return {"error": "Failed to communicate with the captioning service"}



# Serve static files (optional if you need custom CSS or JS files)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Index route to render the HTML page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route to handle image upload and caption retrieval
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image_data = await file.read()
    caption = query(image_data)

    if isinstance(caption, dict) and "error" in caption:
        return JSONResponse(content=caption, status_code=400)

    return JSONResponse(content={"caption": caption})

