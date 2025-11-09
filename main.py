from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentiment_model import SentimentModel


app = FastAPI()

model = SentimentModel()


class Emotion(BaseModel):
    label: str
    score: float
    percentage: float


class AnalysisRequest(BaseModel):
    text: str


class AnalysisResponse(BaseModel):
    top_emotion: Emotion
    emotions: list[Emotion]


@app.get("/", summary="Root")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "API online",
        "description": "API for analyzing text sentiment",
        "endpoints": {
            "analyze": "POST /analyze - Text analysis",
            "health": "GET /health - API status check"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", summary="API status check")
async def health_check():
    """API status check"""
    return True


@app.post("/analyze", response_model=AnalysisResponse, summary="Text analysis")
async def chat_with_agent(request: AnalysisRequest):
    """Main endpoint for text analysis"""
    result = model.analyze_text(request.text)

    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=result["error"]
        )

    emotions = [Emotion(
        label=emotion["label"],
        score=emotion["score"],
        percentage=round(emotion["score"] * 100, 2)
    ) for emotion in result["emotions"]]

    return AnalysisResponse(
        top_emotion=emotions[0],
        emotions=emotions
    )
