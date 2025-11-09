from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from datetime import datetime
import logging

from double_agent import DoubleAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Double Agent API",
    description="API для интеллектуального ассистента с календарем и шутками",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = DoubleAgent()

sessions: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    timestamp: str


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    last_activity: str
    history_length: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agent_model: str
    active_sessions: int


# Endpoints
@app.get("/", summary="Корневой эндпоинт")
async def root():
    """Информация о API и доступные endpoints"""
    return {
        "message": "🤖 Double Agent API работает!",
        "description": "Интеллектуальный ассистент с календарем и шутками",
        "endpoints": {
            "chat": "POST /chat - Общение с агентом",
            "health": "GET /health - Статус сервиса",
            "sessions": "GET /sessions - Список сессий",
            "delete_session": "DELETE /sessions/{session_id} - Удаление сессии"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", response_model=HealthResponse, summary="Проверка здоровья сервиса")
async def health_check():
    """Проверка статуса API и агента"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agent_model=agent.model,
        active_sessions=len(sessions)
    )


@app.post("/chat", response_model=ChatResponse, summary="Общение с агентом")
async def chat_with_agent(request: ChatRequest):
    """
    Основной endpoint для взаимодействия с агентом.

    Поддерживает контекст разговора через session_id.

    **Примеры запросов:**
    - "Расскажи шутку про программистов"
    - "Какие события у меня на завтра?"
    - "Создай встречу с клиентом на пятницу в 15:00"
    """
    try:
        session_id = request.session_id or "default"
        logger.info(f"Обработка запроса для сессии: {session_id}")

        if session_id not in sessions:
            sessions[session_id] = {
                "history": [],
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }

        sessions[session_id]["last_activity"] = datetime.now().isoformat()

        session_history = sessions[session_id]["history"]
        agent.conversation_history = session_history.copy()

        response_text = agent.process_query(request.message)

        sessions[session_id]["history"] = agent.conversation_history.copy()

        logger.info(f"Запрос обработан успешно для сессии: {session_id}")

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке запроса: {str(e)}"
        )


@app.get("/sessions", summary="Список активных сессий")
async def list_sessions():
    """Получение информации о всех активных сессиях"""
    sessions_info = []
    for session_id, data in sessions.items():
        sessions_info.append(SessionInfo(
            session_id=session_id,
            created_at=data["created_at"],
            last_activity=data["last_activity"],
            history_length=len(data["history"])
        ))

    return {
        "total_sessions": len(sessions),
        "sessions": sessions_info
    }


@app.delete("/sessions/{session_id}", summary="Удаление сессии")
async def delete_session(session_id: str):
    """Удаление конкретной сессии по ID"""
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Сессия {session_id} удалена")
        return {"message": f"Сессия {session_id} удалена"}
    else:
        raise HTTPException(status_code=404, detail="Сессия не найдена")


@app.delete("/sessions", summary="Очистка всех сессий")
async def clear_all_sessions():
    """Удаление всех сессий"""
    sessions_count = len(sessions)
    sessions.clear()
    logger.info(f"Все сессии удалены. Всего удалено: {sessions_count}")
    return {"message": f"Все сессии удалены. Всего удалено: {sessions_count}"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик исключений"""
    logger.error(f"Глобальная ошибка: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервера"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
