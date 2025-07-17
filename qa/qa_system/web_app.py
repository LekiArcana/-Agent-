#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 FastAPI 的 Langchain 多 Agent 法律问答系统 Web 应用
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from langchain_qa_system import LangchainLegalQASystem

# 创建 FastAPI 应用
app = FastAPI(
    title="Langchain 多 Agent 法律问答系统",
    description="基于 Langchain 的智能法律咨询平台",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 挂载静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 全局系统实例
qa_system: Optional[LangchainLegalQASystem] = None
system_ready = False
active_sessions: Dict[str, Dict] = {}

# API 模型定义
class QuestionRequest(BaseModel):
    question: str
    answer_style: str = "professional"
    session_id: Optional[str] = None

class SessionRequest(BaseModel):
    title: str = "新对话"

class ChatMessage(BaseModel):
    id: str
    type: str  # "user" or "assistant"
    content: str
    timestamp: str
    session_id: str

class ApiResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Dict] = None

# 系统初始化
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化系统"""
    global qa_system, system_ready
    
    print("🚀 正在启动 Langchain 多 Agent 法律问答系统...")
    
    try:
        qa_system = LangchainLegalQASystem()
        system_ready = await qa_system.initialize()
        
        if system_ready:
            print("✅ 系统初始化成功！Web 服务已就绪")
        else:
            print("❌ 系统初始化失败")
            
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        system_ready = False

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    print("🔄 正在关闭系统...")

# 路由定义

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "system_ready": system_ready
    })

@app.get("/api/system/status")
async def get_system_status():
    """获取系统状态"""
    if not system_ready or not qa_system:
        return JSONResponse({
            "success": False,
            "status": "not_ready",
            "message": "系统未就绪"
        })
    
    try:
        info = qa_system.get_system_info()
        return JSONResponse({
            "success": True,
            "status": "ready",
            "data": info
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"获取系统状态失败: {str(e)}"
        })

@app.post("/api/question")
async def ask_question(request: QuestionRequest):
    """提问接口"""
    if not system_ready or not qa_system:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    try:
        # 处理会话
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            qa_system.create_new_session("Web对话")
        
        # 切换到指定会话
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                "id": session_id,
                "title": "Web对话",
                "created_at": datetime.now().isoformat(),
                "message_count": 0
            }
        
        qa_system.switch_session(session_id)
        
        # 处理问题
        start_time = time.time()
        result = await qa_system.ask_question(
            question=request.question,
            answer_style=request.answer_style,
            show_details=False
        )
        
        # 更新会话信息
        active_sessions[session_id]["message_count"] += 1
        active_sessions[session_id]["last_activity"] = datetime.now().isoformat()
        
        # 构造响应
        if result.get("success"):
            return JSONResponse({
                "success": True,
                "data": {
                    "answer": result["answer"],
                    "session_id": session_id,
                    "execution_time": result.get("execution_time", 0),
                    "answer_style": request.answer_style,
                    "timestamp": datetime.now().isoformat()
                }
            })
        else:
            return JSONResponse({
                "success": False,
                "message": result.get("message", "问答失败"),
                "data": {"session_id": session_id}
            })
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"处理问题时发生错误: {str(e)}"
        })

@app.post("/api/session/create")
async def create_session(request: SessionRequest):
    """创建新会话"""
    if not system_ready or not qa_system:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    try:
        session_id = qa_system.create_new_session(request.title)
        
        # 记录会话信息
        active_sessions[session_id] = {
            "id": session_id,
            "title": request.title,
            "created_at": datetime.now().isoformat(),
            "message_count": 0
        }
        
        return JSONResponse({
            "success": True,
            "data": {
                "session_id": session_id,
                "title": request.title
            }
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"创建会话失败: {str(e)}"
        })

@app.get("/api/sessions")
async def get_sessions():
    """获取所有会话"""
    if not system_ready or not qa_system:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    try:
        # 从系统获取会话列表
        system_sessions = qa_system.list_sessions()
        
        # 合并活跃会话信息
        sessions = []
        for session in system_sessions:
            session_id = session["session_id"]
            session_info = active_sessions.get(session_id, {})
            
            sessions.append({
                "id": session_id,
                "title": session.get("title", "未命名对话"),
                "created_at": session_info.get("created_at", ""),
                "message_count": session_info.get("message_count", 0),
                "last_activity": session_info.get("last_activity", "")
            })
        
        return JSONResponse({
            "success": True,
            "data": {"sessions": sessions}
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"获取会话列表失败: {str(e)}"
        })

@app.post("/api/session/{session_id}/switch")
async def switch_session(session_id: str):
    """切换会话"""
    if not system_ready or not qa_system:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    try:
        success = qa_system.switch_session(session_id)
        
        if success:
            return JSONResponse({
                "success": True,
                "data": {"session_id": session_id}
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "会话不存在或切换失败"
            })
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"切换会话失败: {str(e)}"
        })

@app.get("/api/session/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取指定会话的历史消息"""
    if not system_ready or not qa_system:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    try:
        messages = qa_system.get_session_messages(session_id)
        return JSONResponse({
            "success": True,
            "data": {
                "session_id": session_id,
                "messages": messages
            }
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"获取会话消息失败: {str(e)}"
        })

@app.post("/api/content/analyze")
async def analyze_content(request: dict):
    """内容分析接口"""
    if not system_ready or not qa_system:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    content = request.get("content", "")
    analysis_type = request.get("analysis_type", "legal_content")
    
    if not content.strip():
        return JSONResponse({
            "success": False,
            "message": "内容不能为空"
        })
    
    try:
        result = await qa_system.analyze_content(content, analysis_type)
        
        return JSONResponse({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"内容分析失败: {str(e)}"
        })

@app.get("/api/conversation/summary")
async def get_conversation_summary():
    """获取对话总结"""
    if not system_ready or not qa_system:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    try:
        result = await qa_system.get_conversation_summary()
        
        return JSONResponse({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"获取对话总结失败: {str(e)}"
        })

# 健康检查接口
@app.get("/health")
async def health_check():
    """健康检查"""
    return JSONResponse({
        "status": "healthy" if system_ready else "initializing",
        "timestamp": datetime.now().isoformat(),
        "system_ready": system_ready
    })

# 启动函数
def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """启动 Web 服务器"""
    print(f"🌐 启动 Web 服务器: http://{host}:{port}")
    uvicorn.run("web_app:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    start_server(host="127.0.0.1", port=8000, reload=True) 