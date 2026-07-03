import sys
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import uvicorn

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from agent.graph_builder import hr_agent_app


# =====================================================================
# 1. 异步后台闲置监控守护
# =====================================================================
class SessionTimerManager:
    def __init__(self, timeout_seconds: int = 30):
        self.timers = {}
        self.timeout_seconds = timeout_seconds

    def _trigger_summary(self, thread_id: str, uid: str):
        print(f"\n👻 [后台守护] 监控到会话 {thread_id} 闲置超时，正在后台静默生成总结...")
        config = {"configurable": {"thread_id": thread_id}}
        idle_state = {
            "messages": [HumanMessage(content="__SYS_IDLE_TIMEOUT__")],
            "current_uid": uid,
            "loop_step": 0
        }
        try:
            for _ in hr_agent_app.stream(idle_state, config, stream_mode="values"):
                pass
            print(f"✅ [后台守护] 会话 {thread_id} 的总结已自动压缩并归档入库！")
        except Exception as e:
            print(f"⚠️ [后台守护异常] 总结失败: {e}")

    def reset_timer(self, thread_id: str, uid: str):
        if thread_id in self.timers:
            self.timers[thread_id].cancel()
        t = threading.Timer(self.timeout_seconds, self._trigger_summary, args=(thread_id, uid))
        t.daemon = True
        t.start()
        self.timers[thread_id] = t


session_manager = SessionTimerManager(timeout_seconds=30)

# =====================================================================
# 2. 接口定义与模型
# =====================================================================
app = FastAPI(title="飞羽科技 HR Agent API", version="2.0.0")


class ChatRequest(BaseModel):
    uid: str = Field(...)
    thread_id: str = Field(...)
    message: str = Field(...)


class ResumeRequest(BaseModel):
    thread_id: str = Field(...)
    action: str = Field(..., description="'approve' 或 'reject'")


class ChatResponse(BaseModel):
    code: int = Field(default=200)
    thread_id: str = Field(...)
    status: str = Field(default="completed")
    answer: str = Field(default="")
    interrupt_msg: str = Field(default="")


# =====================================================================
# 3. 核心路由 Endpoint
# =====================================================================
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    state = {"messages": [HumanMessage(content=request.message)], "current_uid": request.uid, "loop_step": 0}
    final_answer = ""

    try:
        for event in hr_agent_app.stream(state, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            if isinstance(last_msg, HumanMessage): continue
            if last_msg.type == "ai" and not last_msg.tool_calls:
                final_answer = last_msg.content

        state_snapshot = hr_agent_app.get_state(config)
        if state_snapshot.next:
            interrupt_message = state_snapshot.tasks[0].interrupts[0].value
            # 如果被挂起，不需要开启定时总结器
            return ChatResponse(code=202, thread_id=request.thread_id, status="waiting_for_review",
                                interrupt_msg=interrupt_message)

        # 正常结束，重置超时计时器
        session_manager.reset_timer(thread_id=request.thread_id, uid=request.uid)

        return ChatResponse(code=200, thread_id=request.thread_id, status="completed", answer=final_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/resume", response_model=ChatResponse)
async def resume_endpoint(request: ResumeRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    state_snapshot = hr_agent_app.get_state(config)
    if not state_snapshot.next:
        raise HTTPException(status_code=400, detail="当前会话未处于等待审批状态。")

    final_answer = ""
    try:
        for event in hr_agent_app.stream(Command(resume=request.action), config, stream_mode="values"):
            last_msg = event["messages"][-1]
            if isinstance(last_msg, HumanMessage): continue
            if last_msg.type == "ai" and not last_msg.tool_calls:
                final_answer = last_msg.content

        return ChatResponse(code=200, thread_id=request.thread_id, status="completed", answer=final_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)