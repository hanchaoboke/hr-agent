# pip install fastapi unicorn pydantic
import sys
from pathlib import Path
from typing import final

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
import uvicorn

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from agent.graph_builder import hr_agent_app

# 1. 定义数据校验模型（pydantic）
# 请具体：定义客户端调用API时提供的参数
class ChatRequest(BaseModel):
    uid:str = Field(..., description='调用此接口的员工UID，例如 ‘1001’')
    thread_id:str = Field(...,description='会话ID，用于保持多轮对话的记忆上下文')
    message:str = Field(...,description='用户的提问内容')

# 响应体模型：定义API返回给客户端的数据格式
class ChatResponse(BaseModel):
    code:int = Field(default=200, description='状态码')
    thread_id: str = Field(description='当前会话ID')
    answer:str = Field(description='Agent 给出的最终文字回答')

# 2. 初始化 FastAPI 应用
app = FastAPI(
    title='飞羽科技 HR 智能助理',                         # API 文档的标题
    version='1.0.0',                                   # API 版本号
    description='基于 LangGraph 的多智能体 HR 服务接口',   # API描述
)

# 3. 定义路由核心
# 这里是API主要业务逻辑入口
@app.post('/api/v1/chat', response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    接受用户提问，调用 LangGraph， 并返回最终结果。

    工作流程：
    1. 接收用户的提问和会话信息
    2. 调用 LangGraph 进行推理
    3. 提取最终文本回答返回给客户端
    """
    # 配置会话记忆ID
    config = {'configurable':{'thread_id': request.thread_id}}

    # 初始化状态
    state = {
        "message": [HumanMessage(content=request.message)],
        "current_uid": request.uid,
        "loop_step":0
    }

    final_answer = ''       # 用于存储最终的回答

    try:
        for event in hr_agent_app.stream(state, config, stream_mode='values'):
            last_message = event['messages'][-1]

            if isinstance(last_message, HumanMessage):
                continue

            if last_message.type == 'ai' and not last_message.tool_calls:
                final_answer = last_message.content

        # 检查是否成功获取了最终答案
        if not final_answer:
            raise ValueError('Agent 运行结束， 未能生成有效的文字回复。')

        return ChatResponse(
            code=200,
            thread_id=request.thread_id,
            answer=final_answer
        )
    except Exception as e:
        print(f'[API exception] {e}')
        raise HTTPException(status_code=500, detail=f'内部推理出错：{str(e)}')


# 预留一个健康接口，用于监控系统运行状态的，用于 Devops和容器编排工具使用的
@app.get('/health')
def health_check():
    return {'status': 'ok', "services":"hr_agent_api"}


# 4. 启动脚本
if __name__ == '__main__':
    print('正在启动 HR Agent API 服务...')
    uvicorn.run('api:app', host='0.0.0.0', port=8000, reload=True)
    # uvicorn.run()     # 启动服务器
    # 'api:app'         # api 是当前文件名，app 是 Fast API实例
    # host='0.0.0.0'    # 监听所有网络接口，允许外部访问
    # port=8000         # 监听 8000 端口
    # reload=True       # 开发模式下代码变更自动重启（生成环境应设置为False）