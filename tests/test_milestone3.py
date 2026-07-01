import sys
import io
from pathlib import Path
from langchain_core.messages import HumanMessage

from PIL import Image as PIL_Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from agent.graph_builder import hr_agent_app

def display_graph(graph):
    try:
        print('咱再说生成 LangGraph 架构图')
        png_data = graph.get_graph(xray=1).draw_mermaid_png()
        images_stream = io.BytesIO(png_data)
        img = PIL_Image.open(images_stream)
        img.show()
        print('架构图成功显示')
    except Exception as e:
        print(f'弹窗失败:{e}')

def chat_with_agent(uid:str, question:str):
    """与 HR Agent 进行流式互动测试"""
    print('='* 60)
    print(f'【员工UID:{uid}】 提问：{question}')
    print('=' * 60)

    # 初始化状态
    initial_state = {
        'messages': [HumanMessage(content=question)],
        "current_uid": uid,
        "loop_step": 0
    }

    # 使用流式输出
    for event in hr_agent_app.stream(initial_state, stream_mode='values'):
        last_msg = event['messages'][-1]

        # 过滤掉系统的初始输入和反思审计的打回提示，让展示日志更干净
        if isinstance(last_msg, HumanMessage) and "[SYSTEM AUDIT FALLED]" not in last_msg.content:
            continue

        if last_msg.type == 'ai' and not last_msg.tool_calls:
            print(f'\n【AI 最终回复】：\n{last_msg.content}\n')
        elif last_msg.type == 'ai' and last_msg.tool_calls:
            for tool in last_msg.tool_calls:
                print(f'【调度工具】 -》 {tool['name']}({tool['args']})')


if __name__ == '__main__':
    # display_graph(hr_agent_app)

    # 场景 A：简单的数据库操作
    chat_with_agent(uid="1002", question="帮我查⼀下我还有⼏天年假？如果可以的话顺便帮我开个在职证明。")
    # 场景 B：触发 RAG + 上下⽂注⼊感知 (P4员⼯，⼆线城市成都)
    chat_with_agent(uid="1002", question="我下周要去北京出差，住宿费最⾼报销多少？")
    # 场景 C：触发⾼难度 RAG 分析
    chat_with_agent(uid="1001", question="我刚⼊职两年，如果休3天事假，需要谁来审批？")