import sys
import threading
from pathlib import Path
from langchain_core.messages import HumanMessage

# 动态挂载项目根目录
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from agent.graph_builder import hr_agent_app


class SessionManager:
    """会话生命周期管理器：负责超时控制与事件分发"""

    def __init__(self, timeout_seconds=5):
        self.timeout_seconds = timeout_seconds
        self.timer = None
        self.current_thread_id = None
        self.current_uid = None

    def trigger_summary(self):
        """倒计时结束触发的方法：向 Graph 发送总结指令"""
        if not self.current_thread_id:
            return

        # config 中必须传入 thread_id，LangGraph 才知道要提取哪一段记忆
        config = {"configurable": {"thread_id": self.current_thread_id}}

        # 构建隐藏的触发指令
        idle_trigger_state = {
            "messages": [HumanMessage(content="__SYS_IDLE_TIMEOUT__")],
            "current_uid": self.current_uid
        }

        print(f"\n[后台守护线程] 监控到用户 {self.current_uid} 闲置超过 {self.timeout_seconds} 秒，触发自动总结。")

        # 调用大模型执行总结
        for event in hr_agent_app.stream(idle_trigger_state, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            if last_msg.type == "ai" and not last_msg.tool_calls:
                print(f"{last_msg.content}\n")

        print("(请继续提问，或按 Ctrl+C 退出)")

    def reset_timer(self):
        """重置倒计时"""
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout_seconds, self.trigger_summary)
        self.timer.start()

    def chat(self, uid: str, thread_id: str, question: str):
        """对外暴露的聊天接口"""
        self.current_uid = uid
        self.current_thread_id = thread_id

        # 用户说话了，先停止计时器
        if self.timer:
            self.timer.cancel()

        print(f"\n[UID:{uid}] 提问: {question}")

        # 配置 memory 必须要的 thread_id
        config = {"configurable": {"thread_id": thread_id}}

        state = {
            "messages": [HumanMessage(content=question)],
            "current_uid": uid,
            "loop_step": 0
        }

        for event in hr_agent_app.stream(state, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            if isinstance(last_msg, HumanMessage):
                continue
            if last_msg.type == "ai" and not last_msg.tool_calls:
                print(f"[AI答复]: {last_msg.content}")

        # 聊天结束，重新开启倒计时
        self.reset_timer()


if __name__ == "__main__":
    print("=== 里程碑 4: 多轮记忆与异步超时总结测试 ===")

    # 设定超时时间为 6 秒（方便测试时能快速看到效果）
    session = SessionManager(timeout_seconds=6)

    # 模拟真实用户的连续两轮提问
    # 使用同一个 thread_id: "session_1001_a" 代表同一个对话窗口

    # 第一轮：只表明身份，不问具体问题
    session.chat(uid="1001", thread_id="session_1001_a", question="你好，我是张三。")

    # 模拟用户思考 3 秒钟，还在超时范围内
    import time

    time.sleep(3)

    # 第二轮：直接使用代词“我”，测试大模型的“多轮记忆”能力！
    # 因为有 memory，Agent 知道“我”就是刚才说的“张三（1001）”
    session.chat(uid="1001", thread_id="session_1001_a", question="我还有多少天年假？")

    print("\n聊天结束，用户离开电脑。开始测试 6 秒闲置自动总结，请不要操作，静静等待...")

    # 主线程维持运行，等待子线程(Timer)触发
    try:
        # 等待 8 秒，确保能看到总结输出
        time.sleep(8)
    except KeyboardInterrupt:
        pass
    finally:
        # 清理定时器
        if session.timer:
            session.timer.cancel()
        print("=== 测试结束 ===")