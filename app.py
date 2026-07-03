import streamlit as st
import sys
import uuid
from pathlib import Path
from langchain_core.messages import HumanMessage
from langgraph.types import Command

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
from agent.graph_builder import hr_agent_app

st.set_page_config(page_title="飞羽科技 HR 智能助理", page_icon="🤖", layout="centered")
st.title("👨‍💼 飞羽科技 HR 智能助理")
st.caption("基于 LangGraph Actor-Critic 架构与 Human-in-the-loop 人机协同")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是 HR 智能助理。请问您需要咨询政策、请假还是开具证明？"}]
if "current_uid" not in st.session_state:
    st.session_state.current_uid = "1001"

with st.sidebar:
    st.header("⚙️ 模拟登录环境")
    st.session_state.current_uid = st.selectbox("选择当前登录员工:", ["1001", "1002", "1003", "1004"], index=0)
    st.divider()
    if st.button("🧹 清空会话记忆"):
        st.session_state.messages = [{"role": "assistant", "content": "记忆已清空，我们重新开始吧！"}]
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

config = {"configurable": {"thread_id": st.session_state.thread_id}}
state_snapshot = hr_agent_app.get_state(config)

if state_snapshot.next:
    interrupt_msg = state_snapshot.tasks[0].interrupts[0].value
    with st.chat_message("assistant"):
        st.warning(f"🛡️ **安全拦截 (Human-in-the-loop)**\n\n{interrupt_msg}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 授权执行 (Approve)", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "*(用户点击了：✅ 授权执行)*"})
                with st.spinner("审批通过，正在生成证明..."):
                    final_answer = ""
                    for event in hr_agent_app.stream(Command(resume="approve"), config, stream_mode="values"):
                        last_msg = event["messages"][-1]
                        if last_msg.type == "ai" and not last_msg.tool_calls:
                            final_answer = last_msg.content
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.rerun()

        with col2:
            if st.button("❌ 驳回请求 (Reject)", type="primary", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "*(用户点击了：❌ 驳回请求)*"})
                with st.spinner("正在取消操作..."):
                    final_answer = ""
                    for event in hr_agent_app.stream(Command(resume="reject"), config, stream_mode="values"):
                        last_msg = event["messages"][-1]
                        if last_msg.type == "ai" and not last_msg.tool_calls:
                            final_answer = last_msg.content
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.rerun()

else:
    if prompt := st.chat_input("输入您的问题，例如：帮我开一份收入证明"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        state = {"messages": [HumanMessage(content=prompt)], "current_uid": st.session_state.current_uid,
                 "loop_step": 0}

        with st.chat_message("assistant"):
            status_container = st.empty()
            response_container = st.empty()
            full_response = ""

            with st.spinner("思考中，正在调度 Agent..."):
                for event in hr_agent_app.stream(state, config, stream_mode="values"):
                    last_msg = event["messages"][-1]
                    if isinstance(last_msg, HumanMessage): continue
                    if last_msg.type == "ai" and last_msg.tool_calls:
                        for tool in last_msg.tool_calls:
                            status_container.info(f"⚙️ 正在调用工具: `{tool['name']}` ...")
                    elif last_msg.type == "ai" and not last_msg.tool_calls:
                        status_container.empty()
                        full_response = last_msg.content
                        response_container.markdown(full_response)

            new_snapshot = hr_agent_app.get_state(config)
            if new_snapshot.next:
                st.rerun()
            else:
                st.session_state.messages.append({"role": "assistant", "content": full_response})