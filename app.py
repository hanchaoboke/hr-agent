import streamlit as st
import sys
import uuid
from pathlib import Path
from langchain_core.messages import HumanMessage

# =====================================================================
# 1. 动态挂载后端引擎
# =====================================================================
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# 引入我们写好的 LangGraph 核心
from agent.graph_builder import hr_agent_app

# =====================================================================
# 2. 页面配置与 Session 初始化
# =====================================================================
st.set_page_config(page_title="飞羽科技 HR 智能助理", page_icon="🤖", layout="centered")
st.title(" HR 智能助理")
st.caption("基于 LangGraph Actor-Critic 架构与高精度 RAG 驱动")

# 初始化 Streamlit 会话状态 (保证刷新页面前，记忆不丢失)
if "thread_id" not in st.session_state:
    # 每次打开网页，生成一个独一无二的会话 ID，交给 LangGraph Memory
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # 用于在前端 UI 渲染聊天记录
    st.session_state.messages = [{"role": "assistant",
                                  "content": "您好！我是 HR 智能助理。请问您需要咨询政策、请假还是开具证明？（为了给您精准答复，请先告诉我您的 UID，如 1001）"}]

if "current_uid" not in st.session_state:
    st.session_state.current_uid = "未知"

# =====================================================================
# 3. 侧边栏：员工身份模拟器
# =====================================================================
with st.sidebar:
    st.header("模拟登录环境")
    st.info("提示：真实系统中这部分由企业 SSO 单点登录自动传入。")
    # 下拉菜单让测试者快速切换身份
    selected_uid = st.selectbox(
        "选择当前登录员工:",
        ["1001", "1002", "1003", "1004"],
        index=0
    )
    st.session_state.current_uid = selected_uid

    st.divider()
    st.markdown("""
    **测试用例建议：**
    * 查假期：*我还有几天年假？*
    * 查报销：*我去北京出差，住宿费给报多少？*
    * 开证明：*帮我开一份收入证明。*
    * 多轮记忆：*直接用代词提问，测试上下文关联。*
    """)

    if st.button("🧹 清空会话记忆"):
        st.session_state.messages = [{"role": "assistant", "content": "记忆已清空，我们重新开始吧！"}]
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# =====================================================================
# 4. 主聊天界面渲染
# =====================================================================
# 遍历并显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================================================
# 5. 聊天输入与大模型调度
# =====================================================================
if prompt := st.chat_input("输入您的问题，例如：帮我开一份在职证明"):

    # 1. 把用户的话显示在界面上
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 准备发给后端的 State 和 Config
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state = {
        "messages": [HumanMessage(content=prompt)],
        "current_uid": st.session_state.current_uid,
        "loop_step": 0
    }

    # 3. 显示 AI 思考过程并流式渲染结果
    with st.chat_message("assistant"):
        status_container = st.empty()
        response_container = st.empty()

        full_response = ""

        # 捕获 LangGraph 的流式输出
        with st.spinner("思考中，正在调度 Agent..."):
            for event in hr_agent_app.stream(state, config, stream_mode="values"):
                last_msg = event["messages"][-1]

                # 过滤系统反馈消息
                if isinstance(last_msg, HumanMessage):
                    continue

                # 播报工具调用过程 (在界面上快速闪过，增加高级感)
                if last_msg.type == "ai" and last_msg.tool_calls:
                    for tool in last_msg.tool_calls:
                        status_container.info(f"⚙️ 正在调用工具: `{tool['name']}` ...")

                # 获取最终的文字回答
                elif last_msg.type == "ai" and not last_msg.tool_calls:
                    status_container.empty()  # 清理工具调用提示
                    full_response = last_msg.content
                    response_container.markdown(full_response)

        # 4. 把 AI 的最终回答存入前端 state
        st.session_state.messages.append({"role": "assistant", "content": full_response})