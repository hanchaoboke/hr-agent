from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

from langgraph.types import interrupt
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.output_parsers import JsonOutputParser
from langgraph.checkpoint.memory import InMemorySaver

from tools.hr_tools import get_employee_profile, check_leave_balance, generate_employment_certificate
from agent.rag_pipeline2 import search_hr_policy

load_dotenv()

# 1. 定义全局共享状态 State
class AgentState(TypedDict):
    messages:Annotated[list[BaseMessage], add_messages] # add_messages 确保消息是追加而非覆盖
    current_uid:str
    loop_step:int

# 2. 初始化大模型与工具绑定
llm =  ChatOpenAI(
    model=os.getenv('DEEPSEEK_MODEL'),
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.0
)

tools = [get_employee_profile, check_leave_balance, generate_employment_certificate, search_hr_policy]
llm_with_tools = llm.bind_tools(tools)
tools_node = ToolNode(tools)

# 3. 定义执行节点 Node
def chatbot_node(state: AgentState):
    """[执行者节点]意图理解、工具调用与内容生成"""
    messages = state.get('messages', [])

    last_message = messages[-1]
    # 拦截应用层发来的“超时总结”隐藏指令
    if isinstance(last_message, HumanMessage) and last_message.content == "__SYS_IDLE_TIMEOUT__":
        print("\n[触发超时] 正在压缩会话历史，生成自动总结...")
        # 让大模型对历史消息进行总结（排除掉这条隐藏指令本身）
        summary_llm = llm.model_copy(update={'temperature': 0.3})
        summary_prompt = (
            "你是一个HR助理。请用简短的一两句话，总结上面对话中员工咨询的核心问题以及你给出的最终结论。\n"
            "直接输出总结结果，并以【会话闲置总结】这几个字开头。"
        )
        # 将提示词作为临时系统消息追加进去让其总结
        response = summary_llm.invoke(messages[:-1] + [SystemMessage(content=summary_prompt)])
        return {"messages": [response]}  # 返回总结信息

    # 首轮对话注入 System prompt
    if len(messages) == 1:
        system_msg = SystemMessage(
            content="你是公司的高级 HR 智能助理。\n"
                    f"当前员工 UID 为 {state['current_uid']}。\n"
                    f"请务必先调用 get_employee_profile 获取该员工的属性，再回答政策问题。\n"
                    f"必须基于工具返回的事实，绝对不能编造数字或条件。"
        )
        messages = [system_msg] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response], 'loop_step':state.get('loop_step', 0) + 1}

# 人工介入节点
def human_review_node(state: AgentState):
    """【人工介入节点】拦截敏感操作，将图挂起等待授权"""
    last_message = state["messages"][-1]

    # 检查大模型是否试图调用敏感工具
    sensitive_tool_call = None
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "generate_employment_certificate":
                sensitive_tool_call = tool_call
                break

    if sensitive_tool_call:
        print(f"\n[系统挂起] 检测到敏感操作：准备生成证明文件。")
        # 使用 interrupt 挂起当前状态机，并将提示信息发送给外部（前端）
        # 此时后端执行会完全暂停，直到外部传入 resume 数据
        user_decision = interrupt("Agent 正在尝试生成包含薪资的证明文件。是否授权执行？(输入 'approve' 或 'reject')")

        if user_decision == "reject":
            print("❌ [人工拒绝] 授权被驳回。")
            # 伪造一个工具调用失败的消息，告知大模型操作被取消
            reject_msg = ToolMessage(
                content="[SYSTEM] 人工审批未通过，操作已被拒绝。请安抚用户并告知由于安全原因无法生成。",
                name=sensitive_tool_call["name"],
                tool_call_id=sensitive_tool_call["id"]
            )
            return {"messages": [reject_msg]}

        print("✅ [人工授权] 审批通过，允许放行。")

    return {"messages": []}

# 审计者节点
class FactCheckResult(BaseModel):
    is_pass: bool = Field(description='如果AI的回答完全忠于知识库原文输出True，捏造了数字或政策则输出False。')
    feedback:str = Field(description='如果False，指出造假点；如果True，输出“PASS”')

def fact_checker_node(state: AgentState):
    """[审计者节点]后置事实检验(self-Reflection)"""
    messages = state.get('messages')
    last_message = messages[-1]

    # 逆向查找 RAG 召回的原文
    rag_context = ""
    for msg in reversed(messages):
        if getattr(msg, "name", "") == "search_hr_policy":
            rag_context = msg.content
            break

    # 若未调用知识库，直接放行
    if not rag_context:
        return {'messages':[]}

    print('\n【审计者介入】正在核查生成内容是否包含幻觉。。。。')

    checker_llm = ChatOpenAI(
        model=os.getenv('DEEPSEEK_MODEL'),
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_BASE_URL'),
        temperature=0.0
    )

    # 示例化通用的 JSON 解析器
    parser = JsonOutputParser(pydantic_object=FactCheckResult)

    check_prompt = (
        "你是一名冷酷的合规审计员。对比以下【知识库原文】和【AI生成回复】。\n"
        f"【知识库原文】:\n{rag_context}\n\n"
        f"【AI生成回复】:\n{last_message.content}\n\n"
        f"严查金额、职级门槛和天数！发现捏造请判 False 并给出修改意见。\n\n"
        f"{parser.get_format_instructions()}"                               # 吧格式要求也拼到 prompt 中
    )

    response = checker_llm.invoke(check_prompt)

    # 手动解析 JSON 并增加容错
    try:
        result = parser.invoke(response)
        is_pass = result.get('is_pass', True)
        feedback = result.get('feedback', "PASS")
    except Exception as e:
        print(f'[审计异常] JSON 解析失败，默认放行。原因：{e}')
        is_pass = True
        feedback = "PASS"

    if is_pass:
        print('[审计通过]回答安全，无幻觉')
        return {"messages":[]}
    else:
        print(f'[发现幻觉] 拦截成功！审计意见：{feedback}')
        correction_msg = HumanMessage(
            content=f'[SYSTEM AUDIT FALLED]事实错误反馈：{feedback}。请立即根据数据库原文重写，绝不可包含虚假数据。'
        )
        return {"messages":[correction_msg]}

# 定义路由逻辑
def router_after_chatbot(state: AgentState):
    """Chatbot 输出之后的路由判断"""
    last_message = state.get('messages')[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 有工具调用时，先去人工审批节点过安检
        return "human_review"
    return "fact_checker"

def router_after_review(state: AgentState) -> str:
    """审批节点后的流向判断"""
    last_message = state["messages"][-1]
    # 如果最后一条消息是 ToolMessage（说明被拒绝了，我们伪造了报错），打回给 chatbot 解释原因
    if isinstance(last_message, ToolMessage):
        return "chatbot"
    # 如果同意了，或者没有敏感工具，正常放行去执行工具
    return "tools"

def router_after_fact_check(state: AgentState):
    """审计完成后的路由判断"""
    last_message = state.get('messages')[-1]

    if isinstance(last_message, HumanMessage):
        if state.get('loop_step', 0) > 4:
            print('【强制熔断】反思次数达到上限，放弃纠错。')
            return 'end'
        print('【打回重写】路由指针流回到 chatbot 节点。。。。。。')
        return 'chatbot'
    else:
        return 'end'

# 5. 构建状态图
workflow = StateGraph(AgentState)

workflow.add_node('chatbot', chatbot_node)
workflow.add_node("human_review", human_review_node) # 人类审批节点
workflow.add_node('tools', tools_node)
workflow.add_node('fact_checker', fact_checker_node)

workflow.add_edge(START, 'chatbot')
workflow.add_conditional_edges('chatbot',
                               router_after_chatbot,
                               {
                                   'human_review':'human_review',
                                   'fact_checker':'fact_checker',
                               })

# 审批节点出来的条件路由
workflow.add_conditional_edges("human_review", router_after_review, {"chatbot": "chatbot", "tools": "tools"})

# 工具节点执行完成以后返回主节点
workflow.add_edge('tools', 'chatbot')
workflow.add_conditional_edges('fact_checker',
                               router_after_fact_check,
                               {
                                   'chatbot':'chatbot',
                                   'end':END
                               })

memory = InMemorySaver()

hr_agent_app = workflow.compile(checkpointer=memory)