from langchain_core.messages import SystemMessage
from app.state.state import AgentState
from app.core.llm import get_llm, get_configured_llm

async def general_agent_node(state: AgentState):
    messages = state['messages']
    
    system_prompt = SystemMessage(content="""你是一个名为 DeepDiagram 的专业 AI 助手，专门用于创建图表。

你的能力：
1. 思维导图（使用 Markmap/Markdown）
2. 流程图（使用 Mermaid）
3. 图表（使用 ECharts）

如果用户的请求是简单的对话（问候、询问你能做什么），请自然简洁地回应。
鼓励他们创建可视化内容。

语言：用与用户输入相同的语言回应。

不要调用任何工具。只聊天。
""")
    
    llm = get_configured_llm(state)
    
    # Add time context to system prompt
    from app.core.llm import get_time_instructions
    system_prompt.content += get_time_instructions()
    
    response = await llm.ainvoke([system_prompt] + messages)
    return {"messages": [response]}
