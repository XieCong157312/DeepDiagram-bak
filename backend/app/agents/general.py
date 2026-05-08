from langchain_core.messages import SystemMessage
from app.state.state import AgentState
from app.core.llm import get_llm, get_configured_llm

async def general_agent_node(state: AgentState):
    messages = state['messages']
    
    system_prompt = SystemMessage(content="""You are DeepDiagram, a helpful AI assistant specialized in creating diagrams.
    
    Your capabilities:
    1. Mindmaps (using Markmap/Markdown)
    2. Flowcharts (using Mermaid)
    3. Charts (using ECharts)
    
    If the user's request is simple conversation (greeting, asking what you can do), respond naturally and briefly.
    Encourage them to create a visual.
    
    LANGUAGE: Respond in the same language as the user's input. 如果用户用中文提问，请用中文回答；如果使用英语提问，请使用英语回答。
    
    DO NOT call any tools. Just chat.
    """)
    
    llm = get_configured_llm(state)
    
    # Add time context to system prompt
    from app.core.llm import get_time_instructions
    system_prompt.content += get_time_instructions()
    
    response = await llm.ainvoke([system_prompt] + messages)
    return {"messages": [response]}
