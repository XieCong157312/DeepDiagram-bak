from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from app.state.state import AgentState
from app.core.config import settings
from app.core.llm import get_llm, get_configured_llm
import re

async def router_node(state: AgentState):
    """
    Analyzes the user's input and determines the appropriate agent.
    Supports explicit routing via @agent syntax.
    """
    messages = state['messages']
    last_message = messages[-1]
    
    # 0. Check for explicit @agent routing
    explicit_intent = None
    if isinstance(last_message, HumanMessage) and isinstance(last_message.content, str):
        content = last_message.content.lower().strip()
        
        # Explicit mapping
        mappings = {
            "@mindmap": "mindmap",
            "@flow": "flowchart",
            "@flowchart": "flowchart",
            "@mermaid": "mermaid",
            "@chart": "charts",
            "@charts": "charts",
            "@drawio": "drawio",
            "@infographic": "infographic"
        }
        
        for keyword, intent_name in mappings.items():
            if keyword in content:
                explicit_intent = intent_name
                # Clean the keyword from the message so the agent doesn't see it
                # We update the original message object content for the downstream agent
                cleaned = re.sub(rf"{keyword}\s*", "", last_message.content, flags=re.IGNORECASE).strip()
                if not cleaned:
                    cleaned = f"Generate a default {intent_name} diagram."
                last_message.content = cleaned
                print(f"DEBUG ROUTER | Explicit Routing Triggered: {keyword} -> {intent_name} | Cleaned: {last_message.content}")
                break

    agent_descriptions = {
        "mindmap": "最适合层次结构、头脑风暴、概述想法和组织概念。输出：Markdown/Markmap。",
        "flow": "仅最适合标准流程图。输出：React Flow JSON。",
        "mermaid": "最适合序列图、类图、状态图、甘特图、Git 图、实体关系图 (ERD) 和用户旅程。如果用户明确要求'Mermaid'，使用此项。输出：Mermaid 语法。",
        "charts": "最适合定量数据可视化（销售、统计、趋势）。输出：ECharts（条形图、线图、饼图等）。",
        "drawio": "最适合专业、重型架构图、云基础设施和详细 UML。仅当用户明确要求'Draw.io'或复杂'架构'时使用。",
        "infographic": "最适合信息图、数据海报、视觉叙事、流程可视化、比较图表、时间线和创意数据展示。如果用户要求'信息图'、'数据海报'、'视觉摘要'或'信息图'，使用此项。",
        "general": "处理问候、与绘图无关的问题，或不适合其他类别的请求。"
    }
    
    # Identify Full Agent Execution History
    execution_history = []
    for msg in messages[:-1]:
        if msg.type == "ai" and "### Execution Trace:" in str(msg.content):
            # Extract the whole trace block for the router
            parts = str(msg.content).split("### Execution Trace:")
            if len(parts) > 1:
                trace = parts[1].strip()
                # Safety truncate each trace block
                if len(trace) > 1000:
                    trace = trace[:1000] + "... [TRUNCATED]"
                execution_history.append(trace)
    
    execution_history_text = "\n---\n".join(execution_history) if execution_history else "None"
    
    # Identify Last Active Agent
    last_active_agent = "None"
    for msg in reversed(messages[:-1]):
        if msg.type == "ai" and "agentName:" in str(msg.content):
            match = re.search(r"agentName:\s*(\w+)", str(msg.content))
            if match:
                last_active_agent = match.group(1)
                break
    
    # If we have an explicit intent, we can skip the LLM call but we STILL want to return 
    # the intent inside the unified flow for consistency.
    if explicit_intent:
        print(f"DEBUG ROUTER | Proceeding with Explicit Intent: {explicit_intent}")
        return {"intent": explicit_intent}

    descriptions_text = "\n".join([f"- '{key}': {desc}" for key, desc in agent_descriptions.items()])

    system_prompt = f"""你是一个智能的 DeepDiagram 路由器。
    你的目标是分析用户的意图并路由到最合适的图表代理。
    
    代理执行历史（代理 + 工具）： 
    {execution_history_text}
    
    上次活跃代理：{last_active_agent}
    
    （如果用户的请求是后续、完善或"修复"之前的結果，除非他们明确要求不同的工具或主题根本改变，否则倾向于 {last_active_agent}）

    上下文感知规则：
    1. 如果"当前视觉上下文"是"图表"且用户要求"添加"、"删除"、"更改"、"更新"数字或项目 -> 你必须路由到 'charts'。
    2. 如果"当前视觉上下文"是"思维导图"且用户要求"添加节点"、"扩展" -> 你必须路由到 'mindmap'。
    3. 如果"当前视觉上下文"是"流程图"且用户要求"更改形状"、"连接" -> 你必须路由到 'flow'。
    4. 如果"当前视觉上下文"是"Mermaid 图表"且用户要求"添加参与者"、"更改流程" -> 你必须路由到 'mermaid'。
    5. 如果"当前视觉上下文"是"Draw.io 架构"且用户要求"添加云组件"、"更改布局" -> 你必须路由到 'drawio'。
    6. 如果用户提到"Mermaid"或要求"序列图"、"类图"、"甘特图" -> 你必须路由到 'mermaid'。
    
    代理能力：
    {descriptions_text}
    
    只输出关键词：'mindmap', 'flow', 'mermaid', 'charts', 'drawio', 'general'。
    """
    
    # Helper to safely summarize PREVIOUS message content for history (concise text only)
    def summarize_history_content(content):
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        text_parts.append("[User uploaded an image]")
            text = " ".join(text_parts)
        else:
            text = str(content)
        
        # Safety truncation for routing efficiency
        if len(text) > 1000:
            return text[:1000] + "... [TRUNCATED]"
        return text

    # Summarize history except for the very last message
    conversation_text = ""
    for msg in messages[:-1]:
        role = "User" if msg.type == "human" else "Assistant"
        content_summary = summarize_history_content(msg.content)
        conversation_text += f"{role}: {content_summary}\n"
    
    # Final Routing Prompt
    routing_instructions = f"""{system_prompt}
    
    CONVERSATION HISTORY (Summarized):
    {conversation_text}
    
    Please analyze the user's latest message (which may include an image) and classify the intent.
    Respond in the same language as the user's input (e.g., if the user asks in Chinese, respond in Chinese).
    """
    
    # Add time context
    from app.core.llm import get_time_instructions
    routing_instructions += get_time_instructions()
    
    # We pass the instruction as a SystemMessage and the ACTUAL last message as is.
    # This ensures that if the last message has image_url, the LLM will see it as an image, NOT as long text tokens.
    msgs_to_invoke = [
        SystemMessage(content=routing_instructions),
        messages[-1] # The real last message with multimodal content
    ]
    
    llm = get_configured_llm(state)
    response = await llm.ainvoke(msgs_to_invoke)
    intent = response.content.strip().lower()
    
    print(f"DEBUG ROUTER | Last Agent: {last_active_agent} | Raw Intent: {intent}")

    if "mindmap" in intent:
        return {"intent": "mindmap"}
    elif "flow" in intent:
        return {"intent": "flowchart"}
    elif "mermaid" in intent:
        return {"intent": "mermaid"}
    elif "chart" in intent:
        return {"intent": "charts"}
    elif "drawio" in intent or "draw.io" in intent or "architecture" in intent or "network" in intent:
        return {"intent": "drawio"} 
    elif "infographic" in intent or "信息图" in intent or "poster" in intent:
        return {"intent": "infographic"}
    elif "general" in intent:
        return {"intent": "general"}
    else:
        return {"intent": "general"} # Default to general for safety

def route_decision(state: AgentState) -> Literal["mindmap_agent", "flow_agent", "mermaid_agent", "charts_agent", "drawio_agent", "infographic_agent", "general_agent"]:
    intent = state.get("intent")
    if intent == "mindmap":
        return "mindmap_agent"
    elif intent == "flowchart":
        return "flow_agent"
    elif intent == "mermaid":
        return "mermaid_agent"
    elif intent == "charts":
        return "charts_agent"
    elif intent == "drawio":
        return "drawio_agent"
    elif intent == "infographic":
        return "infographic_agent"
    elif intent == "general":
        return "general_agent"
    return "general_agent"
