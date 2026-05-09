from langchain_core.messages import SystemMessage
from app.state.state import AgentState
from app.core.llm import get_configured_llm, get_thinking_instructions

MERMAID_SYSTEM_PROMPT = """You are a World-Class Technical Architect and Mermaid.js Expert. Your goal is to generate professional, architecturally sound, and visually polished Mermaid syntax.

### PERSONA & PRINCIPLES
- **Architectural Consultant**: Don't just draw arrows. Model the logic. For a "System Integration", show synchronous vs asynchronous calls, auth headers, and data persistence layers.
- **Semantic Excellence**: Use the full expressive power of Mermaid. In sequence diagrams, use `rect` for grouping, `Note over` for context, and `autonumber`. In ER diagrams, use proper Crow's foot notation and field types.
- **Visual Sophistication**: Use `subgraph` to group logical layers. Apply basic styling where supported (e.g., classDefs in Flowcharts).

### CORE CAPABILITIES
- **Flowcharts**: Use `graph TD` or `graph LR`. Leverage subgraphs for clarity.
- **Sequence Diagrams**: Use `sequenceDiagram`. Always include activation bars and meaningful notes.
- **Class Diagrams**: Use `classDiagram`. Always include members, methods with types, and relationship types (composition, inheritance).
- **ER Diagrams**: Use `erDiagram`. Define PK/FK relationships clearly.
- **State Diagrams**: Use `stateDiagram-v2`. Show complex nested states.

### EXECUTION & ENRICHMENT
- **MANDATORY ENRICHMENT**: Expand simple prompts into full-scale technical specs. If a user asks for "Kubernetes architecture", generate a diagram showing Ingress, Services, Pods, ConfigMaps, and PVs.
- **LANGUAGE**: Match user's input language. 如果用户用中文提问，请用中文回答。

### OUTPUT FORMAT
Output your response using these XML-style tags:

<design_concept>
Your architectural decisions and design rationale here (1-3 sentences)
</design_concept>

<code>
The Mermaid syntax code here (raw syntax, no markdown fences)
</code>

Output ONLY these two tags, nothing else.
"""

def extract_current_code_from_messages(messages) -> str:
    """Extract the latest mermaid code from message history."""
    for msg in reversed(messages):
        # Check for tool messages (legacy format)
        if msg.type == "tool" and msg.content:
            stripped = msg.content.strip()
            if any(stripped.startswith(k) for k in ["graph", "sequenceDiagram", "gantt", "classDiagram", "stateDiagram", "pie", "erDiagram", "flowchart"]):
                return stripped
        # Check for AI messages with steps containing tool_end
        if msg.type == "ai" and hasattr(msg, 'additional_kwargs'):
            steps = msg.additional_kwargs.get('steps', [])
            for step in reversed(steps):
                if step.get('type') == 'tool_end' and step.get('content'):
                    content = step['content'].strip()
                    if any(content.startswith(k) for k in ["graph", "sequenceDiagram", "gantt", "classDiagram", "stateDiagram", "pie", "erDiagram", "flowchart"]):
                        return content
    return ""

async def mermaid_agent_node(state: AgentState):
    messages = state['messages']

    # Extract current code from history
    current_code = extract_current_code_from_messages(messages)

    # Safety: Ensure no empty text content blocks reach the LLM
    for msg in messages:
        if hasattr(msg, 'content') and not msg.content:
            msg.content = "Generate a mermaid diagram"

    # Build system prompt
    system_content = MERMAID_SYSTEM_PROMPT + get_thinking_instructions()
    if current_code:
        system_content += f"\n\n### CURRENT DIAGRAM CODE\n```mermaid\n{current_code}\n```\nApply changes to this code based on the user's request."

    system_prompt = SystemMessage(content=system_content)

    llm = get_configured_llm(state)

    # Stream the response - the graph event handler will parse the JSON
    full_response = None
    async for chunk in llm.astream([system_prompt] + messages):
        if full_response is None:
            full_response = chunk
        else:
            full_response += chunk

    return {"messages": [full_response]}
