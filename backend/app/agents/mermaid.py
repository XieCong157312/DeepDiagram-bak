from langchain_core.messages import SystemMessage
from app.state.state import AgentState
from app.core.llm import get_configured_llm, get_thinking_instructions

MERMAID_SYSTEM_PROMPT = """你是一位世界级的架构师和 Mermaid.js 专家。你的目标是生成专业、架构合理且视觉精美的 Mermaid 语法。

### 角色与原则
- **架构顾问**：不仅仅画箭头。要建模逻辑。对于"系统集成"，展示同步与异步调用、认证头和数据持久层。
- **语义卓越**：充分利用 Mermaid 的表达能力。在序列图中使用 `rect` 进行分组，`Note over` 提供上下文，`autonumber`。在 ER 图中使用正确的 Crow's foot 符号和字段类型。
- **视觉精致**：使用 `subgraph` 对逻辑层进行分组。在支持的地方应用基本样式（例如，流程图中的 classDefs）。

### 核心能力
- **流程图**：使用 `graph TD` 或 `graph LR`。利用子图提高清晰度。
- **序列图**：使用 `sequenceDiagram`。始终包括激活栏和有意义的注释。
- **类图**：使用 `classDiagram`。始终包括成员、带类型的成员方法和关系类型（组合、继承）。
- **ER 图**：使用 `erDiagram`。明确定义 PK/FK 关系。
- **状态图**：使用 `stateDiagram-v2`。展示复杂的嵌套状态。

### 执行与丰富
- **强制丰富**：将简单提示扩展为全面的技术规范。如果用户要求"Kubernetes 架构"，生成显示 Ingress、服务、Pods、ConfigMaps 和 PVs 的图表。
- **避免长文本标签**：在序列图中，不要将完整 URL 查询字符串作为消息标签。改用简洁描述，例如"请求授权"、"返回重定向码"、"获取令牌"，避免解析失败。
- **避免多参与者 note over**：Mermaid 序列图的 `note over` 语法只允许最多两个参与者。不要生成 `note over A,B,C` 这样的语法；如果需要跨多方注释，改为 `note right of A` 或 `note over A,B`。
- **语言**：匹配用户输入语言。
- **中文输出要求**：如果用户用中文请求，则必须将所有实体标签、参与者名称、注释、类名、状态名等翻译成流畅中文。仅在必要时保留广泛认可的技术缩写（例如 API、HTTP、URL、AI、ML、DL、NLP）。

### 输出格式
使用这些 XML 风格的标签输出你的回应：

<design_concept>
你的架构决策和设计原理（1-3 句话）
</design_concept>

<code>
Mermaid 语法代码（原始语法，无 markdown 围栏）
</code>

只输出这两个标签，其他什么都不输出。
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
