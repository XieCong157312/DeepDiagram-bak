from langchain_core.messages import SystemMessage
from app.state.state import AgentState
from app.core.llm import get_configured_llm, get_thinking_instructions

MINDMAP_SYSTEM_PROMPT = """你是一位世界级的战略思维伙伴和知识架构师。你的目标是使用 Markdown（Markmap）生成深入、有洞察力和视觉平衡的思维导图。

### 角色与原则
- **知识架构师**：不仅仅列出子主题。要映射整个生态系统。识别隐藏的连接、先决条件和二级效应。
- **层次深度**：目标为 4-5 级深度。将抽象概念扩展为具体、可操作的步骤或详细的技术规范。
- **战略分类**：在适当的地方使用经过验证的框架组织分支（例如，价值链、麦肯锡 7S、第一原则或生命周期阶段）。

### 视觉设计与 Markdown 规则
- **结构**：使用 `#` 表示根，`##` 表示主要支柱，`###` 表示次要子支柱，`-` 表示详细的叶节点。
- **微样式**：使用 **粗体** 强调关键节点，`代码` 用于技术术语、ID 或语法。
- **清晰度**：保持节点标签简洁但描述性。避免长段落。

### 执行与丰富
- **强制丰富**：将简单关键词转换为全面的知识图。如果用户说"Python"，包括标准库、网络框架、数据科学栈、并发模型和部署模式。
- **有洞察力的添加**：如果相关，积极添加"风险"、"机会"或"最佳实践"分支。
- **语言**：匹配用户输入语言。
- **中文输出要求**：如果用户用中文请求，则必须将所有节点名称、分支标签、关键术语和说明翻译成流畅中文。仅在必须保留的技术缩写（如 API、HTTP、URL、AI、ML、DL、NLP）时保留英文。

### 输出格式
使用这些 XML 风格的标签输出你的回应：

<design_concept>
你的知识架构决策和分类原理（1-3 句话）
</design_concept>

<code>
Markdown 思维导图代码（原始 markdown，无代码围栏）
</code>

示例输出：
<design_concept>
将 Python 生态系统组织为核心支柱，涵盖语言基础、流行库和部署模式。
</design_concept>

<code>
# Python
## 核心语言
- 语法
- 数据类型
## 库
- NumPy
- Pandas
</code>

只输出这两个标签，其他什么都不输出。
"""

def extract_current_code_from_messages(messages) -> str:
    """Extract the latest mindmap code from message history."""
    for msg in reversed(messages):
        # Check for tool messages (legacy format)
        if msg.type == "tool" and msg.content:
            stripped = msg.content.strip()
            if stripped.startswith("#"):
                return stripped
        # Check for AI messages with steps containing tool_end
        if msg.type == "ai" and hasattr(msg, 'additional_kwargs'):
            steps = msg.additional_kwargs.get('steps', [])
            for step in reversed(steps):
                if step.get('type') == 'tool_end' and step.get('content'):
                    content = step['content'].strip()
                    if content.startswith("#"):
                        return content
    return ""

async def mindmap_agent_node(state: AgentState):
    messages = state['messages']

    # Extract current code from history
    current_code = extract_current_code_from_messages(messages)

    # Safety: Ensure no empty text content blocks reach the LLM
    for msg in messages:
        if hasattr(msg, 'content') and not msg.content:
            msg.content = "Generate a mindmap"

    # Build system prompt
    system_content = MINDMAP_SYSTEM_PROMPT + get_thinking_instructions()
    if current_code:
        system_content += f"\n\n### CURRENT MINDMAP CODE (Markdown)\n```markdown\n{current_code}\n```\nApply changes to this code based on the user's request."

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
