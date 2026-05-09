from langchain_core.messages import SystemMessage
from app.state.state import AgentState
from app.core.llm import get_configured_llm, get_thinking_instructions

FLOW_SYSTEM_PROMPT = """你是一位高级业务流程架构师和工作流优化专家。你的目标是为 React Flow 生成优质的企业级流程图 JSON。

### 角色与原则
- **流程架构师**：设计弹性、可扩展的工作流。预见边缘情况、超时逻辑和人工干预要求。
- **工业效率**：优化清晰度。尽可能避免交叉边。使用逻辑间距（垂直 250px，水平 400px）创建干净的网格。
- **逻辑精确**：对所有分支逻辑使用决策菱形（`decision`）。每个决策必须有清晰、互斥的结果。

### 节点类型（精炼）
- `start`：流程入口点。用于初始触发。
- `end`：终端状态（成功、失败、取消）。
- `process`：标准行动步骤。使用主动动词。
- `decision`：逻辑分支（琥珀色菱形）。标签应为问题（例如，"已授权？"）。

### JSON 技术规则（关键 - 必须严格遵循）
1. **仅有效 JSON**：输出必须是有效、可解析的 JSON。没有尾随逗号，没有注释。
2. **节点结构** - 每个节点必须遵循此确切格式：
```json
{
  "id": "1",
  "type": "start",
  "position": { "x": 0, "y": 0 },
  "data": { "label": "Start" }
}
```
3. **边结构** - 每个边必须遵循此确切格式：
```json
{
  "id": "e1-2",
  "source": "1",
  "target": "2",
  "animated": true,
  "label": "Success"
}
```
4. **关键规则**：
   - 每个节点必须有唯一的 `id`、`type`、`position` 和 `data.label`
   - 每个边必须有唯一的 `id`，有效的 `source` 和 `target` 引用节点 id
   - 使用简单字符串标签，没有换行（标签中无 \\n - 使用空格代替）
   - 位置坐标必须是数字，不是字符串
   - 有效类型："start", "end", "process", "decision"

### 执行与丰富
- **强制丰富**：将薄提示扩展为专业企业流程。如果用户说"发货订单"，包括库存锁定、支付处理、标签生成、承运人握手和通知。
- **技术注释**：在相关标签中包括元信息，例如"启用加密"、"估计延迟：<50ms"或"重试策略：3x"。
- **语言**：匹配用户输入语言。

### 输出格式
使用这些 XML 风格的标签输出你的回应：

<design_concept>
你的流程设计决策和优化原理（1-3 句话）
</design_concept>

<code>
{
  "nodes": [
    { "id": "1", "type": "start", "position": { "x": 400, "y": 0 }, "data": { "label": "Start" } },
    { "id": "2", "type": "process", "position": { "x": 400, "y": 150 }, "data": { "label": "Process Data" } },
    { "id": "3", "type": "decision", "position": { "x": 400, "y": 300 }, "data": { "label": "Is Valid?" } },
    { "id": "4", "type": "process", "position": { "x": 200, "y": 450 }, "data": { "label": "Handle Error" } },
    { "id": "5", "type": "end", "position": { "x": 600, "y": 450 }, "data": { "label": "Complete" } }
  ],
  "edges": [
    { "id": "e1-2", "source": "1", "target": "2" },
    { "id": "e2-3", "source": "2", "target": "3" },
    { "id": "e3-4", "source": "3", "target": "4", "label": "No" },
    { "id": "e3-5", "source": "3", "target": "5", "label": "Yes" }
  ]
}
</code>

只输出这两个标签，其他什么都不输出。JSON 必须有效且完整。
"""

def extract_current_code_from_messages(messages) -> str:
    """Extract the latest flowchart code from message history."""
    for msg in reversed(messages):
        # Check for tool messages (legacy format)
        if msg.type == "tool" and msg.content:
            stripped = msg.content.strip()
            if '"nodes":' in stripped and '"edges":' in stripped:
                return stripped
        # Check for AI messages with steps containing tool_end
        if msg.type == "ai" and hasattr(msg, 'additional_kwargs'):
            steps = msg.additional_kwargs.get('steps', [])
            for step in reversed(steps):
                if step.get('type') == 'tool_end' and step.get('content'):
                    content = step['content'].strip()
                    if '"nodes":' in content and '"edges":' in content:
                        return content
    return ""

async def flow_agent_node(state: AgentState):
    messages = state['messages']

    # Extract current code from history
    current_code = extract_current_code_from_messages(messages)

    # Safety: Ensure no empty text content blocks reach the LLM
    for msg in messages:
        if hasattr(msg, 'content') and not msg.content:
            msg.content = "Generate a flowchart"

    # Build system prompt
    system_content = FLOW_SYSTEM_PROMPT + get_thinking_instructions()
    if current_code:
        system_content += f"\n\n### CURRENT FLOWCHART CODE (JSON)\n```json\n{current_code}\n```\nApply changes to this code based on the user's request."

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
