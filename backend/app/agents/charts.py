from langchain_core.messages import SystemMessage
from app.state.state import AgentState
from app.core.llm import get_configured_llm, get_thinking_instructions

CHARTS_SYSTEM_PROMPT = """你是一位世界级的数据可视化工程师和 ECharts 专家。你的目标是生成专业、有洞察力和美学上最先进的 ECharts 配置。

### 美学指南（高级设计）
- **现代调色板**：使用优雅、高对比度的调色板（例如，Pastel、Midnight 或 Apple 风格的鲜艳渐变）。
- **视觉深度**：对线图使用带有半透明渐变的 `areaStyle`。对条形图使用 `itemStyle: { borderRadius: [8, 8, 0, 0] }`。
- **排版**：设置干净、可读的字体样式。对标题与轴标签使用层次字体大小。
- **交互性**：始终启用带有 `axisPointer` 的 `tooltip` 和数据导出选项的 `toolbox`。

### 数据叙事原则
- **上下文清晰**：每个图表必须有清晰的 `title` 和突出关键收获的有洞察力的 `subtext`。
- **数据合成**：如果用户提供稀疏数据，合成专业、现实的数据集（例如，行业标准 KPI、季节趋势）以使可视化有价值。
- **战略选择**：选择最合适的图表类型（例如，多维分析的雷达图、转换的漏斗图、绩效指标的仪表盘）。
- **语言**：匹配用户输入语言。
- **中文输出要求**：如果用户用中文请求，则必须将所有标题、图例、轴标签、系列名、提示文本、类别名等翻译成流畅中文。仅在必要时保留广泛认可的技术缩写（例如 API、HTTP、URL、AI、ML、DL、NLP）。

### 输出格式
使用这些 XML 风格的标签输出你的回应：

<design_concept>
你的可视化策略和设计原理（1-3 句话）
</design_concept>

<code>
ECharts 选项 JSON（完整、有效的 JSON 对象）
</code>

示例输出：
<design_concept>
使用带有渐变颜色的条形图显示区域销售比较，带有交互式工具提示进行详细数据探索。
</design_concept>

<code>
{
  "backgroundColor": "transparent",
  "title": {"text": "Sales Report"},
  "series": [{"type": "bar", "data": [120, 200, 150]}]
}
</code>

只输出这两个标签，其他什么都不输出。
"""

def extract_current_code_from_messages(messages) -> str:
    """Extract the latest chart code from message history."""
    for msg in reversed(messages):
        # Check for tool messages (legacy format)
        if msg.type == "tool" and msg.content:
            stripped = msg.content.strip()
            if '"series":' in stripped or '"xAxis":' in stripped:
                return stripped
        # Check for AI messages with steps containing tool_end
        if msg.type == "ai" and hasattr(msg, 'additional_kwargs'):
            steps = msg.additional_kwargs.get('steps', [])
            for step in reversed(steps):
                if step.get('type') == 'tool_end' and step.get('content'):
                    content = step['content'].strip()
                    if '"series":' in content or '"xAxis":' in content:
                        return content
    return ""

async def charts_agent_node(state: AgentState):
    messages = state['messages']

    # Extract current code from history
    current_code = extract_current_code_from_messages(messages)

    # Safety: Ensure no empty text content blocks reach the LLM
    for msg in messages:
        if hasattr(msg, 'content') and not msg.content:
            msg.content = "Generate a chart"

    # Build system prompt
    system_content = CHARTS_SYSTEM_PROMPT + get_thinking_instructions()
    if current_code:
        system_content += f"\n\n### CURRENT CHART CODE\n```json\n{current_code}\n```\nApply changes to this code based on the user's request."

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
