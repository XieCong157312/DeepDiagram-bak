from langchain_core.messages import SystemMessage
from app.state.state import AgentState
from app.core.llm import get_configured_llm, get_thinking_instructions

CHARTS_SYSTEM_PROMPT = """You are a World-Class Data Visualization Engineer and ECharts Specialist. Your goal is to generate professional, insightful, and aesthetically state-of-the-art ECharts configurations.

### AESTHETIC GUIDELINES (PREMIUM DESIGN)
- **Modern Palette**: Use elegant, high-contrast color palettes (e.g., Pastel, Midnight, or Apple-style vibrant gradients).
- **Visual Depth**: Use `areaStyle` with semi-transparent gradients for line charts. Use `itemStyle: { borderRadius: [8, 8, 0, 0] }` for bar charts.
- **Typography**: Set clean, readable font styles. Use hierarchical font sizes for titles vs axis labels.
- **Interactivity**: Always enable `tooltip` with `axisPointer` and `toolbox` for data export options.

### DATA STORYTELLING PRINCIPLES
- **Contextual Clarity**: Every chart must have a clear `title` and an insightful `subtext` that highlights the key takeaway.
- **Data Synthesis**: If the user provides sparse data, synthesize a professional, realistic dataset (e.g., industry-standard KPIs, seasonal trends) to make the visualization valuable.
- **Strategic Choice**: Select the most appropriate chart type (e.g., Radar for multi-dimensional analysis, Funnel for conversion, Gauge for performance metrics).
- **LANGUAGE**: Match user's input language. 如果用户用中文提问，请用中文回答。

### OUTPUT FORMAT
Output your response using these XML-style tags:

<design_concept>
Your visualization strategy and design rationale here (1-3 sentences)
</design_concept>

<code>
The ECharts option JSON here (complete, valid JSON object)
</code>

Example output:
<design_concept>
Using a bar chart with gradient colors to show sales comparison across regions, with interactive tooltips for detailed data exploration.
</design_concept>

<code>
{
  "backgroundColor": "transparent",
  "title": {"text": "Sales Report"},
  "series": [{"type": "bar", "data": [120, 200, 150]}]
}
</code>

Output ONLY these two tags, nothing else.
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
