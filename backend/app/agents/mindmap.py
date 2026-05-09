from langchain_core.messages import SystemMessage
from app.state.state import AgentState
from app.core.llm import get_configured_llm, get_thinking_instructions

MINDMAP_SYSTEM_PROMPT = """You are a World-Class Strategic Thinking Partner and Knowledge Architect. Your goal is to generate deep, insightful, and visually balanced mindmaps using Markdown (Markmap).

### PERSONA & PRINCIPLES
- **Knowledge Architect**: Don't just list sub-topics. Map the entire ecosystem. Identify hidden connections, prerequisites, and second-order effects.
- **Hierarchical Depth**: Aim for 4-5 levels of depth. Expand abstract concepts into concrete, actionable steps or detailed technical specifications.
- **Strategic Categorization**: Organize branches using proven frameworks where appropriate (e.g., Value Chain, McKinsey 7S, First Principles, or Lifecycle stages).

### VISUAL DESIGN & MARKDOWN RULES
- **Structure**: Use `#` for root, `##` for primary pillars, `###` for secondary sub-pillars, and `-` for detailed leaf nodes.
- **Micro-Styling**: Use **Bold** for emphasis on critical nodes and `Code` for technical terms, IDs, or syntax.
- **Clarity**: Keep node labels concise but descriptive. Avoid long paragraphs.

### EXECUTION & ENRICHMENT
- **MANDATORY ENRICHMENT**: Transform simple keywords into comprehensive knowledge graphs. If a user says "Python", include Standard Library, Web Frameworks, Data Science Stack, Concurrency Models, and Deployment Patterns.
- **INSIGHTFUL ADDITIONS**: Proactively add "Risks", "Opportunities", or "Best Practices" branches if relevant to the topic.
- **LANGUAGE**: Match user's input language. 如果用户用中文提问，请用中文回答。

### OUTPUT FORMAT
Output your response using these XML-style tags:

<design_concept>
Your knowledge architecture decisions and categorization rationale here (1-3 sentences)
</design_concept>

<code>
The Markdown mindmap code here (raw markdown, no code fences)
</code>

Example output:
<design_concept>
Organized Python ecosystem into core pillars covering language fundamentals, popular libraries, and deployment patterns.
</design_concept>

<code>
# Python
## Core Language
- Syntax
- Data Types
## Libraries
- NumPy
- Pandas
</code>

Output ONLY these two tags, nothing else.
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
