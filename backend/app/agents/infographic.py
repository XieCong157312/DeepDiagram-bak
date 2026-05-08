from langchain_core.messages import SystemMessage, HumanMessage
from app.state.state import AgentState
from app.core.llm import get_configured_llm, get_thinking_instructions
from app.data.template_syntax import (
    TEMPLATES,
    ALL_TEMPLATES,
    get_template_category,
    get_syntax_rules_for_template,
    get_data_field_for_template,
    get_common_syntax_rules,
)

# Step 1: Template selection prompt
TEMPLATE_SELECTOR_PROMPT = """You are a professional infographic design consultant. Your task is to select the BEST template for the user's needs.

### Available Templates by Category

**Chart Templates (chart-*)** - For data visualization with numeric values
{chart_templates}

**Compare Templates (compare-*)** - For comparisons, SWOT analysis, pros/cons
{compare_templates}

**Hierarchy Templates (hierarchy-*)** - For tree structures, org charts, mind maps
{hierarchy_templates}

**List Templates (list-*)** - For displaying items in rows, columns, or grids
{list_templates}

**Relation Templates (relation-*)** - For flowcharts and relationship diagrams
{relation_templates}

**Sequence Templates (sequence-*)** - For processes, timelines, step-by-step flows
{sequence_templates}

### Template Selection Guide
- **Data with numbers/percentages**: chart-* (pie, bar, column, line)
- **Process/Steps/Timeline**: sequence-* (timeline, stairs, snake-steps, roadmap)
- **Pros vs Cons / Two-sided comparison**: compare-binary-*
- **SWOT Analysis**: compare-swot
- **4 Quadrants**: compare-quadrant-*
- **Feature list / Multiple items**: list-* (grid, row, column)
- **Organization chart / Tree structure**: hierarchy-tree-*
- **Mind map / Brainstorming**: hierarchy-mindmap-*
- **Flowchart / Workflow**: relation-dagre-flow-*
- **Word cloud**: chart-wordcloud

### OUTPUT FORMAT
You MUST output ONLY the template name, nothing else. Example:
chart-pie-compact-card

Analyze the user's request and select the single most appropriate template.
"""

# Step 2: Code generation prompt (template-specific)
CODE_GENERATOR_PROMPT = """You are a World-Class Graphic Designer. Generate AntV Infographic DSL syntax for the template: {template_name}

### TEMPLATE CATEGORY: {category}

### DATA STRUCTURE FOR THIS TEMPLATE
This template uses the `{data_field}` field for data items.

### TEMPLATE-SPECIFIC RULES
{syntax_rules}

### SYNTAX EXAMPLE FOR THIS TEMPLATE
```
{syntax_example}
```

{common_syntax_rules}

{additional_syntax}

### DESIGN PHILOSOPHY
- **Narrative Flow**: Tell a story, not just present data
- **Visual Metaphor**: Select meaningful icons
- **Aesthetic Balance**: Professional color palettes
- **LANGUAGE**: Match user's input language. 如果用户用中文提问，请用中文回答。
- **CRITICAL**: ALL field values MUST be plain strings. NEVER use arrays, objects, or nested structures.
- **NO COMMENTS**: NEVER include comments (// or /* */) in the DSL.

### OUTPUT FORMAT
Output your response using these XML-style tags:

<design_concept>
Your creative direction and design rationale here (1-3 sentences)
</design_concept>

<code>
The AntV Infographic DSL code here (raw DSL, no markdown fences)
</code>

Output ONLY these two tags, nothing else.
"""


def build_template_selector_prompt() -> str:
    """Build the template selector prompt with all available templates."""
    return TEMPLATE_SELECTOR_PROMPT.format(
        chart_templates=", ".join(TEMPLATES["chart"]),
        compare_templates=", ".join(TEMPLATES["compare"]),
        hierarchy_templates=", ".join(TEMPLATES["hierarchy"]),
        list_templates=", ".join(TEMPLATES["list"]),
        relation_templates=", ".join(TEMPLATES["relation"]),
        sequence_templates=", ".join(TEMPLATES["sequence"]),
    )


def build_code_generator_prompt(template_name: str) -> str:
    """Build the code generator prompt for a specific template."""
    category = get_template_category(template_name)
    data_field = get_data_field_for_template(template_name)
    rules = get_syntax_rules_for_template(template_name)

    syntax_rules = "\n".join([f"- {note}" for note in rules.get("notes", [])])
    syntax_example = rules.get("syntax_example", "")

    # Build additional syntax for special cases
    additional_syntax = ""
    if category == "relation":
        relation_syntax = rules.get("relation_syntax", [])
        if relation_syntax:
            additional_syntax = "### RELATION EDGE SYNTAX\n" + "\n".join([f"- {s}" for s in relation_syntax])
    elif category == "compare":
        special_syntax = rules.get("special_syntax", {})
        if "compare-binary" in template_name and "compare-binary" in special_syntax:
            additional_syntax = f"### BINARY COMPARE EXAMPLE\n```\n{special_syntax['compare-binary']}\n```"
        elif "compare-quadrant" in template_name and "compare-quadrant" in special_syntax:
            additional_syntax = f"### QUADRANT EXAMPLE\n```\n{special_syntax['compare-quadrant']}\n```"

    return CODE_GENERATOR_PROMPT.format(
        template_name=template_name,
        category=category.upper(),
        data_field=data_field,
        syntax_rules=syntax_rules,
        syntax_example=syntax_example,
        common_syntax_rules=get_common_syntax_rules(),
        additional_syntax=additional_syntax,
    )


def extract_current_code_from_messages(messages) -> str:
    """Extract the latest infographic code from message history."""
    for msg in reversed(messages):
        if msg.type == "tool" and msg.content:
            stripped = msg.content.strip()
            if stripped.startswith('infographic '):
                return stripped
        if msg.type == "ai" and hasattr(msg, 'additional_kwargs'):
            steps = msg.additional_kwargs.get('steps', [])
            for step in reversed(steps):
                if step.get('type') == 'tool_end' and step.get('content'):
                    content = step['content'].strip()
                    if content.startswith('infographic '):
                        return content
    return ""


def extract_template_from_code(code: str) -> str:
    """Extract template name from existing infographic code."""
    if code.startswith('infographic '):
        first_line = code.split('\n')[0]
        parts = first_line.split(' ', 1)
        if len(parts) > 1:
            return parts[1].strip()
    return ""


async def select_template(llm, user_request: str) -> str:
    """Step 1: Use LLM to select the best template for the user's needs."""
    selector_prompt = SystemMessage(content=build_template_selector_prompt())
    selection_message = HumanMessage(content=f"Select the best template for: {user_request}")

    response = await llm.ainvoke([selector_prompt, selection_message])
    template_name = response.content.strip()

    # Validate template name
    if template_name in ALL_TEMPLATES:
        return template_name

    # Try to find a matching template
    for template in ALL_TEMPLATES:
        if template in template_name or template_name in template:
            return template

    # Default fallback
    return "list-row-horizontal-icon-arrow"


async def infographic_agent_node(state: AgentState):
    messages = state['messages']

    # Extract current code from history
    current_code = extract_current_code_from_messages(messages)

    # Safety: Ensure no empty text content blocks reach the LLM
    for msg in messages:
        if hasattr(msg, 'content') and not msg.content:
            msg.content = "Generate an infographic"

    llm = get_configured_llm(state)

    # Get the user's request (last human message)
    user_request = ""
    for msg in reversed(messages):
        if msg.type == "human":
            user_request = msg.content
            break

    # Determine template to use
    if current_code:
        # If modifying existing code, use the same template
        template_name = extract_template_from_code(current_code)
        if not template_name:
            template_name = await select_template(llm, user_request)
    else:
        # Step 1: Select the best template
        template_name = await select_template(llm, user_request)

    # Step 2: Generate code using template-specific prompt
    code_prompt = build_code_generator_prompt(template_name)
    system_content = code_prompt + get_thinking_instructions()

    if current_code:
        system_content += f"\n\n### CURRENT INFOGRAPHIC CODE\n```\n{current_code}\n```\nApply changes to this code based on the user's request."

    system_prompt = SystemMessage(content=system_content)

    # Stream the response - the graph event handler will parse the JSON
    full_response = None
    async for chunk in llm.astream([system_prompt] + messages):
        if full_response is None:
            full_response = chunk
        else:
            full_response += chunk

    return {"messages": [full_response]}
