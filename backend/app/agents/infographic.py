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
TEMPLATE_SELECTOR_PROMPT = """你是一位专业的信息图设计顾问。你的任务是为用户的需求选择最佳模板。

### 可用模板按类别

**图表模板 (chart-*)** - 用于带有数值的数据可视化
{chart_templates}

**比较模板 (compare-*)** - 用于比较、SWOT 分析、优缺点
{compare_templates}

**层次模板 (hierarchy-*)** - 用于树结构、组织图、思维导图
{hierarchy_templates}

**列表模板 (list-*)** - 用于在行、列或网格中显示项目
{list_templates}

**关系模板 (relation-*)** - 用于流程图和关系图
{relation_templates}

**序列模板 (sequence-*)** - 用于流程、时间线、逐步流程
{sequence_templates}

### 模板选择指南
- **带有数字/百分比的数据**：chart-* (pie, bar, column, line)
- **流程/步骤/时间线**：sequence-* (timeline, stairs, snake-steps, roadmap)
- **优缺点 / 两侧比较**：compare-binary-*
- **SWOT 分析**：compare-swot
- **4 象限**：compare-quadrant-*
- **功能列表 / 多个项目**：list-* (grid, row, column)
- **组织图 / 树结构**：hierarchy-tree-*
- **思维导图 / 头脑风暴**：hierarchy-mindmap-*
- **流程图 / 工作流**：relation-dagre-flow-*
- **词云**：chart-wordcloud

### 输出格式
你必须只输出模板名称，其他什么都不输出。示例：
chart-pie-compact-card

Analyze the user's request and select the single most appropriate template.
"""

# Step 2: Code generation prompt (template-specific)
CODE_GENERATOR_PROMPT = """你是一位世界级的图形设计师。为模板生成 AntV Infographic DSL 语法：{template_name}

### 模板类别：{category}

### 此模板的数据结构
此模板使用 `{data_field}` 字段作为数据项。

### 模板特定规则
{syntax_rules}

### 此模板的语法示例
```
{syntax_example}
```

{common_syntax_rules}

{additional_syntax}

### 设计哲学
- **叙事流**：讲述故事，而不仅仅呈现数据
- **视觉隐喻**：选择有意义的图标
- **美学平衡**：专业调色板
- **语言**：匹配用户输入语言
- **关键**：所有字段值必须是纯字符串。永远不要使用数组、对象或嵌套结构。
- **无注释**：永远不要在 DSL 中包含注释（// 或 /* */）。

### 输出格式
使用这些 XML 风格的标签输出你的回应：

<design_concept>
你的创意方向和设计原理（1-3 句话）
</design_concept>

<code>
AntV Infographic DSL 代码（原始 DSL，无 markdown 围栏）
</code>

只输出这两个标签，其他什么都不输出。
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
