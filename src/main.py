import os
import operator
import yaml
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# --- 1. 配置加载 ---
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
llm_config = config.get("llm", {})
optimizer_llm_config = config.get("optimizer_llm", llm_config) # 如果未配置优化器，则使用默认LLM

load_dotenv()

# --- 2. 状态定义 (更新) ---
class AgencyState(TypedDict):
    # 输入
    raw_request: Optional[str]
    reference_prompt: Optional[str]
    optimization_goal: Optional[str]
    
    # 工作流产物
    creative_brief: str
    research_summary: str
    strategy_memo: str
    draft_prompt: str
    qa_report: dict
    revision_history: Annotated[List[str], operator.add]
    iteration_count: int
    final_package: str
    is_optimization: bool # 标记当前模式

# --- 3. 模型和工具初始化 ---
llm = ChatOpenAI(
    model=llm_config.get("model_name", "gpt-4o"), 
    temperature=llm_config.get("temperature", 0.7)
)
optimizer_llm = ChatOpenAI(
    model=optimizer_llm_config.get("model_name", "gpt-4-turbo"),
    temperature=optimizer_llm_config.get("temperature", 0.5)
)
search_tool = TavilySearchResults(max_results=3)


# --- 4. 节点定义 ---

# 4.1 需求分析节点（新的入口点）
def requirement_analyst_node(state: AgencyState):
    print("--- Agent: Requirement Analyst ---")
    if state.get("reference_prompt") and state.get("optimization_goal"):
        print("    Decision: Detected reference prompt. Routing to Optimization Mode.")
        return {"is_optimization": True}
    elif state.get("raw_request"):
        print("    Decision: Detected raw request. Routing to Creation Mode.")
        return {"is_optimization": False}
    else:
        raise ValueError("Input error: You must provide either 'raw_request' or both 'reference_prompt' and 'optimization_goal'.")

# 4.2 提示词优化师节点
def prompt_optimizer_node(state: AgencyState):
    print("--- Agent: Prompt Optimizer ---")
    brief = f"""
    **Project Title**: Prompt Optimization Initiative
    **Core Objective**: Refine an existing prompt based on specific user feedback.
    **Existing Prompt**: 
    ```
    {state['reference_prompt']}
    ```
    **Optimization Goal**: {state['optimization_goal']}
    **Output Format**: A revised, improved version of the prompt.
    """
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert Prompt Optimizer. Your task is to revise a given prompt based on a specific goal.
        Analyze the original prompt and the user's objective carefully.
        Apply advanced prompting techniques (e.g., clear instructions, role-playing, few-shot examples, structured output) to achieve the goal.
        
        **Creative Brief for this Optimization Task:**
        {brief}

        Produce the new, optimized prompt. Output only the prompt text itself, ready for use.
        """
    )
    chain = prompt | optimizer_llm
    optimized_prompt = chain.invoke({"brief": brief}).content
    
    return {
        "creative_brief": brief, # 存储简报以供QA使用
        "draft_prompt": optimized_prompt,
        "iteration_count": 0,
        "revision_history": []
    }

# 4.3 原有节点
def client_onboarding_node(state: AgencyState):
    print("--- Agent: Client Onboarding Specialist ---")
    prompt = ChatPromptTemplate.from_template(
        """You are a Client Onboarding Specialist at a top-tier creative agency.
        Your task is to convert a client's informal request into a structured "Creative Brief".
        Client Request: "{request}"
        Produce a Markdown document with these sections:
        - **Project Title**: A catchy name for this prompt generation project.
        - **Core Objective**: What is the ultimate goal the client wants to achieve with this prompt?
        - **Target Audience**: Who is the generated content for? Describe their demographics and psychographics.
        - **Key Elements & Constraints**: What must be included or excluded?
        - **Desired Tone & Style**: e.g., Professional, witty, academic, whimsical.
        - **Output Format**: What should the final output from the LLM look like? (e.g., JSON, list, blog post).
        """
    )
    chain = prompt | llm
    brief = chain.invoke({"request": state["raw_request"]}).content
    return {"creative_brief": brief, "iteration_count": 0, "revision_history": []}

def market_research_node(state: AgencyState):
    print("--- Agent: Market Research Analyst ---")
    prompt = ChatPromptTemplate.from_template(
        """You are a Market Research Analyst. Based on the Creative Brief, identify 3-4 key areas for research.
        Generate search queries for these areas.
        Creative Brief:\n{brief}\n\nOutput only a list of search queries, one per line."""
    )
    chain = prompt | llm
    queries_str = chain.invoke({"brief": state["creative_brief"]}).content
    queries = queries_str.strip().split("\n")
    print(f"    Conducting research with queries: {queries}")
    research_results = ""
    for query in queries:
        result = search_tool.invoke(query)
        research_results += f"Query: {query}\nResult:\n{result}\n\n"

    synthesis_prompt = ChatPromptTemplate.from_template(
        """You are a senior research analyst. Synthesize the raw research data into a concise "Research Summary" memo,
        focusing on actionable insights relevant to the Creative Brief.
        Creative Brief:\n{brief}\n\nRaw Research Data:\n{data}"""
    )
    synthesis_chain = synthesis_prompt | llm
    summary = synthesis_chain.invoke({"brief": state["creative_brief"], "data": research_results}).content
    return {"research_summary": summary}

def creative_council_node(state: AgencyState):
    print("--- Agent: Creative Council ---")
    prompt = ChatPromptTemplate.from_template(
        """You are the Creative Council. Based on the Creative Brief and Research Summary, 
        brainstorm 3 distinct, high-level creative strategies for the prompt's design.
        Creative Brief:\n{brief}\n\nResearch Summary:\n{research}\n\nOutput a "Strategy Memo"."""
    )
    chain = prompt | llm
    memo = chain.invoke({"brief": state["creative_brief"], "research": state["research_summary"]}).content
    return {"strategy_memo": memo}

def prompt_architect_node(state: AgencyState):
    print("--- Agent: Lead Prompt Architect ---")
    prompt = ChatPromptTemplate.from_template(
        """You are a Lead Prompt Architect. Synthesize all provided documents into a single, comprehensive, and effective prompt.
        Creative Brief:\n{brief}\n\nResearch Summary:\n{research}\n\nStrategy Memo:\n{strategy}\n
        Construct the prompt now. It should be clear, detailed, and directly address the Core Objective."""
    )
    chain = prompt | llm
    draft = chain.invoke({
        "brief": state["creative_brief"],
        "research": state["research_summary"],
        "strategy": state["strategy_memo"]
    }).content
    return {"draft_prompt": draft}

class QAReport(BaseModel):
    score: int = Field(description="A quality score from 1-10.")
    approval_status: str = Field(description="Must be 'Approved' or 'Revision Required'.")
    feedback: str = Field(description="Actionable, constructive feedback for revision.")

def qa_panel_node(state: AgencyState):
    print("--- Agent: Quality Assurance Panel ---")
    structured_llm = llm.with_structured_output(QAReport)
    prompt = ChatPromptTemplate.from_template(
        """You are the QA Panel. Evaluate the Draft Prompt based on the original Creative Brief.
        Creative Brief:\n{brief}\n\nDraft Prompt:\n{draft}\n
        Assess against: Alignment with Brief, Clarity, and Robustness.
        Provide a score (1-10), a decision ('Approved' or 'Revision Required'), and clear feedback."""
    )
    chain = prompt | structured_llm
    report = chain.invoke({"brief": state["creative_brief"], "draft": state["draft_prompt"]})
    qa_report_dict = report.dict()
    history_entry = f"Version {state['iteration_count']+1} QA: {qa_report_dict['approval_status']} (Score: {qa_report_dict['score']})\nFeedback: {qa_report_dict['feedback']}"
    print(f"    QA Decision: {qa_report_dict['approval_status']} | Score: {qa_report_dict['score']}")
    return {"qa_report": qa_report_dict, "revision_history": [history_entry]}

def revision_specialist_node(state: AgencyState):
    print("--- Agent: Revision Specialist ---")
    new_iteration_count = state['iteration_count'] + 1
    prompt = ChatPromptTemplate.from_template(
        """You are a Revision Specialist. Apply the QA Panel's feedback to improve the prompt.
        Current Draft Prompt:\n{draft}\n\nQA Panel Feedback:\n{feedback}\n
        Produce the revised prompt. Output only the new prompt text."""
    )
    chain = prompt | llm
    revised_draft = chain.invoke({
        "draft": state["draft_prompt"],
        "feedback": state["qa_report"]["feedback"]
    }).content
    return {"draft_prompt": revised_draft, "iteration_count": new_iteration_count}

def delivery_manager_node(state: AgencyState):
    print("--- Agent: Delivery Manager ---")
    prompt = ChatPromptTemplate.from_template(
        """You are the Delivery Manager. Assemble the final client-facing "Prompt Delivery Package".
        The package should be a well-formatted Markdown document containing:
        1. A thank you note.
        2. A summary of the prompt's objective.
        3. The final, approved prompt in a code block.
        4. Brief "Usage Guidelines".
        5. A summary of the revision history.
        
        Use this info:
        Final Prompt:\n{final_prompt}\n\nCreative Brief:\n{brief}\n\nRevision History:\n{history}"""
    )
    chain = prompt | llm
    package = chain.invoke({
        "final_prompt": state["draft_prompt"],
        "brief": state["creative_brief"],
        "history": "\n".join(state["revision_history"])
    }).content
    return {"final_package": package}


# --- 5. 路由器定义 (更新) ---
def mode_router(state: AgencyState):
    """根据 is_optimization 标志决定工作流路径。"""
    if state["is_optimization"]:
        return "optimizer"
    else:
        return "creator"

def qa_router(state: AgencyState):
    print("--- Router: Making QA decision ---")
    if state["iteration_count"] >= 3:
        print("    Decision: Max iterations reached. Approving.")
        return "deliver"
    status = state["qa_report"]["approval_status"]
    if status == "Approved":
        print("    Decision: Prompt approved. Proceeding to delivery.")
        return "deliver"
    else:
        print("    Decision: Revision required. Looping back.")
        return "revise"

# --- 6. 构建图 (Graph) ---
workflow = StateGraph(AgencyState)

# 添加所有节点
workflow.add_node("analyst", requirement_analyst_node)
workflow.add_node("onboarding", client_onboarding_node)
workflow.add_node("research", market_research_node)
workflow.add_node("council", creative_council_node)
workflow.add_node("architect", prompt_architect_node)
workflow.add_node("optimizer", prompt_optimizer_node) # 新优化节点
workflow.add_node("qa_panel", qa_panel_node)
workflow.add_node("reviser", revision_specialist_node)
workflow.add_node("delivery", delivery_manager_node)

# 设置入口点
workflow.set_entry_point("analyst")

# 添加条件路由来选择模式
workflow.add_conditional_edges(
    "analyst",
    mode_router,
    {
        "creator": "onboarding",
        "optimizer": "optimizer"
    }
)

# 定义“从零创造”路径
workflow.add_edge("onboarding", "research")
workflow.add_edge("research", "council")
workflow.add_edge("council", "architect")
workflow.add_edge("architect", "qa_panel")

# 定义“优化”路径 - 直接跳到QA环节
workflow.add_edge("optimizer", "qa_panel")

# 定义 QA 后的循环和结束逻辑
workflow.add_conditional_edges(
    "qa_panel",
    qa_router,
    {
        "revise": "reviser",
        "deliver": "delivery"
    }
)
workflow.add_edge("reviser", "qa_panel")
workflow.add_edge("delivery", END)

# 编译图
app = workflow.compile()

# 可视化
try:
    print("\n--- Agency Workflow Diagram ---")
    print(app.get_graph().draw_ascii())
    print("-----------------------------\n")
except Exception as e:
    print(f"Could not draw graph: {e}")

print("\n" + "#"*70)
print("### RUNNING SCENARIO 1: CREATION MODE ###")
print("#"*70 + "\n")

creation_request = {
    "raw_request": ("I need a prompt that generates a detailed character persona for a Dungeons & Dragons campaign. "
                    "The persona should be for a 'rogue with a heart of gold' archetype, "
                    "but I want it to be unique and avoid clichés. The output should be a structured markdown file.")
}
final_state_creation = app.invoke(creation_request)

print("\n\n" + "="*50)
print(f"     PROMPTCRAFT AGENCY: FINAL DELIVERY (Creation Mode)")
print("="*50 + "\n")
print(final_state_creation['final_package'])
print("\n" + "="*50)

print("\n\n" + "#"*70)
print("### RUNNING SCENARIO 2: OPTIMIZATION MODE ###")
print("#"*70 + "\n")

optimization_request = {
    "reference_prompt": "Tell me about Elon Musk.",
    "optimization_goal": "I need to revise this prompt so that the output is always a balanced, neutral biography focusing equally on his achievements and controversies. The output must be exactly 3 paragraphs long."
}
final_state_optimization = app.invoke(optimization_request)

print("\n\n" + "="*50)
print(f"     PROMPTCRAFT AGENCY: FINAL DELIVERY (Optimization Mode)")
print("="*50 + "\n")
print(final_state_optimization['final_package'])
print("\n" + "="*50)
