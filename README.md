
<div align="center">
  <img src="https://img.shields.io/badge/Framework-LangGraph-purple?style=for-the-badge&logo=github" alt="Framework Badge">
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python" alt="Language Badge">
  <img src="https://img.shields.io/badge/Paradigm-Multi--Agent_System-orange?style=for-the-badge&logo=openai" alt="Paradigm Badge">
  <img src="https://img.shields.io/github/stars/cotix-ai/Prompt-Generator?style=for-the-badge&color=gold" alt="Stars Badge">
</div>

<br>

<h1 align="center">
  PromptCraft Agency: An Autonomous Multi-Agent System for Advanced Prompt Engineering
</h1>

<p align="center">
  <i>Bridging the gap between a simple request and a production-ready, highly-optimized LLM prompt.</i>
</p>

<br>

>[!IMPORTANT]
> **Core Idea**: PromptCraft Agency is an autonomous system that simulates a specialized creative agency, using a team of AI agents to collaboratively transform a user's high-level goal into a robust, tested, and polished LLM prompt.

## Table of Contents

- [✨ Introduction](#-introduction)
- [💡 Core Philosophy](#-core-philosophy)
- [🧠 Architectural Core](#-architectural-core)
- [🧩 Core Components Explained](#-core-components-explained)
    - [Agent 1: Client Onboarding Specialist](#agent-1-client-onboarding-specialist)
    - [Agent 2: Market Research Analyst](#agent-2-market-research-analyst)
    - [Agent 3: Prompt Optimizer](#agent-3-prompt-optimizer)
    - [Agent 4: Quality Assurance Panel](#agent-4-quality-assurance-panel)
- [🔄 Workflow](#-workflow)
- [🚀 Unique Advantages & Innovations](#-unique-advantages--innovations)
- [🛠️ Quick Start](#️-quick-start)
- [🤝 How to Contribute](#-how-to-contribute)
- [📄 License](#-license)

<br>

---

## ✨ Introduction

This project introduces **PromptCraft Agency**, a novel framework built on **LangGraph** that elevates the process of prompt engineering by orchestrating a team of specialized AI agents.

**PromptCraft Agency** redefines prompt creation not as a single, monolithic task, but as a structured, collaborative workflow similar to a real-world creative agency. It moves beyond the limitations of manual, trial-and-error prompt writing, which often lacks rigor and context. This architecture synergizes the structured reasoning of LLMs with real-world data from web searches, creating a robust and deliberate system that can either generate novel prompts from scratch or refine existing ones based on specific goals.

<br>

---

## 💡 Core Philosophy

**PromptCraft Agency** is more than just a prompt generator; it represents a fundamental shift in how we approach interaction with LLMs. We believe the next leap in AI application development requires systems that can structure their own creative and analytical processes, just as human expert teams do.

> "The future of AI-powered solutions lies not in better single-shot prompts, but in building autonomous, self-correcting systems that manage the entire lifecycle of a task."

This design aims to overcome the inherent limitations of single-agent approaches, where a lack of diverse perspectives or external grounding can lead to suboptimal or brittle outputs.

<br>

---

## 🧠 Architectural Core

The **StateGraph** is the cornerstone of the **PromptCraft Agency** architecture and serves as the **"single source of truth"** for the entire workflow. This mechanism liberates the system from the constraints of a simple, linear chain of thought.

**Core Functionality:**
The system operates by coordinating a "team" of AI agents, each with a distinct role and responsibility, passing a shared `AgencyState` object between them:
1.  **Requirement Analysis**: Determines the user's intent—either to create a new prompt or optimize an existing one.
2.  **Creative Briefing**: Translates a vague user request into a structured, actionable plan.
3.  **Research & Synthesis**: Gathers and synthesizes external information to ground the prompt in reality.
4.  **Drafting & Optimization**: Constructs or refines the prompt based on all available information.
5.  **Quality Assurance & Revision**: Implements a rigorous, iterative review cycle to ensure the prompt meets all objectives.

Therefore, the final output is not the result of a single guess, but the product of a validated, multi-step process that mirrors professional best practices.

<br>

---

## 🧩 Core Components Explained

The agents within **PromptCraft Agency** function with a clear division of labor, collaborating to achieve an intelligent, holistic process.

### Agent 1: Client Onboarding Specialist (Role: Strategist)
*   **Objective:** To transform a user's raw, informal request into a structured "Creative Brief."
*   **Implementation:** This agent queries an LLM with a template designed to extract key project parameters: the core objective, target audience, key constraints, desired tone, and output format. This brief becomes the foundational document for the entire project.

### Agent 2: Market Research Analyst (Role: Researcher)
*   **Objective:** To enrich the creative process with real-world context and data, preventing generic or naive outputs.
*   **Implementation:** The agent first uses an LLM to generate relevant search queries based on the Creative Brief. It then executes these queries using the `TavilySearchResults` tool and synthesizes the findings into a concise "Research Summary."

### Agent 3: Prompt Optimizer (Role: Refiner)
*   **Objective:** To revise a user's existing prompt based on a specific optimization goal.
*   **Implementation:** This agent is activated in the "Optimization Mode." It analyzes the user's reference prompt and their stated goal, then leverages an LLM (potentially a more powerful model designated for this task) to apply advanced prompting techniques and generate an improved version.

### Agent 4: Quality Assurance Panel (Role: Critic)
*   **Objective:** To rigorously evaluate the draft prompt against the Creative Brief and provide actionable feedback for improvement.
*   **Implementation:** This agent uses an LLM with a structured output function (`with_structured_output`) to produce a `QAReport` containing a quality score (1-10), an approval status ('Approved' or 'Revision Required'), and detailed feedback. This enables a robust, automated review loop.

<br>

---

## 🔄 Workflow

**PromptCraft Agency** follows a clear, state-driven workflow managed by LangGraph:

1.  **Analysis (Routing):** The `Requirement Analyst` node inspects the initial input to determine the workflow path: **Creation Mode** or **Optimization Mode**.
2.  **Creation Path:**
    *   **Onboarding:** The `Client Onboarding Specialist` creates the `Creative Brief`.
    *   **Research:** The `Market Research Analyst` gathers external data.
    *   **Strategy:** The `Creative Council` brainstorms approaches.
    *   **Drafting:** The `Prompt Architect` writes the initial prompt.
3.  **Optimization Path:**
    *   The `Prompt Optimizer` directly refines the user's provided prompt, creating a new draft.
4.  **Review Loop (Shared):**
    *   **QA:** The `QA Panel` evaluates the draft prompt. Based on its decision, the graph routes to either `Revise` or `Deliver`.
    *   **Revision:** The `Revision Specialist` applies the QA feedback and sends the updated prompt back to the `QA Panel`. This loop continues until approval or a max iteration limit is reached.
5.  **Delivery (Termination):** The `Delivery Manager` assembles the final, polished prompt into a "Prompt Delivery Package" for the user, and the workflow ends.

<br>

---

## 🚀 Unique Advantages & Innovations

While advanced prompt engineering techniques have improved LLM outputs, many still operate on a single forward-pass model. There is significant room for improvement in **robustness, error correction, and overcoming initial flawed assumptions.**

**This is precisely what PromptCraft Agency is designed to address.**

Through its unique **multi-agent, state-driven architecture**, PromptCraft Agency offers:

*   **Significantly Reduced "Hallucination":** By grounding the prompt creation process with external web research, the system is less likely to generate factually incorrect or uninspired content.
*   **True Iterative Improvement:** The built-in QA and revision loop automates the critical process of testing and refinement, which is often done manually and haphazardly.
*   **Dual-Mode Functionality:** The system is versatile, capable of both greenfield creation and targeted optimization of existing assets—a key requirement for real-world applications.
*   **Configurable Expertise:** By allowing users to define different LLMs for different agents (e.g., a more powerful model for optimization) via `config.yaml`, the system's performance and cost can be finely tuned.

<br>

---

## 🤝 How to Contribute

We welcome and encourage contributions to this project! If you have ideas, suggestions, or bug reports, please feel free to submit a Pull Request or create an Issue.
