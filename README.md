# rag-orchestration-pipeline
A lightweight RAG pipeline with explicit orchestration, validation, and retry logic, designed to explore retrieval strategies, chunking, and system reliability in real-world workflows.
--
NEXT VERSION (v0.4.0) — AGENT ORCHESTRATION LAYER

Overview
The system evolves from an adaptive RAG decision engine into a task-oriented execution system. The existing orchestrator is preserved and re-used as a tool within a higher-level agent framework. The system transitions from single-query resolution to multi-step goal execution.

Core Shift
- From: query → answer
- To: goal → plan → execute → synthesize

Architecture Changes
- Introduce an agent layer above the current orchestrator
- Treat the existing RAG system as a callable tool
- Add coordination logic for multi-step workflows
- Maintain deterministic control and observability

New Components

agents/
- planner_agent.py
  Generates structured step sequences from high-level goals

- synthesizer_agent.py
  Aggregates outputs from multiple steps into a final response

control/
- agent_orchestrator.py
  Executes planned steps sequentially and manages tool usage

tools/
- rag_tool.py
  Wrapper around existing orchestrator to expose it as a reusable tool

memory/
- session_memory.py (initial version optional)
  Stores intermediate step results and context across execution

Execution Flow
1. Receive high-level user goal
2. Planner agent decomposes goal into ordered steps
3. Agent orchestrator iterates over steps
4. Each step invokes the RAG tool (existing system)
5. Results are accumulated in memory
6. Synthesizer agent produces final structured output

Design Principles
- Preserve existing retrieval and decision logic without modification
- Separate execution (RAG) from coordination (agents)
- Maintain explicit control flow (no implicit agent loops)
- Keep system debuggable and observable at each step
- Avoid premature complexity in memory and agent interaction

Initial Use Case
- Comparative and multi-aspect queries
  Examples:
  - comparison of technologies
  - multi-factor analysis
  - structured research questions

Limitations (Initial State)
- No persistent or long-term memory
- Planner quality dependent on prompt design
- No cost-aware orchestration
- No parallel execution of steps
- No formal evaluation framework for multi-step correctness

Next Steps
- Improve planner reliability (structured outputs)
- Introduce lightweight memory management
- Add cost-aware execution strategies
- Implement step-level evaluation and logging
- Explore parallelization for independent steps
- Extend toolset beyond RAG (APIs, computation, data sources)