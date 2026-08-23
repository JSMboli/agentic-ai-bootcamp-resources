"""LangGraph examples for realistic fulfilment-centre agent workflows.

This module provides two distinct architectural approaches for building AI
agents:
1. Deterministic State Graph Workflow (`build_workflow_agent`): A hardcoded graph
   where individual nodes handle specialized tasks (risk assessment, retrieval,
   calculation, drafting) sequentially, featuring a Human-in-the-Loop review gate.
2. Dynamic Tool-Calling Agent (`build_native_tool_agent`): An autonomous, model-driven
   loop where the LLM dynamically decides which tools to invoke based on user prompts.

It also provides a simple linear script version (`run_simple_operations_agent`)
for benchmarking without graph framework overhead.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, TypedDict


# IMPORTS & COMPATIBILITY LAYER

# Attempt relative imports first (for package execution like `python -m package.module`).
# Fall back to absolute imports if the script is executed directly (e.g., `python file.py`).
try:
    from .providers import ProviderRouter, make_messages
    from .tools import (
        knowledge_lookup,
        pick_rate,
        retrieve_observations,
        risk_assessment,
    )
except ImportError:
    from providers import ProviderRouter, make_messages
    from tools import (
        knowledge_lookup,
        pick_rate,
        retrieve_observations,
        risk_assessment,
    )


# STATE DEFINITION

class OperationsState(TypedDict, total=False):
    """Represents the shared dictionary state passed between nodes in the workflow graph.

    Using `total=False` allows keys to be added or omitted dynamically as the
    state travels through sequential nodes in the LangGraph pipeline without
    triggering strict TypeDict missing-key errors.

    Attributes:
        user_request (str): The raw input query or instruction provided by the user.
        risk (Dict[str, str]): Output from the risk assessment tool (e.g., {"risk": "high"}).
        evidence (Any): Knowledge base search results and site visit observations.
        metric (Any): Calculated metrics (e.g., pick rate computations).
        draft (str): Preliminary LLM-generated response before review.
        llm_metadata (Dict[str, Any]): Telemetry and token usage from the LLM provider.
        review_decision (Any): Human reviewer feedback/decision (approve, edit, or reject).
        final (str): The final verified response returned to the end-user.
    """

    user_request: str
    risk: Dict[str, str]
    evidence: Any
    metric: Any
    draft: str
    llm_metadata: Dict[str, Any]
    review_decision: Any
    final: str


# HELPER FUNCTIONS

def _progress(enabled: bool, step: str, message: str) -> None:
    """Prints progress tracking messages to the console if progress logging is enabled.

    Args:
        enabled (bool): Flag indicating whether logging output should be printed.
        step (str): Current step label or number (e.g., '1/6' or 'tool').
        message (str): Explanatory text describing the step being executed.
    """
    if enabled:
        # flush=True ensures immediate printing to stdout without buffering delay
        print(f"[{step}] {message}", flush=True)


# ==============================================================================
# SIMPLE LINEAR AGENT PIPELINE
# ==============================================================================
def run_simple_operations_agent(
    user_request: str, provider: str | None = None
) -> Dict[str, Any]:
    """Runs a straightforward, non-graph, step-by-step operations agent pipeline.

    This function demonstrates the baseline logic without using LangGraph framework abstractions:
    1. Assess request safety/risk using the risk assessment tool.
    2. Retrieve relevant evidence (knowledge articles & observations).
    3. Run deterministic calculations (e.g., pick rate) if mentioned in request.
    4. Construct a prompt and prompt an LLM provider.
    5. Prepend a safety warning if the request was flagged as high-risk.

    Args:
        user_request (str): The operational question or task description.
        provider (str | None, optional): Specific LLM provider identifier (e.g., "groq").
            Defaults to None (uses default provider from ProviderRouter).

    Returns:
        Dict[str, Any]: A dictionary containing the input request, intermediate tool outputs,
            draft text, model metadata, and final response.
    """
    # Initialize provider router to handle LLM communication and fallback logic
    router = ProviderRouter(verbose=True)

    # Step 1: Run deterministic safety/risk evaluation on the user's prompt
    risk = risk_assessment(user_request)

    # Step 2: Gather grounding evidence from knowledge lookup and site observations
    evidence = {
        "knowledge": knowledge_lookup(user_request, top_k=3),
        # Retrieve observations only if fulfillment/Amazon keywords are detected
        "observations": retrieve_observations("all")[:3]
        if any(k in user_request.lower() for k in ["amazon", "fulfilment"])
        else [],
    }

    # Step 3: Run domain math calculation if keyword "pick rate" is present
    metric = pick_rate(735, 6.5) if "pick rate" in user_request.lower() else None

    # Step 4: Construct grounded prompt containing gathered context and guardrails
    prompt = (
        f"Request: {user_request}\n"
        f"Risk: {risk}\n"
        f"Evidence: {evidence}\n"
        f"Metric: {metric}\n\n"
        "Produce a grounded answer for an adult learner. "
        "State limitations and never turn a productivity metric into an employment decision."
    )

    # Step 5: Generate response via LLM
    response = router.chat(make_messages(prompt), provider=provider)

    # Step 6: Apply simple safety policy gate for high-risk flags
    final = (
        response.text
        if risk.get("risk") != "high"
        else "Human review required before any consequential action.\n\n"
        + response.text
    )

    # Step 7: Return all state components
    return {
        "user_request": user_request,
        "risk": risk,
        "evidence": evidence,
        "metric": metric,
        "draft": response.text,
        "llm_metadata": response.summary(),
        "final": final,
    }


# DETERMINISTIC WORKFLOW AGENT (LANGGRAPH + HUMAN INTERRUPT)

def build_workflow_agent(provider: str = "groq", progress: bool = True):
    """Builds and compiles a stateful, step-by-step operational workflow using LangGraph.

    This workflow hardcodes the control flow through specialized nodes:
    1. Risk Assessment -> 2. Evidence Retrieval -> 3. Metric Calculation ->
    4. Draft Synthesis -> 5. Human Review Gate (Interrupt) -> 6. Finalization.

    Why this architecture?
    - High auditability and predictability: Tool execution is hardcoded rather than left to
      probabilistic tool choices by smaller models.
    - Safety compliance: Incorporates `interrupt()` for human approval when risk is 'high'.

    Args:
        provider (str, optional): Target LLM provider for drafting. Defaults to "groq".
        progress (bool, optional): Whether to log node progress steps. Defaults to True.

    Raises:
        RuntimeError: If `langgraph` package is not installed.

    Returns:
        CompiledStateGraph: A ready-to-invoke LangGraph application with memory checkpointer.
    """
    # Lazy import of LangGraph dependencies to keep startup fast and raise clean errors if missing
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt
    except ImportError as exc:
        raise RuntimeError(
            "Install LangGraph: `%pip install -U langgraph`"
        ) from exc

    router = ProviderRouter(verbose=True)

    # --- NODE DEFINITIONS ---

    def assess(state: OperationsState):
        """Node 1: Evaluates safety/compliance risk of the user's request."""
        _progress(progress, "1/6", "Assessing request risk…")
        return {"risk": risk_assessment(state["user_request"])}

    def retrieve(state: OperationsState):
        """Node 2: Fetches relevant articles and observations based on query topics."""
        _progress(progress, "2/6", "Retrieving grounded teaching evidence…")
        q = state["user_request"]

        # Check for context keywords to determine observation gathering scope
        observations = (
            retrieve_observations("all")[:4]
            if any(
                k in q.lower()
                for k in ["amazon", "fulfilment", "packing", "picking"]
            )
            else []
        )
        return {
            "evidence": {
                "knowledge": knowledge_lookup(q, 4),
                "observations": observations,
            }
        }

    def calculate(state: OperationsState):
        """Node 3: Executes deterministic computations where applicable."""
        _progress(progress, "3/6", "Running deterministic calculations where required…")
        return {
            "metric": (
                pick_rate(735, 6.5)
                if "pick rate" in state["user_request"].lower()
                else None
            )
        }

    def draft(state: OperationsState):
        """Node 4: Uses LLM to synthesize tool results into a coherent draft."""
        _progress(
            progress,
            "4/6",
            f"Synthesising a grounded draft with provider={provider}…",
        )
        prompt = (
            "You are an operations-support agent prototype for teaching.\n"
            f"User request: {state['user_request']}\n"
            f"Risk assessment: {state['risk']}\n"
            f"Retrieved evidence: {state['evidence']}\n"
            f"Calculated metric: {state.get('metric')}\n\n"
            "Write a concise, grounded response. Clearly distinguish observations, retrieved guidance and calculations. "
            "Do not make employment, disciplinary or safety-critical decisions. If evidence is insufficient, say so."
        )
        # Fallback disabled to catch provider-specific errors during development/testing
        response = router.chat(
            make_messages(prompt),
            provider=provider,
            fallback_on_error=False,
            max_tokens=700,
        )
        return {"draft": response.text, "llm_metadata": response.summary()}

    def review_gate(state: OperationsState):
        """Node 5: Applies policy gate; pauses graph execution via `interrupt` if high-risk."""
        _progress(progress, "5/6", "Applying the human-review policy gate…")

        # Bypass review gate if risk level is low/medium
        if state.get("risk", {}).get("risk") != "high":
            return {"review_decision": {"decision": "not_required"}}

        # Payload sent to human reviewer when execution pauses
        payload = {
            "reason": "High-risk request requires human review before release or action.",
            "request": state["user_request"],
            "risk": state["risk"],
            "draft": state.get("draft", ""),
            "provider": state.get("llm_metadata", {}).get("provider"),
            "model": state.get("llm_metadata", {}).get("model"),
            "options": ["approve", "edit", "reject"],
        }
        # Interrupt graph execution and wait for human input/resume command
        decision = interrupt(payload)
        return {"review_decision": decision}

    def finalise(state: OperationsState):
        """Node 6: Finalizes output according to the human reviewer's decision."""
        _progress(
            progress, "6/6", "Finalising output and preserving review decision…"
        )
        decision = state.get("review_decision") or {"decision": "not_required"}

        # Normalize string response into dictionary format if simple string passed
        if isinstance(decision, str):
            decision = {"decision": decision}

        action = str(decision.get("decision", "not_required")).lower()

        # Handle review outcome options
        if action == "reject":
            return {
                "final": "The proposed output was rejected by the human reviewer."
            }
        if action == "edit":
            return {
                "final": decision.get("edited_text") or state.get("draft", "")
            }

        return {"final": state.get("draft", "")}

    # --- GRAPH ASSEMBLY ---
    builder = StateGraph(OperationsState)

    # Add nodes to graph
    builder.add_node("assess_risk", assess)
    builder.add_node("retrieve_evidence", retrieve)
    builder.add_node("calculate_metrics", calculate)
    builder.add_node("draft_response", draft)
    builder.add_node("human_review", review_gate)
    builder.add_node("finalise", finalise)

    # Define sequential execution flow
    builder.add_edge(START, "assess_risk")
    builder.add_edge("assess_risk", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "calculate_metrics")
    builder.add_edge("calculate_metrics", "draft_response")
    builder.add_edge("draft_response", "human_review")
    builder.add_edge("human_review", "finalise")
    builder.add_edge("finalise", END)

    # Compile graph with in-memory checkpointer (required for human interrupt/resume functionality)
    return builder.compile(checkpointer=InMemorySaver())


# AUTONOMOUS TOOL-CALLING AGENT (LANGGRAPH + LANGCHAIN TOOLS)

def build_native_tool_agent(provider: str = "groq", progress: bool = True):
    """Builds a dynamic tool-calling agent loop driven by the LLM's decisions.

    Unlike `build_workflow_agent`, this workflow uses a conditional loop where the LLM
    decides which tools to call, inspects tool results, and iteratively decides whether
    to call more tools or produce the final user response.

    Args:
        provider (str, optional): Provider identifier to back the LangChain model.
            Defaults to "groq".
        progress (bool, optional): Whether to display tool invocation logging.
            Defaults to True.

    Raises:
        RuntimeError: If required LangChain or LangGraph libraries are missing.

    Returns:
        CompiledStateGraph: Compiled reactive agent graph.
    """
    try:
        from langchain_core.messages import SystemMessage
        from langchain_core.tools import tool
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode
    except ImportError as exc:
        raise RuntimeError(
            "Install langgraph, langchain-core and the provider integration packages."
        ) from exc

    # Initialize provider router and retrieve underlying LangChain chat model wrapper
    router = ProviderRouter(verbose=True)
    model = router.langchain_model(
        provider=provider, temperature=0, max_tokens=650
    )

    # --- TOOL DEFINITIONS ---
    # Python functions decorated with `@tool` convert docstrings into schema parameters
    # that the LLM reads to decide when and how to invoke each tool.

    @tool
    def lookup_guidance(query: str) -> str:
        """Search the bootcamp knowledge base for grounded agentic-AI guidance."""
        _progress(progress, "tool", f"lookup_guidance(query={query!r})")
        return str(knowledge_lookup(query, top_k=3))

    @tool
    def lookup_visit_observations(area: str = "all") -> str:
        """Retrieve synthetic/classroom fulfilment-centre visit observations."""
        _progress(progress, "tool", f"lookup_visit_observations(area={area!r})")
        return str(retrieve_observations(area)[:5])

    @tool
    def calculate_pick_rate(items_picked: float, hours: float) -> str:
        """Calculate items picked per hour with an interpretation warning."""
        _progress(
            progress,
            "tool",
            f"calculate_pick_rate(items={items_picked}, hours={hours})",
        )
        return str(pick_rate(items_picked, hours))

    @tool
    def assess_request_risk(request: str) -> str:
        """Assess request risk and whether human review is recommended."""
        _progress(progress, "tool", "assess_request_risk(…)")
        return str(risk_assessment(request))

    # Collect tool definitions and bind them to model capabilities
    tools = [
        lookup_guidance,
        lookup_visit_observations,
        calculate_pick_rate,
        assess_request_risk,
    ]
    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools)

    # Core system instructions defining agent boundaries and ethical behavior
    system = SystemMessage(
        content=(
            "You are a bounded fulfilment-centre support-agent prototype for adult learners. "
            "Use tools rather than inventing operational facts. Never use productivity metrics alone for employment decisions. "
            "For safety, personal-data, disciplinary or consequential requests, explain that human review is required."
        )
    )

    # --- GRAPH NODE & CONDITIONAL ROUTING FUNCTIONS ---

    # Deliberately omit a MessagesState annotation here. In the previous build,
    # MessagesState was imported inside this factory while postponed annotations
    # caused LangGraph's type-hint inspection to look for it in module globals,
    # producing `NameError: MessagesState is not defined` at runtime.
    def call_model(state):
        """Invokes the model with system context and conversation history."""
        _progress(
            progress,
            "model",
            f"Invoking {provider} model for tool selection/synthesis…",
        )
        response = model_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    def route(state) -> Literal["tools", "__end__"]:
        """Inspects the last message to route to tool execution or end the conversation."""
        last = state["messages"][-1]
        # If the model produced tool calls, route to the tool execution node
        return "tools" if getattr(last, "tool_calls", None) else END

    # --- GRAPH ASSEMBLY ---
    builder = StateGraph(MessagesState)

    # Add nodes
    builder.add_node("model", call_model)
    builder.add_node("tools", tool_node)

    # Define edges: START -> model -> (conditional: tools OR END) -> back to model
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route, ["tools", END])
    builder.add_edge("tools", "model")

    return builder.compile()


# APPLICATION ENTRYPOINT FACTORY

def build_langgraph_operations_app(provider: str | None = None):
    """Convenience factory function to instantiate the primary LangGraph operations agent.

    Args:
        provider (str | None, optional): Target LLM provider name. Defaults to "groq".

    Returns:
        CompiledStateGraph: Instantiated workflow agent ready to run.
    """
    return build_workflow_agent(provider=provider or "groq")