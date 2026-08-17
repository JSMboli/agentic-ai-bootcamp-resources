"""LangGraph examples for realistic fulfilment-centre agent workflows."""
from __future__ import annotations

from typing import Any, Dict, Literal, TypedDict

try:
    from .providers import ProviderRouter, make_messages
    from .tools import knowledge_lookup, pick_rate, risk_assessment, retrieve_observations
except ImportError:
    from providers import ProviderRouter, make_messages
    from tools import knowledge_lookup, pick_rate, risk_assessment, retrieve_observations


class OperationsState(TypedDict, total=False):
    user_request: str
    risk: Dict[str, str]
    evidence: Any
    metric: Any
    draft: str
    llm_metadata: Dict[str, Any]
    review_decision: Any
    final: str


def _progress(enabled: bool, step: str, message: str) -> None:
    if enabled:
        print(f"[{step}] {message}", flush=True)


def run_simple_operations_agent(user_request: str, provider: str | None = None) -> Dict[str, Any]:
    router = ProviderRouter(verbose=True)
    risk = risk_assessment(user_request)
    evidence = {
        "knowledge": knowledge_lookup(user_request, top_k=3),
        "observations": retrieve_observations("all")[:3]
        if any(k in user_request.lower() for k in ["amazon", "fulfilment"])
        else [],
    }
    metric = pick_rate(735, 6.5) if "pick rate" in user_request.lower() else None
    prompt = (
        f"Request: {user_request}\nRisk: {risk}\nEvidence: {evidence}\nMetric: {metric}\n\n"
        "Produce a grounded answer for an adult learner. State limitations and never turn a productivity metric into an employment decision."
    )
    response = router.chat(make_messages(prompt), provider=provider)
    final = response.text if risk.get("risk") != "high" else "Human review required before any consequential action.\n\n" + response.text
    return {
        "user_request": user_request,
        "risk": risk,
        "evidence": evidence,
        "metric": metric,
        "draft": response.text,
        "llm_metadata": response.summary(),
        "final": final,
    }


def build_workflow_agent(provider: str = "groq", progress: bool = True):
    """Deterministic tools + LLM synthesis + LangGraph human interrupt.

    The graph deliberately mixes deterministic nodes with an LLM node. This is
    easier to audit and more reliable with small local models than asking the LLM
    to choose every tool itself.
    """
    try:
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import interrupt
    except ImportError as exc:
        raise RuntimeError("Install LangGraph: `%pip install -U langgraph`") from exc

    router = ProviderRouter(verbose=True)

    def assess(state: OperationsState):
        _progress(progress, "1/6", "Assessing request risk…")
        return {"risk": risk_assessment(state["user_request"])}

    def retrieve(state: OperationsState):
        _progress(progress, "2/6", "Retrieving grounded teaching evidence…")
        q = state["user_request"]
        observations = (
            retrieve_observations("all")[:4]
            if any(k in q.lower() for k in ["amazon", "fulfilment", "packing", "picking"])
            else []
        )
        return {"evidence": {"knowledge": knowledge_lookup(q, 4), "observations": observations}}

    def calculate(state: OperationsState):
        _progress(progress, "3/6", "Running deterministic calculations where required…")
        return {"metric": pick_rate(735, 6.5) if "pick rate" in state["user_request"].lower() else None}

    def draft(state: OperationsState):
        _progress(progress, "4/6", f"Synthesising a grounded draft with provider={provider}…")
        prompt = (
            "You are an operations-support agent prototype for teaching.\n"
            f"User request: {state['user_request']}\n"
            f"Risk assessment: {state['risk']}\n"
            f"Retrieved evidence: {state['evidence']}\n"
            f"Calculated metric: {state.get('metric')}\n\n"
            "Write a concise, grounded response. Clearly distinguish observations, retrieved guidance and calculations. "
            "Do not make employment, disciplinary or safety-critical decisions. If evidence is insufficient, say so."
        )
        response = router.chat(
            make_messages(prompt), provider=provider, fallback_on_error=False, max_tokens=700
        )
        return {"draft": response.text, "llm_metadata": response.summary()}

    def review_gate(state: OperationsState):
        _progress(progress, "5/6", "Applying the human-review policy gate…")
        if state.get("risk", {}).get("risk") != "high":
            return {"review_decision": {"decision": "not_required"}}
        payload = {
            "reason": "High-risk request requires human review before release or action.",
            "request": state["user_request"],
            "risk": state["risk"],
            "draft": state.get("draft", ""),
            "provider": state.get("llm_metadata", {}).get("provider"),
            "model": state.get("llm_metadata", {}).get("model"),
            "options": ["approve", "edit", "reject"],
        }
        decision = interrupt(payload)
        return {"review_decision": decision}

    def finalise(state: OperationsState):
        _progress(progress, "6/6", "Finalising output and preserving review decision…")
        decision = state.get("review_decision") or {"decision": "not_required"}
        if isinstance(decision, str):
            decision = {"decision": decision}
        action = str(decision.get("decision", "not_required")).lower()
        if action == "reject":
            return {"final": "The proposed output was rejected by the human reviewer."}
        if action == "edit":
            return {"final": decision.get("edited_text") or state.get("draft", "")}
        return {"final": state.get("draft", "")}

    builder = StateGraph(OperationsState)
    builder.add_node("assess_risk", assess)
    builder.add_node("retrieve_evidence", retrieve)
    builder.add_node("calculate_metrics", calculate)
    builder.add_node("draft_response", draft)
    builder.add_node("human_review", review_gate)
    builder.add_node("finalise", finalise)
    builder.add_edge(START, "assess_risk")
    builder.add_edge("assess_risk", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "calculate_metrics")
    builder.add_edge("calculate_metrics", "draft_response")
    builder.add_edge("draft_response", "human_review")
    builder.add_edge("human_review", "finalise")
    builder.add_edge("finalise", END)
    return builder.compile(checkpointer=InMemorySaver())


def build_native_tool_agent(provider: str = "groq", progress: bool = True):
    """Model-driven tool-calling loop; Groq is recommended for the live demo."""
    try:
        from langchain_core.messages import SystemMessage
        from langchain_core.tools import tool
        from langgraph.graph import MessagesState, StateGraph, START, END
        from langgraph.prebuilt import ToolNode
    except ImportError as exc:
        raise RuntimeError("Install langgraph, langchain-core and the provider integration packages.") from exc

    router = ProviderRouter(verbose=True)
    model = router.langchain_model(provider=provider, temperature=0, max_tokens=650)

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
        _progress(progress, "tool", f"calculate_pick_rate(items={items_picked}, hours={hours})")
        return str(pick_rate(items_picked, hours))

    @tool
    def assess_request_risk(request: str) -> str:
        """Assess request risk and whether human review is recommended."""
        _progress(progress, "tool", "assess_request_risk(…)")
        return str(risk_assessment(request))

    tools = [lookup_guidance, lookup_visit_observations, calculate_pick_rate, assess_request_risk]
    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools)
    system = SystemMessage(content=(
        "You are a bounded fulfilment-centre support-agent prototype for adult learners. "
        "Use tools rather than inventing operational facts. Never use productivity metrics alone for employment decisions. "
        "For safety, personal-data, disciplinary or consequential requests, explain that human review is required."
    ))

    # Deliberately omit a MessagesState annotation here. In the previous build,
    # MessagesState was imported inside this factory while postponed annotations
    # caused LangGraph's type-hint inspection to look for it in module globals,
    # producing `NameError: MessagesState is not defined` at runtime.
    def call_model(state):
        _progress(progress, "model", f"Invoking {provider} model for tool selection/synthesis…")
        response = model_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    def route(state) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route, ["tools", END])
    builder.add_edge("tools", "model")
    return builder.compile()


def build_langgraph_operations_app(provider: str | None = None):
    return build_workflow_agent(provider=provider or "groq")
