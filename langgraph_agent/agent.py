
import argparse
import json
import re
from pathlib import Path
from typing import TypedDict, Optional

import ollama
from langgraph.graph import StateGraph, END

from mock_tools import lookup_user_history, lookup_policy, create_ticket, send_notification

CONFIDENCE_THRESHOLD = 0.6
VALID_CATEGORIES = {"hardware", "software", "access", "network", "other"}


class RequestState(TypedDict):
    request_id: str
    issue_text: str
    user_id: str
    category: Optional[str]
    urgency: Optional[str]
    recommendation: Optional[dict]
    tools_called: list
    tool_results: Optional[dict]
    needs_human_review: bool
    decision: Optional[str]
    action_result: Optional[str]
    policy_recheck: bool
    total_tokens: int


def _extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def step1_classify(state: RequestState, model: str) -> RequestState:
    prompt = f"""Classify this request into exactly one category and one urgency level.
Valid categories: hardware, software, access, network, other
Valid urgency levels: low, medium, high

Respond with ONLY a JSON object using one value from each list, for example:
{{"category": "hardware", "urgency": "medium"}}

Request: {state['issue_text']}"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    state["total_tokens"] = state.get("total_tokens", 0) + response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
    parsed = _extract_json(response["message"]["content"]) or {}
    state["category"] = parsed.get("category", "other")
    state["urgency"] = parsed.get("urgency", "medium")
    return state


def step2_3_assessment_agent(state: RequestState, model: str) -> RequestState:
    decide_prompt = f"""You are an assessment agent. You have two tools:
- lookup_user_history: returns the requester's past request history
- lookup_policy: returns whether this category requires an extra security check

Request category: {state['category']}, urgency: {state['urgency']}
Issue: {state['issue_text']}

Which tool(s) do you need to call to assess this request? Respond with ONLY a JSON
array of tool names, e.g. ["lookup_user_history", "lookup_policy"] or [] if none needed."""

    response = ollama.chat(model=model, messages=[{"role": "user", "content": decide_prompt}])
    state["total_tokens"] = state.get("total_tokens", 0) + response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
    tools_wanted = []
    try:
        match = re.search(r"\[.*\]", response["message"]["content"], re.DOTALL)
        tools_wanted = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        pass

    tool_results = {}
    state["tools_called"] = []
    for tool_name in tools_wanted:
        if tool_name == "lookup_user_history":
            tool_results["user_history"] = lookup_user_history(state["user_id"])
            state["tools_called"].append(tool_name)
        elif tool_name == "lookup_policy":
            tool_results["policy"] = lookup_policy(state["category"])
            state["tools_called"].append(tool_name)

    state["tool_results"] = tool_results

    assess_prompt = f"""Based on this information, produce a structured recommendation.
recommended_action must be exactly one of these two values: auto_approve, escalate

Respond with ONLY a JSON object with exactly these 3 fields, for example:
{{"category": "access", "urgency": "high", "recommended_action": "escalate"}}

Category: {state['category']}, Urgency: {state['urgency']}
Tool results: {json.dumps(tool_results)}
Issue: {state['issue_text']}"""

    N_SAMPLES = 3
    samples = []
    for _ in range(N_SAMPLES):
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": assess_prompt}],
            options={"temperature": 0.7},
        )
        state["total_tokens"] = state.get("total_tokens", 0) + response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
        parsed = _extract_json(response["message"]["content"]) or {}
        samples.append({
            "category": parsed.get("category", state["category"]),
            "urgency": parsed.get("urgency", state["urgency"]),
            "recommended_action": parsed.get("recommended_action", "escalate"),
        })

    triples = [(s["category"], s["urgency"], s["recommended_action"]) for s in samples]
    majority_triple = max(set(triples), key=triples.count)
    agreement_rate = triples.count(majority_triple) / len(triples)

    majority_category, majority_urgency, majority_action = majority_triple

    state["recommendation"] = {
        "category": majority_category,
        "urgency": majority_urgency,
        "recommended_action": majority_action,
        "confidence": round(agreement_rate, 2),
    }
    return state


def step3_5_confidence_check(state: RequestState) -> RequestState:
    confidence = state["recommendation"].get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    state["recommendation"]["confidence"] = confidence
    state["needs_human_review"] = confidence < CONFIDENCE_THRESHOLD
    return state

def route_after_confidence(state: RequestState) -> str:
    return "human_review" if state["needs_human_review"] else "route_and_approve"


def step4_human_review(state: RequestState) -> RequestState:
    state["decision"] = "escalate"
    return state


def step4_route_and_approve(state: RequestState) -> RequestState:

    category = state["recommendation"].get("category", state["category"])
    if category not in VALID_CATEGORIES:
        state["decision"] = "escalate"
        return state
    policy = lookup_policy(category)
    requires_extra_check = policy.get("requires_extra_security_check", False)
    state["policy_recheck"] = requires_extra_check

    urgency = state["recommendation"].get("urgency", state["urgency"])
    if requires_extra_check:
        
        state["decision"] = "escalate"
    else:
        state["decision"] = "auto_approve" if urgency == "low" else "escalate"

    return state


def step5_action(state: RequestState) -> RequestState:
    if state["needs_human_review"]:
        state["action_result"] = send_notification(f"Request {state['request_id']} flagged for human review")
    elif state["decision"] == "auto_approve":
        state["action_result"] = send_notification(f"Request {state['request_id']} auto-approved")
    else:
        urgency = state["recommendation"].get("urgency", state["urgency"])
        state["action_result"] = create_ticket(priority=urgency, assigned_to="IT_general")
    return state


def build_graph(model: str):
    graph = StateGraph(RequestState)

    graph.add_node("classify", lambda state: step1_classify(state, model))
    graph.add_node("assess", lambda state: step2_3_assessment_agent(state, model))
    graph.add_node("confidence_check", step3_5_confidence_check)
    graph.add_node("human_review", step4_human_review)
    graph.add_node("route_and_approve", step4_route_and_approve)
    graph.add_node("action", step5_action)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "assess")
    graph.add_edge("assess", "confidence_check")
    graph.add_conditional_edges(
        "confidence_check",
        route_after_confidence,
        {"human_review": "human_review", "route_and_approve": "route_and_approve"},
    )
    graph.add_edge("human_review", "action")
    graph.add_edge("route_and_approve", "action")
    graph.add_edge("action", END)

    return graph.compile()


def run_pipeline(record: dict, model: str, compiled_graph) -> RequestState:
    initial_state: RequestState = {
        "request_id": record["request_id"],
        "issue_text": record["issue_text"],
        "user_id": record.get("user_id", "UNKNOWN"),
        "category": None, "urgency": None,
        "recommendation": None, "tools_called": [], "tool_results": None,
        "needs_human_review": False, "decision": None, "action_result": None,
        "policy_recheck": False,
        "total_tokens": 0,
    }
    return compiled_graph.invoke(initial_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    dataset_path = Path(__file__).parent.parent / "data" / "synthetic_requests.json"
    if not dataset_path.exists():
        raise SystemExit("Run data/generate_dataset.py first.")

    records = json.loads(dataset_path.read_text(encoding="utf-8"))[: args.limit]
    compiled_graph = build_graph(args.model)

    for record in records:
        result = run_pipeline(record, args.model, compiled_graph)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("-" * 60)