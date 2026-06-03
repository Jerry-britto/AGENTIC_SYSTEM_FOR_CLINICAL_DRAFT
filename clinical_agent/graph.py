from langgraph.graph import StateGraph, END
from clinical_agent.state import AgentState
from clinical_agent.nodes import (
    planner_node,
    reader_node,
    reconciler_node,
    safety_verifier_node,
    synthesizer_node
)

def route_based_on_action(state: AgentState) -> str:
    """
    Router function to determine the next node based on the Planner's decision.
    """
    if state["is_finished"] or state["iteration_count"] > 8:
        return "synthesizer"
        
    next_action = state["extracted_data"].get("_next_action", "READ_DOCUMENTS")
    
    if next_action == "READ_DOCUMENTS":
        return "reader"
    elif next_action == "RECONCILE_MEDICATIONS":
        return "reconciler"
    elif next_action == "VERIFY_SAFETY":
        return "verifier"
    elif next_action == "SYNTHESIZE_DRAFT":
        return "synthesizer"
        
    return "reader"

def compile_clinical_agent_graph():
    """
    Assembles and compiles the clinical agent LangGraph state machine.
    """
    # 1. Initialize Graph with state definition
    workflow = StateGraph(AgentState)

    # 2. Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("reader", reader_node)
    workflow.add_node("reconciler", reconciler_node)
    workflow.add_node("verifier", safety_verifier_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 3. Set entrypoint
    workflow.set_entry_point("planner")

    # 4. Add conditional routing from the planner
    workflow.add_conditional_edges(
        "planner",
        route_based_on_action,
        {
            "reader": "reader",
            "reconciler": "reconciler",
            "verifier": "verifier",
            "synthesizer": "synthesizer"
        }
    )

    # 5. Connect nodes back to the planner to complete the loop
    workflow.add_edge("reader", "planner")
    workflow.add_edge("reconciler", "planner")
    workflow.add_edge("verifier", "planner")
    
    # 6. Synthesizer node is terminal
    workflow.add_edge("synthesizer", END)

    # 7. Compile workflow
    return workflow.compile()
