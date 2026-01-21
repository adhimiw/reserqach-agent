from __future__ import annotations

from typing import Any, Callable

from .deps import SupportDeps
from .state import SupportState
from .nodes.listen import listen_node
from .nodes.analyze import analyze_node
from .nodes.retrieve import retrieve_node
from .nodes.decide import decide_node
from .nodes.respond import respond_node
from .nodes.improve import improve_node


class _SimpleChain:
    def __init__(self, steps: list[Callable[[SupportState], dict[str, Any]]]):
        self._steps = steps

    def invoke(self, state: SupportState, config: dict[str, Any] | None = None) -> SupportState:
        s: SupportState = dict(state)
        for step in self._steps:
            patch = step(s)
            s.update(patch)
        return s

    async def ainvoke(self, state: SupportState, config: dict[str, Any] | None = None) -> SupportState:
        return self.invoke(state, config=config)


def build_support_chain(deps: SupportDeps):
    try:
        from langgraph.graph import StateGraph, END
    except Exception:  # pragma: no cover
        StateGraph = None
        END = None

    if StateGraph is None:
        return _SimpleChain(
            steps=[
                lambda s: listen_node(s, deps),
                lambda s: analyze_node(s, deps),
                lambda s: retrieve_node(s, deps),
                lambda s: decide_node(s, deps),
                lambda s: respond_node(s, deps),
                lambda s: improve_node(s, deps),
            ]
        )

    workflow = StateGraph(SupportState)

    workflow.add_node("listen", lambda s: listen_node(s, deps))
    workflow.add_node("analyze", lambda s: analyze_node(s, deps))
    workflow.add_node("retrieve", lambda s: retrieve_node(s, deps))
    workflow.add_node("decide", lambda s: decide_node(s, deps))
    workflow.add_node("respond", lambda s: respond_node(s, deps))
    workflow.add_node("improve", lambda s: improve_node(s, deps))

    workflow.set_entry_point("listen")
    workflow.add_edge("listen", "analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "decide")
    workflow.add_edge("decide", "respond")
    workflow.add_edge("respond", "improve")
    workflow.add_edge("improve", END)

    return workflow.compile()
