"""Review workflow (dossier §9.2), orchestrated with LangGraph after the deterministic analysis.

    load ─► research? ─► quality_gate ─┬─► human_review (interrupt) ─┐
                                       ├─────────────────────────────┴─► report ─► persist
                                       └─ (rejected analysis) ───────────► report

- Routing and gate rules are plain, tested Python: no LLM decides anything here.
- research (optional) asks the RAG service for a sourced explanation; if the LLM is
  unavailable the workflow continues in degraded mode with a warning.
- human_review pauses the graph (LangGraph interrupt). The state is checkpointed in
  SQLite, so a review can be submitted later, even after an API restart.
- The graph carries ids and summaries; images stay in the artifact storage (§9.3).
"""

import operator
import secrets
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from pixelmind import __version__
from pixelmind.agents.report import render_report
from pixelmind.rag.generator import GenerationError
from pixelmind.services.store import AnalysisStore

DEFAULT_QUESTION = (
    "Explique ce résultat d'analyse : que signifient ces chiffres, quelle est leur "
    "fiabilité compte tenu du modèle et de la qualité des données, et quelles limites "
    "faut-il garder en tête ?"
)
REVIEW_DECISIONS = ("approved", "corrected", "rejected")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _step(node: str, **details) -> list[dict]:
    return [{"node": node, "at": _now(), **details}]


class WorkflowState(TypedDict, total=False):
    workflow_id: str
    analysis_id: str
    requested_by: str
    explain: bool
    question: str | None
    analysis: dict
    research: dict | None
    gate: dict
    review: dict | None
    report_uri: str | None
    status: str
    warnings: Annotated[list[str], operator.add]
    trace: Annotated[list[dict], operator.add]


def quality_gate_rules(analysis: dict) -> dict:
    """PASS / human review / rejected (dossier §5.3, §9.4)."""
    if analysis["status"] == "rejected":
        return {"decision": "rejected", "reasons": analysis.get("warnings", [])}
    reasons = [f"qualité des données {analysis['quality_status']}: {i['message']}"
               for i in analysis.get("quality_issues", []) if i["status"] != "PASS"]  # fmt: skip
    if analysis.get("human_review_required") and not reasons:
        reasons.append("résultat hors de la distribution d'entraînement")
    if analysis["quality_status"] != "PASS" or analysis.get("human_review_required"):
        return {"decision": "review", "reasons": reasons}
    return {"decision": "pass", "reasons": []}


class WorkflowStore:
    """Queryable index of workflows (the full state lives in the LangGraph checkpoint)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as con, con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS workflows (workflow_id TEXT PRIMARY KEY, "
                "analysis_id TEXT, created_at TEXT, updated_at TEXT, status TEXT, "
                "review_decision TEXT)"
            )
            try:  # databases created before Sprint 7
                con.execute("ALTER TABLE workflows ADD COLUMN review_decision TEXT")
            except sqlite3.OperationalError:
                pass

    def upsert(self, workflow_id: str, analysis_id: str, status: str,
               review_decision: str | None = None) -> None:  # fmt: skip
        now = _now()
        with closing(sqlite3.connect(self.db_path)) as con, con:
            con.execute(
                "INSERT INTO workflows VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(workflow_id) "
                "DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at, "
                "review_decision = COALESCE(excluded.review_decision, review_decision)",
                (workflow_id, analysis_id, now, now, status, review_decision),
            )

    def list(self, analysis_id: str | None = None, limit: int = 50) -> list[dict]:
        q = ("SELECT workflow_id, analysis_id, created_at, updated_at, status, review_decision "
             "FROM workflows")  # fmt: skip
        args: tuple = ()
        if analysis_id:
            q, args = q + " WHERE analysis_id = ?", (analysis_id,)
        with closing(sqlite3.connect(self.db_path)) as con:
            rows = con.execute(q + " ORDER BY created_at DESC LIMIT ?", (*args, limit)).fetchall()
        keys = ("workflow_id", "analysis_id", "created_at", "updated_at", "status",
                "review_decision")  # fmt: skip
        return [dict(zip(keys, r, strict=True)) for r in rows]


class ReviewWorkflow:
    def __init__(
        self, store: AnalysisStore, analyses_dir: Path, qa_factory: Callable[[], object],
        db_path: Path,
    ) -> None:  # fmt: skip
        self.store, self.analyses_dir, self.qa_factory = store, analyses_dir, qa_factory
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self.index = WorkflowStore(db_path)
        self.graph = self._build().compile(checkpointer=SqliteSaver(self._conn))

    # --- nodes -------------------------------------------------------------

    def _load(self, state: WorkflowState) -> dict:
        analysis = self.store.get(state["analysis_id"])
        if analysis is None:
            raise KeyError(f"analysis {state['analysis_id']} not found")
        return {"analysis": analysis, "trace": _step("load", status=analysis["status"])}

    def _research(self, state: WorkflowState) -> dict:
        question = state.get("question") or DEFAULT_QUESTION
        try:
            r = self.qa_factory().ask(
                question,
                analysis=state["analysis"],
                asked_by=state.get("requested_by", "anonymous"),
            )
        except (GenerationError, FileNotFoundError) as exc:
            msg = f"explication documentaire indisponible : {exc}"
            return {"research": None, "warnings": [msg], "trace": _step("research", ok=False)}
        research = {k: r[k] for k in ("question_id", "question", "answer", "abstained",
                                      "citations")}  # fmt: skip
        research["model"] = r["generation"]["model"] or r["generation"]["provider"]
        return {"research": research,
                "trace": _step("research", ok=True, question_id=r["question_id"])}  # fmt: skip

    def _gate(self, state: WorkflowState) -> dict:
        gate = quality_gate_rules(state["analysis"])
        return {"gate": gate, "trace": _step("quality_gate", decision=gate["decision"])}

    def _human_review(self, state: WorkflowState) -> dict:
        decision = interrupt({
            "analysis_id": state["analysis_id"],
            "reasons": state["gate"]["reasons"],
            "allowed_decisions": list(REVIEW_DECISIONS),
        })  # fmt: skip
        review = {**decision, "reviewed_at": _now()}
        return {"review": review, "trace": _step("human_review", decision=decision["decision"])}

    def _report(self, state: WorkflowState) -> dict:
        rejected = state["gate"]["decision"] == "rejected" or (
            (state.get("review") or {}).get("decision") == "rejected"
        )
        status = "rejected" if rejected else "completed"
        out_dir = self.analyses_dir / state["analysis_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        steps = [*state.get("trace", []), *_step("report")]
        md = render_report({**state, "status": status, "trace": steps}, __version__)
        (out_dir / "report.md").write_text(md)
        uri = f"/api/v1/analyses/{state['analysis_id']}/artifacts/report.md"
        return {"status": status, "report_uri": uri, "trace": _step("report")}

    def _persist(self, state: WorkflowState) -> dict:
        decision = (state.get("review") or {}).get("decision")
        self.index.upsert(state["workflow_id"], state["analysis_id"], state["status"], decision)
        return {"trace": _step("persist")}

    # --- graph -------------------------------------------------------------

    def _build(self) -> StateGraph:
        g = StateGraph(WorkflowState)
        g.add_node("load", self._load)
        g.add_node("research", self._research)
        g.add_node("quality_gate", self._gate)
        g.add_node("human_review", self._human_review)
        g.add_node("report", self._report)
        g.add_node("persist", self._persist)

        def after_load(s: WorkflowState) -> Literal["research", "quality_gate"]:
            explain = s.get("explain") and s["analysis"]["status"] != "rejected"
            return "research" if explain else "quality_gate"

        def after_gate(s: WorkflowState) -> Literal["human_review", "report"]:
            return "human_review" if s["gate"]["decision"] == "review" else "report"

        g.add_edge(START, "load")
        g.add_conditional_edges("load", after_load)
        g.add_edge("research", "quality_gate")
        g.add_conditional_edges("quality_gate", after_gate)
        g.add_edge("human_review", "report")
        g.add_edge("report", "persist")
        g.add_edge("persist", END)
        return g

    # --- public API --------------------------------------------------------

    @staticmethod
    def _config(workflow_id: str) -> dict:
        return {"configurable": {"thread_id": workflow_id}}

    def start(self, analysis_id: str, explain: bool = False, question: str | None = None,
              requested_by: str = "anonymous") -> dict:  # fmt: skip
        wid = f"wf_{datetime.now(UTC):%Y%m%d}_{secrets.token_hex(4)}"
        self.index.upsert(wid, analysis_id, "running")
        init = {"workflow_id": wid, "analysis_id": analysis_id, "explain": explain,
                "question": question, "requested_by": requested_by, "warnings": [],
                "trace": []}  # fmt: skip
        self.graph.invoke(init, self._config(wid))
        return self.get(wid)

    def review(self, workflow_id: str, decision: dict) -> dict:
        view = self.get(workflow_id)
        if view is None:
            raise KeyError(workflow_id)
        if view["status"] != "awaiting_review":
            raise ValueError(f"workflow is {view['status']}, not awaiting review")
        if decision.get("decision") not in REVIEW_DECISIONS:
            raise ValueError(f"decision must be one of {REVIEW_DECISIONS}")
        self.graph.invoke(Command(resume=decision), self._config(workflow_id))
        return self.get(workflow_id)

    def get(self, workflow_id: str) -> dict | None:
        snap = self.graph.get_state(self._config(workflow_id))
        if not snap.values:
            return None
        v = snap.values
        pending = [i.value for t in snap.tasks for i in t.interrupts]
        status = "awaiting_review" if pending else v.get("status", "running")
        if pending:
            self.index.upsert(workflow_id, v["analysis_id"], status)
        return {
            "workflow_id": workflow_id,
            "analysis_id": v["analysis_id"],
            "requested_by": v.get("requested_by", "anonymous"),
            "status": status,
            "pending_review": pending[0] if pending else None,
            "gate": v.get("gate"),
            "review": v.get("review"),
            "research": v.get("research"),
            "report_uri": v.get("report_uri"),
            "warnings": v.get("warnings", []),
            "trace": v.get("trace", []),
        }

    def close(self) -> None:
        self._conn.close()
