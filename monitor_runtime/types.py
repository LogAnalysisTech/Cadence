from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence


ReviewStatus = Literal["pass", "no_issue", "insufficient_evidence", "warning", "fail", "intervene"]
OverallRecommendation = Literal["continue", "intervene", "request_human_review", "stop"]
ReviewSeverity = Literal["hard_fail", "warning", "soft_signal", "none"]
EnforcementVerdict = Literal["pass", "advise", "block", "revise"]
DecisionDisposition = Literal["pass", "advise", "block"]
SubmissionIssue = Literal[
    "none", "empty_patch", "patch_not_applicable"
]
BlockingEvidenceKind = Literal[
    "target_behavior_failure",
    "observed_contract_violation",
    "repeated_advisory",
    "deterministic_submission",
    "none",
]
ProposedStepContractStatus = Literal["compliant", "violates", "uncertain"]
ReviewContractConsistency = Literal["consistent", "contradictory", "uncertain"]


@dataclass(frozen=True)
class AgentStep:
    """A proposed or completed executable step produced by the coding agent."""

    step_id: str
    phase: str | None
    content: str = ""
    action: Any | None = None
    observation: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MonitorEvent:
    """A non-agent event inserted by the realtime monitor."""

    event_id: str
    message: str
    event_type: str = "monitor_intervention"
    phase: str = "MONITOR_INTERVENTION"
    intervention_for_phase: str | None = None
    discarded_proposed_step_id: str | None = None
    selected_capabilities: tuple[str, ...] = ()
    verdict: str = "intervene"
    metadata: dict[str, Any] = field(default_factory=dict)


TrajectoryEvent = AgentStep | MonitorEvent


@dataclass(frozen=True)
class PhaseSegment:
    """The most recent completed run of agent-authored steps with the same phase."""

    phase: str
    steps: tuple[AgentStep, ...]
    previous_phase: str | None
    next_phase: str
    proposed_step: AgentStep


@dataclass(frozen=True)
class CapabilitySelection:
    """Small set of playbook capabilities selected for one phase review."""

    capability_ids: tuple[str, ...]
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenFinding:
    """Unresolved blocking issue carried into later reviews until evidence closes it."""

    finding_id: str
    phase: str
    failure_modes: tuple[str, ...] = ()
    blocking_conditions: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    guidance: tuple[str, ...] = ()
    capability_ids: tuple[str, ...] = ()
    evidence_anchors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetiredFinding:
    """Non-enforcing record used to suppress a retired root cause from reopening."""

    finding_id: str
    phase: str
    failure_modes: tuple[str, ...] = ()
    capability_ids: tuple[str, ...] = ()
    evidence_anchors: tuple[str, ...] = ()
    retired_reason: str = ""
    initial_block_step_id: str | None = None
    follow_up_step_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskContractClaim:
    """Evidence-backed task requirement extracted verbatim from the user request."""

    claim_id: str
    statement: str
    evidence: tuple[str, ...] = ()
    acceptance_signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskContract:
    """Immutable acceptance contract established by the first supported review."""

    claims: tuple[TaskContractClaim, ...]
    established_by_review_id: str


@dataclass(frozen=True)
class ReviewLedgerEntry:
    """Compact prior-review record supplied to later reviewers for consistency."""

    review_id: str
    phase: str
    conclusion: str
    status: ReviewStatus
    verdict: EnforcementVerdict
    severity: ReviewSeverity
    proposed_step_contract_status: ProposedStepContractStatus = "uncertain"
    review_contract_consistency: ReviewContractConsistency = "uncertain"
    conflicting_contract_claim_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    guidance: tuple[str, ...] = ()
    resolved_finding_ids: tuple[str, ...] = ()
    unresolved_finding_ids: tuple[str, ...] = ()
    outcome: str = "observed"


@dataclass(frozen=True)
class PhaseReview:
    """Structured reviewer judgment and any enforceable remediation requirements."""

    status: ReviewStatus
    verdict: EnforcementVerdict = "pass"
    creates_new_finding: bool | None = None
    overall_recommendation: OverallRecommendation = "continue"
    severity: ReviewSeverity = "none"
    failure_modes: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    general_guidance: tuple[str, ...] = ()
    specific_guidance: tuple[str, ...] = ()
    next_step: str = ""
    blocking_conditions: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    resolved_finding_ids: tuple[str, ...] = ()
    unresolved_finding_ids: tuple[str, ...] = ()
    conclusion: str = ""
    proposed_contract_claims: tuple[TaskContractClaim, ...] = ()
    proposed_step_contract_status: ProposedStepContractStatus = "uncertain"
    review_contract_consistency: ReviewContractConsistency = "uncertain"
    conflicting_contract_claim_ids: tuple[str, ...] = ()
    superseded_review_ids: tuple[str, ...] = ()
    repeated_advisory_review_ids: tuple[str, ...] = ()
    verification_evidence_assessment: dict[str, Any] = field(default_factory=dict)
    related_finding_id: str | None = None
    evidence_anchors: tuple[str, ...] = ()
    blocking_evidence_kind: BlockingEvidenceKind = "none"
    blocking_evidence_step_ids: tuple[str, ...] = ()
    submission_issue: SubmissionIssue = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class MonitorDecision:
    """Result of checking a proposed step before executing its action."""

    execute_proposed_step: bool
    disposition: DecisionDisposition = "pass"
    checked_phase: str | None = None
    monitor_event: MonitorEvent | None = None
    advisory_event: MonitorEvent | None = None
    retire_finding_ids_after_advice: tuple[str, ...] = ()
    normalization_reason: str | None = None
    selected_capabilities: CapabilitySelection | None = None
    review: PhaseReview | None = None
    timing: dict[str, float | int] = field(default_factory=dict)
    memory_snapshot: dict[str, Any] = field(default_factory=dict)
    phase_policy: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class CapabilitySelector(Protocol):
    def select_capabilities(self, segment: PhaseSegment) -> CapabilitySelection:
        """Select capability ids required by the completed phase segment."""


class PhaseReviewer(Protocol):
    def review_phase(
        self,
        segment: PhaseSegment,
        capability_entries: Sequence[str],
        *,
        open_findings: Sequence[OpenFinding] = (),
        retired_findings: Sequence[RetiredFinding] = (),
        task_contract: TaskContract | None = None,
        review_ledger: Sequence[ReviewLedgerEntry] = (),
        finding_lifecycle: dict[str, bool | int] | None = None,
        focus_window: Any | None = None,
        prior_advise_guidance: Sequence[str] = (),
        prior_block_guidance: Sequence[str] = (),
        is_submission_review: bool = False,
    ) -> PhaseReview:
        """Review a completed phase segment with selected playbook entries.

        `prior_advise_guidance` and `prior_block_guidance` are the run's
        non-repetition history (raw guidance text already delivered under
        each verdict category, most-recent-last). `is_submission_review`
        marks the run's one submission-boundary semantic review, which is
        exempt from the advise non-repetition check (see
        REVIEWER_PROTOCOL_REDESIGN.md section 3.4's fourth revision).
        """


@dataclass(frozen=True)
class BeforeActionResult:
    """Shared outcome of a Cadence or baseline `before_action` call.

    `execute` tells the adapter whether to let the proposed action reach the
    environment. `message` is guidance text to surface to the agent -- either
    immediately (when `is_blocking` is True, the proposal is discarded and the
    message must be injected before requerying) or after the action executes
    and its observation is known (when `is_blocking` is False, the adapter is
    expected to stash `message` and only actually inject it once the
    observation is available -- see `after_action`/adapter docstrings for
    which side owns that hand-off).
    """

    execute: bool
    message: str | None = None
    is_blocking: bool = False
    classified_phase: str | None = None
    rule_ids: tuple[str, ...] = ()
    # When True, the adapter must force the environment to submit the
    # current diff itself (deterministically, not by asking the agent to)
    # instead of the normal discard-and-regenerate handling for a block.
    # Only ever set by monitors that decide a run is unsalvageable and want
    # to cut losses (see CadenceMonitor's give_up verdict).
    force_submit: bool = False
    judged_verdict: str | None = None
    execution_mode: str | None = None
    execution_override: str | None = None
