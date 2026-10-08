"""
Decision tree v2 for CAIS method selection.

Implements the revised tree from CAIS_assumptions.pdf (page 21): structural
nodes, criteria diamonds, hard gates, advisories, and END states. This module
only decides *which* method to run (or that the pipeline must stop). It does
not execute methods and does not touch the estimator/executor rosters.

Node verdict policy (confirmed with the team):
  passed=True   -> branch succeeds
  passed=False  -> gate: stop the pipeline; advisory: warn and continue
  passed=None   -> never a silent pass: warn and continue

Checks are injected. Tests pass stub verdicts; production uses
``build_default_checks`` which binds the real functions from
``cais.methods.pre_model_assumption_utils``. Unknown/missing checks become
"not implemented yet" placeholders (can't tell -> warn and continue).

The traversal result (``TreeResult``) records the path, every check verdict,
warnings, and any post-estimation checks to run later, so any run can be
debugged after the fact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cais.models import AssumptionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leaf identifiers. Implemented leaves use the executor names from
# cais.methods.METHOD_MAPPING verbatim, so the tree result can be executed
# directly. Unimplemented variants keep descriptive names and are flagged in
# LEAF_IMPLEMENTED; they become base-method + options once implemented.
# ---------------------------------------------------------------------------
M_OLS_PRE_TREATMENT = "linear_regression"
M_DIFF_IN_MEANS = "diff_in_means"
M_IV = "instrumental_variable"
M_DID = "difference_in_differences"
M_DID_TIME_WINDOW = "difference_in_differences_time_window"
M_RDD = "regression_discontinuity_design"
M_DONUT_RDD = "donut_rdd"
M_FUZZY_RDD = "fuzzy_rdd"  # reserved: draft variant of RDD, not traversed yet
M_IPW = "propensity_score_weighting"
M_PSM = "propensity_score_matching"
M_TRIMMED_IPW = "trimmed_ipw"
M_FRONTDOOR = "frontdoor_adjustment"
M_GPS = "generalized_propensity_score"

# Leaves reachable by the traversal with their implementation status.
LEAF_IMPLEMENTED: Dict[str, bool] = {
    M_OLS_PRE_TREATMENT: True,
    M_DIFF_IN_MEANS: True,
    M_IV: True,
    M_DID: True,
    M_DID_TIME_WINDOW: False,
    M_RDD: True,
    M_DONUT_RDD: False,
    M_FUZZY_RDD: False,
    M_IPW: True,
    M_PSM: True,
    M_TRIMMED_IPW: False,
    M_FRONTDOOR: False,  # left unimplemented for now (team decision)
    M_GPS: True,
}

# Every check the traversal can ask for. Unknown entries are placeholders.
ALL_CHECK_KEYS: tuple = (
    "sutva",
    "stable_group_composition",
    "baseline_outcome_balance",
    "parallel_trends",
    "no_anticipation",
    "iv_relevance",
    "iv_exclusion",
    "iv_exogeneity",
    "iv_monotonicity",
    "cond_ignorability",
    "positivity",
    "rdd_no_manipulation",
    "rdd_covariate_continuity",
    "rdd_continuity_potential_outcomes",
    "frontdoor_full_mediation",
    "frontdoor_no_tm_confounding",
    "frontdoor_t_blocks_my",
    "frontdoor_positivity",
    "gps_cond_ignorability",
    "gps_positivity",
)

DESCRIPTION_PROMPT = (
    "Please provide a brief description of the dataset (columns, context, background). "
    "Type 'skip' to continue without one."
)
SUTVA_PROMPT = (
    "The SUTVA assessment is inconclusive. Please add any information about interference "
    "between units or variation in how the treatment was administered. Type 'skip' to continue."
)

MAX_PROMPTS = 3


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class TreeStep:
    """One recorded decision along the traversal."""

    node: str
    kind: str  # structural | criteria | gate | advisory | leaf | process | end
    outcome: str  # Yes/No/Pass/Fail/pass/fail/can't tell/selected/...
    detail: str = ""


@dataclass
class TreeResult:
    """Everything the caller needs to execute (or explain why not)."""

    method: Optional[str] = None
    variant: Optional[str] = None
    implemented: bool = True
    ended: bool = False
    end_reason: Optional[str] = None
    steps: List[TreeStep] = field(default_factory=list)
    assumptions: Dict[str, AssumptionResult] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    planned_post_checks: List[str] = field(default_factory=list)
    description: Optional[str] = None

    @property
    def path(self) -> str:
        return " -> ".join(f"{s.node}[{s.outcome}]" for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "variant": self.variant,
            "implemented": self.implemented,
            "ended": self.ended,
            "end_reason": self.end_reason,
            "warnings": list(self.warnings),
            "planned_post_checks": list(self.planned_post_checks),
            "description": self.description,
            "path": self.path,
            "steps": [
                {"node": s.node, "kind": s.kind, "outcome": s.outcome, "detail": s.detail}
                for s in self.steps
            ],
            "assumptions": {k: v.model_dump() for k, v in self.assumptions.items()},
        }


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

class _Traversal:
    def __init__(
        self,
        properties: Dict[str, Any],
        checks: Optional[Dict[str, Callable[[], AssumptionResult]]] = None,
        checks_factory: Optional[Callable[[Optional[str]], Dict[str, Callable[[], AssumptionResult]]]] = None,
        description: Optional[str] = None,
        prompt_callback: Optional[Callable[[str], Optional[str]]] = None,
        df: Any = None,
        llm: Any = None,
    ):
        self.props = dict(properties or {})
        self.df = df
        self.llm = llm
        self.prompt_callback = prompt_callback
        self.description = description or self.props.get("dataset_description")

        if checks is not None:
            self.factory = lambda _desc: checks
        elif checks_factory is not None:
            self.factory = checks_factory
        else:
            self.factory = lambda desc: build_default_checks(self.props, self.df, desc, self.llm)

        self.checks = self.factory(self.description)
        self.result = TreeResult(description=self.description)

    # -- recording helpers ------------------------------------------------

    def step(self, node: str, kind: str, outcome: str, detail: str = "") -> None:
        self.result.steps.append(TreeStep(node, kind, outcome, detail))
        logger.info("[tree v2] %-55s %-10s %s", node, outcome, detail)

    def end(self, reason: str) -> TreeResult:
        self.result.ended = True
        self.result.method = None
        self.result.end_reason = reason
        self.step("END", "end", "stop", reason)
        return self.result

    def leaf(self, method: str, variant: Optional[str] = None) -> TreeResult:
        implemented = LEAF_IMPLEMENTED.get(method, True)
        self.result.method = method
        self.result.variant = variant
        self.result.implemented = implemented
        self.step(
            method,
            "leaf",
            "selected" if implemented else "selected (not implemented)",
            detail=f"variant: {variant}" if variant else "",
        )
        if not implemented:
            self.result.warnings.append(f"{method} is not implemented yet; this study cannot be estimated.")
        return self.result

    def _safe_check(self, check_name: str) -> AssumptionResult:
        fn = self.checks.get(check_name)
        if fn is None:
            return AssumptionResult(
                passed=None, reasoning=f"Check '{check_name}' is not implemented yet."
            )
        try:
            res = fn()
        except Exception as exc:  # checks must never crash the traversal
            logger.warning("Check '%s' raised: %s", check_name, exc)
            return AssumptionResult(passed=None, reasoning=f"Check '{check_name}' failed to run: {exc}")
        if not isinstance(res, AssumptionResult):
            return AssumptionResult(
                passed=None, reasoning=f"Check '{check_name}' did not return an AssumptionResult."
            )
        return res

    def checked(self, node: str, check_name: str, kind: str) -> Optional[bool]:
        res = self._safe_check(check_name)
        self.result.assumptions[check_name] = res
        outcome = {True: "pass", False: "fail", None: "can't tell"}[res.passed]
        self.step(node, kind, outcome, res.reasoning)
        return res.passed

    def advisory(self, node: str, check_name: str) -> Optional[bool]:
        passed = self.checked(node, check_name, "advisory")
        if passed is False:
            self.result.warnings.append(f"{node}: failed (advisory).")
        elif passed is None:
            self.result.warnings.append(f"{node}: could not be assessed (advisory).")
        return passed

    def gate(self, node: str, check_name: str, fail_reason: str) -> bool:
        """Run a gate. Returns True when traversal may continue."""
        passed = self.checked(node, check_name, "gate")
        if passed is False:
            self.end(fail_reason)
            return False
        if passed is None:
            self.result.warnings.append(f"{node}: could not be tested; continuing with warning.")
        return True

    def _prompt(self, message: str) -> Optional[str]:
        if self.prompt_callback is None:
            return None
        try:
            answer = self.prompt_callback(message)
        except Exception as exc:
            logger.warning("prompt_callback raised: %s", exc)
            return None
        if not answer or str(answer).strip().lower() in {"skip", "none"}:
            return None
        return str(answer).strip()

    # -- traversal ----------------------------------------------------------

    def run(self) -> TreeResult:
        self._entry()
        if not self.result.ended:
            is_rct = bool(self.props.get("is_rct"))
            self.step("Is this a randomized trial?", "structural", "Yes" if is_rct else "No")
            if is_rct:
                self._rct_branch()
            else:
                self._observational_branch()
        self.result.description = self.description
        return self.result

    def _entry(self) -> None:
        # 1. dataset description (prompt loop in the draft; batch mode skips it)
        if self.description:
            self.step("Is there a dataset description provided?", "structural", "Yes")
        else:
            self.step("Is there a dataset description provided?", "structural", "No")
            for _ in range(MAX_PROMPTS):
                answer = self._prompt(DESCRIPTION_PROMPT)
                if answer:
                    self.description = answer
                    self.checks = self.factory(self.description)
                    self.step("Prompt user for dataset description", "process", "provided")
                    break
            if not self.description:
                self.result.warnings.append(
                    "No dataset description provided (batch mode); LLM-argued checks will report can't tell."
                )

        # 2. data structure supported
        supported = self.props.get("is_structure_supported", True)
        self.step("Is the data structure supported?", "structural", "Yes" if supported else "No")
        if not supported:
            self.end("Data structure not supported (e.g. survival data, competing risks).")
            return

        # 3. global SUTVA advisory (can't tell -> ask for info, max 3 prompts)
        attempts = 0
        passed = self.checked("Advisory: SUTVA", "sutva", "advisory")
        while passed is None and attempts < MAX_PROMPTS:
            answer = self._prompt(SUTVA_PROMPT)
            if not answer:
                break
            self.description = (self.description or "") + "\n" + answer
            self.checks = self.factory(self.description)
            attempts += 1
            passed = self.checked("Advisory: SUTVA", "sutva", "advisory")
        if passed is False:
            self.result.warnings.append("SUTVA assessment failed (advisory); possible interference.")
        elif passed is None:
            self.result.warnings.append("SUTVA could not be assessed (advisory).")

    def _rct_branch(self) -> None:
        is_enc = self.props.get("is_encouragement_design")
        if is_enc is None:
            instr = self.props.get("instrument_variable")
            is_enc = bool(instr) and instr != self.props.get("treatment_variable")
        self.step("Is this an encouragement design?", "structural", "Yes" if is_enc else "No")

        if not is_enc:
            has_pre = self.props.get("has_pre_treatment_variables")
            if has_pre is None:
                has_pre = bool(self.props.get("covariates"))
            self.step("Are there valid pre-treatment variables?", "criteria", "Yes" if has_pre else "No")
            self.leaf(M_OLS_PRE_TREATMENT if has_pre else M_DIFF_IN_MEANS)
            return

        relevance = self.checked("Gate: Relevance (High F-stat)", "iv_relevance", "gate")
        if relevance is True:
            self.leaf(M_DIFF_IN_MEANS, variant="on_assignment")
            return
        if relevance is None:
            self.result.warnings.append(
                "Instrument relevance could not be tested; routing to the validity check."
            )
        # F < 10 (or untestable): only proceed with a valid instrument
        has_valid = self.props.get("has_valid_instrument")
        if has_valid is None:
            has_valid = bool(self.props.get("instrument_variable"))
        self.step("Is there a valid instrument?", "criteria", "Yes" if has_valid else "No")
        if not has_valid:
            self.end("Weak instrument (F < 10) and no valid instrument for the encouragement design.")
            return
        self._iv_advisories()
        self.leaf(M_IV)

    def _observational_branch(self) -> None:
        treatment_type = self.props.get("treatment_variable_type", "binary")
        non_binary = treatment_type not in (None, "binary")
        self.step("Is treatment binary?", "structural", "No" if non_binary else "Yes")
        if non_binary:
            self._options(non_binary=True)
            return

        has_temporal = bool(self.props.get("has_temporal_structure", False))
        self.step("Is temporal information available?", "structural", "Pass" if has_temporal else "Fail")
        if has_temporal:
            self._did_path()
        else:
            self._running_variable_path()

    def _did_path(self) -> None:
        if not self.gate(
            "Is the group composition stable?",
            "stable_group_composition",
            "Group composition is not stable; pipeline stopped.",
        ):
            return
        self.advisory("Advisory: Baseline outcome balance", "baseline_outcome_balance")
        if not self.gate("Gate: parallel trends", "parallel_trends", "Parallel trends violated."):
            return

        anticipation = self.checked("Gate: no anticipation (placebo)", "no_anticipation", "gate")
        if anticipation is False:
            if self.props.get("anticipation_known_bounded", False):
                self.leaf(M_DID_TIME_WINDOW)
            else:
                self.end("Anticipation effects present and not bounded; no-anticipation gate failed.")
            return
        if anticipation is None:
            self.result.warnings.append("No-anticipation could not be tested; continuing with warning.")
        self.leaf(M_DID)

    def _running_variable_path(self) -> None:
        has_rv = self.props.get("has_running_variable")
        if has_rv is None:
            has_rv = bool(self.props.get("running_variable"))
        self.step(
            "Is there a running variable (group/unit variable)?",
            "structural",
            "Yes" if has_rv else "No",
        )
        if not has_rv:
            self._options(non_binary=False)
            return

        self.advisory(
            "Advisory: Continuity of potential outcomes at the cutoff",
            "rdd_continuity_potential_outcomes",
        )
        no_manipulation = self.checked(
            "Gates: No manipulation / Covariate continuity (McCrary test)",
            "rdd_no_manipulation",
            "gate",
        )
        continuity = self.checked(
            "Gates: No manipulation / Covariate continuity (McCrary test)",
            "rdd_covariate_continuity",
            "gate",
        )
        if no_manipulation is False or continuity is False:
            if self.props.get("local_manipulation", False):
                self.leaf(M_DONUT_RDD)
            else:
                self.end("McCrary/density gate failed: manipulation at the cutoff.")
            return
        if no_manipulation is None or continuity is None:
            self.result.warnings.append(
                "RDD gate: at least one component could not be tested; continuing with warning."
            )
        self.leaf(M_RDD)

    # -- identification options (instrument / mediator / backdoor) ----------

    def _options(self, non_binary: bool) -> None:
        if self._try_instrument():
            return
        if self._try_mediator():
            return
        if self._try_backdoor(non_binary):
            return
        self.end(
            "No viable identification strategy: no valid instrument, candidate mediator, "
            "or backdoor adjustment set."
        )

    def _try_instrument(self) -> bool:
        has = self.props.get("has_valid_instrument")
        if has is None:
            has = bool(self.props.get("instrument_variable"))
        self.step("Is there a valid instrument?", "criteria", "Yes" if has else "No")
        if not has:
            return False

        self._iv_advisories()
        f_stat = self.checked("Gate: High F-stat", "iv_relevance", "gate")
        if f_stat is False:
            self.end("Weak instrument (F < 10).")
            return True
        if f_stat is None:
            self.result.warnings.append("Instrument strength could not be tested; continuing with warning.")
        self.leaf(M_IV)
        return True

    def _iv_advisories(self) -> None:
        self.advisory("Advisory: Exclusion restriction", "iv_exclusion")
        self.advisory("Advisory: Exogeneity", "iv_exogeneity")
        self.advisory("Advisory: Monotonicity", "iv_monotonicity")

    def _try_mediator(self) -> bool:
        has = self.props.get("has_candidate_mediator")
        if has is None:
            has = bool(self.props.get("mediator_variable"))
        self.step("Is there a candidate mediator?", "criteria", "Yes" if has else "No")
        if not has:
            return False

        self.advisory("Advisory: Full mediation", "frontdoor_full_mediation")
        self.advisory("Advisory: No TM confounding", "frontdoor_no_tm_confounding")
        self.advisory("Advisory: T blocks M->Y confounding", "frontdoor_t_blocks_my")
        positivity = self.checked("Gate: Positivity", "positivity", "gate")
        if positivity is False:
            self.end("Positivity failed for the mediator (frontdoor) strategy.")
            return True
        if positivity is None:
            self.result.warnings.append("Positivity could not be tested; continuing with warning.")
        self.leaf(M_FRONTDOOR)
        return True

    def _try_backdoor(self, non_binary: bool) -> bool:
        has = self.props.get("has_valid_backdoor_set")
        if has is None:
            has = bool(self.props.get("covariates"))
        self.step("Is there a valid (backdoor) adjustment set?", "criteria", "Yes" if has else "No")
        if not has:
            return False

        ignorability = self.advisory("Advisory: Cond. Ignorability", "cond_ignorability")
        positivity = self.checked("Gate: Positivity", "positivity", "gate")
        if positivity is False:
            if non_binary:
                self.end("Positivity failed for the generalized propensity score strategy.")
            else:
                self.leaf(M_TRIMMED_IPW)
            return True
        if positivity is None:
            self.result.warnings.append("Positivity could not be tested; continuing with warning.")

        if non_binary:
            self.leaf(M_GPS)
            return True

        # binary: covariate balance chooses between IPW and matching
        self.step(
            "Are the covariates in the two groups balanced?",
            "criteria",
            {True: "Yes", False: "No", None: "can't tell"}[ignorability],
        )
        if ignorability is True:
            self.leaf(M_IPW)
            self.result.planned_post_checks.append("balance_after_weighting")
        else:  # False, or untestable -> matching (safer default, flagged)
            if ignorability is None:
                self.result.warnings.append(
                    "Covariate balance could not be assessed; defaulting to matching."
                )
            self.leaf(M_PSM)
            self.result.planned_post_checks.append("balance_after_matching")
        return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def select_method_v2(
    properties: Dict[str, Any],
    *,
    df: Any = None,
    description: Optional[str] = None,
    llm: Any = None,
    checks: Optional[Dict[str, Callable[[], AssumptionResult]]] = None,
    checks_factory: Optional[Callable[[Optional[str]], Dict[str, Callable[[], AssumptionResult]]]] = None,
    prompt_callback: Optional[Callable[[str], Optional[str]]] = None,
) -> TreeResult:
    """Traverse the revised decision tree.

    ``properties`` keys (missing keys fall back to sensible defaults):
      treatment_variable, outcome_variable, treatment_variable_type,
      covariates, instrument_variable, mediator_variable, running_variable,
      has_running_variable, cutoff_value, time_variable, group_variable,
      treatment_period_start, placebo_period_start, is_rct,
      has_temporal_structure,
      is_encouragement_design, has_pre_treatment_variables,
      has_valid_instrument, has_candidate_mediator, has_valid_backdoor_set,
      is_structure_supported, anticipation_known_bounded, local_manipulation.

    ``checks`` wins over ``checks_factory``; when neither is given, the real
    checks are bound via ``build_default_checks``.
    """
    traversal = _Traversal(
        properties=properties,
        checks=checks,
        checks_factory=checks_factory,
        description=description,
        prompt_callback=prompt_callback,
        df=df,
        llm=llm,
    )
    return traversal.run()


# ---------------------------------------------------------------------------
# Default check wiring (real implementations)
# ---------------------------------------------------------------------------

def _fit_propensity_scores(vars) -> Any:
    from sklearn.linear_model import LogisticRegression

    df = vars.df
    cols = list(vars.covariates)  # treatment must NOT be a feature of its own propensity model
    x = df[cols].astype(float).values
    y = df[vars.treatment].astype(int).values
    model = LogisticRegression(max_iter=1000).fit(x, y)
    return model.predict_proba(x)[:, 1]


def build_default_checks(properties: Dict[str, Any], df, description, llm):
    """Bind the real pre-model assumption checks for the traversal.

    Imports are local so the tree module stays importable without the full
    stats stack when tests inject stubs.
    """
    from cais.models import AssumptionVariables
    from cais.methods import pre_model_assumption_utils as pm

    instruments = []
    if properties.get("instrument_variable"):
        instruments = [properties["instrument_variable"]]

    vars = AssumptionVariables(
        df=df,
        treatment=properties.get("treatment_variable"),
        outcome=properties.get("outcome_variable"),
        covariates=list(properties.get("covariates") or []),
        instruments=instruments,
        running_variable=properties.get("running_variable"),
        cutoff=properties.get("cutoff_value"),
        time_var=properties.get("time_variable"),
        group_var=properties.get("group_variable"),
        mediator=properties.get("mediator_variable"),
        treatment_period_start=properties.get("treatment_period_start"),
        placebo_period_start=properties.get("placebo_period_start"),
        dataset_description=description,
        variables_summary={
            "treatment": properties.get("treatment_variable"),
            "outcome": properties.get("outcome_variable"),
            "covariates": properties.get("covariates"),
            "instrument": properties.get("instrument_variable"),
            "mediator": properties.get("mediator_variable"),
            "running_variable": properties.get("running_variable"),
        },
    )

    _ps_cache: Dict[str, Any] = {}

    def propensity_scores():
        if "ps" not in _ps_cache:
            _ps_cache["ps"] = _fit_propensity_scores(vars)
        return _ps_cache["ps"]

    def placeholder(reason: str):
        return lambda: AssumptionResult(passed=None, reasoning=reason)

    return {
        "sutva": lambda: pm.check_sutva(vars, llm=llm),
        "stable_group_composition": lambda: pm.check_stable_group_composition(vars, llm=llm),
        "baseline_outcome_balance": lambda: pm.check_baseline_outcome_balance(vars),
        "parallel_trends": lambda: pm.check_parallel_trends(vars),
        "no_anticipation": lambda: pm.check_no_anticipation(vars),
        "iv_relevance": lambda: pm.check_iv_relevance(vars),
        "iv_exclusion": lambda: pm.check_iv_exclusion(vars, llm=llm),
        "iv_exogeneity": lambda: pm.check_iv_exogeneity(vars, llm=llm),
        "iv_monotonicity": lambda: pm.check_iv_monotonicity(vars, llm=llm),
        "cond_ignorability": lambda: pm.check_cond_ignorability(vars),
        "positivity": lambda: pm.check_positivity(vars, propensity_scores=propensity_scores()),
        "rdd_no_manipulation": lambda: pm.check_rdd_no_manipulation(vars),
        "rdd_covariate_continuity": lambda: pm.check_rdd_covariate_continuity(vars),
        "rdd_continuity_potential_outcomes": lambda: pm.check_rdd_continuity_potential_outcomes(vars, llm=llm),
        "frontdoor_full_mediation": lambda: pm.check_frontdoor_full_mediation(vars, llm=llm),
        "frontdoor_no_tm_confounding": lambda: pm.check_frontdoor_no_TM_confounding(vars, llm=llm),
        "frontdoor_t_blocks_my": lambda: pm.check_frontdoor_T_blocks_MY(vars, llm=llm),
        "frontdoor_positivity": lambda: pm.check_frontdoor_positivity(vars),
        "gps_cond_ignorability": placeholder("GPS pre-model cond. ignorability check is not implemented yet."),
        "gps_positivity": placeholder("GPS pre-model positivity check is not implemented yet."),
    }
