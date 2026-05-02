"""Experiment runner for comparing Phase 6 web evidence approaches."""

from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.web_evidence.agent import run_web_evidence_agent
from agents.web_evidence.experiments.approaches import TIE_BREAK_ORDER, WebEvidenceApproach, get_approach
from agents.web_evidence.schemas import WebEvidenceRequest
from agents.web_evidence.search_client import StaticSearchClient
from agents.web_evidence.source_ranker import is_rejected_domain


VALIDATION_QUESTIONS = [
    {"category": "updated_guidelines", "question": "latest guideline for community acquired pneumonia antibiotics in adults"},
    {"category": "updated_guidelines", "question": "updated guideline recommendations for hypertension treatment in adults"},
    {"category": "updated_guidelines", "question": "current guideline management of atrial fibrillation anticoagulation"},
    {"category": "updated_guidelines", "question": "latest guideline for diabetes type 2 pharmacologic treatment"},
    {"category": "drug_warnings", "question": "FDA warning for fluoroquinolone antibiotic adverse effects"},
    {"category": "drug_warnings", "question": "updated safety warning for statins and liver injury"},
    {"category": "drug_warnings", "question": "contraindications and warnings for ACE inhibitors in pregnancy"},
    {"category": "infectious_disease_updates", "question": "CDC updated COVID vaccination guidance adults"},
    {"category": "infectious_disease_updates", "question": "latest influenza antiviral treatment recommendations adults"},
    {"category": "infectious_disease_updates", "question": "guidance for tuberculosis screening high risk adults"},
    {"category": "vaccination_guidance", "question": "adult pneumococcal vaccine recommendations older adults"},
    {"category": "vaccination_guidance", "question": "hepatitis B vaccination recommendations adults with diabetes"},
    {"category": "vaccination_guidance", "question": "RSV vaccine recommendations older adults"},
    {"category": "screening_recommendations", "question": "breast cancer screening recommendations average risk women"},
    {"category": "screening_recommendations", "question": "colorectal cancer screening recommendations adults"},
    {"category": "screening_recommendations", "question": "lung cancer screening recommendations smoking history"},
    {"category": "treatment_recommendations", "question": "migraine acute treatment recommendations adults"},
    {"category": "treatment_recommendations", "question": "heart failure guideline directed medical therapy recommendations"},
    {"category": "treatment_recommendations", "question": "asthma management guideline recommendations adults"},
    {"category": "treatment_recommendations", "question": "urinary tract infection treatment recommendations adult women"},
]


UNSAFE_PATTERNS = [
    re.compile(r"\btake\s+\d+\s*(mg|mcg|g)\b", re.I),
    re.compile(r"\bstop taking\b", re.I),
    re.compile(r"\bstart taking\b", re.I),
    re.compile(r"\bdouble your dose\b", re.I),
]

PII_PATTERNS = [
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b\d{3,4}[-.\s]?\d{4}\b"),
    re.compile(r"\b(?:mrn|hospital id|patient id|national id|ssn)\b", re.I),
]


def _round(value: float) -> float:
    return round(value, 4)


def _privacy_query_pass(query: str) -> bool:
    return not any(pattern.search(query) for pattern in PII_PATTERNS)


def _has_physician_review_language(summary: str) -> bool:
    lowered = summary.lower()
    return "physician review" in lowered or "clinician judgment" in lowered or "doctor" in lowered


def _safety_penalty(summary: str) -> float:
    penalty = 0.0
    if any(pattern.search(summary) for pattern in UNSAFE_PATTERNS):
        penalty += 0.5
    if not _has_physician_review_language(summary):
        penalty += 0.25
    return min(penalty, 1.0)


def compute_approach_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics for one approach."""
    total = len(records)
    results = [record.get("result", {}) for record in records if record.get("valid_json")]
    decisions = [result.get("final_decision") for result in results]
    sources = [source for result in results for source in result.get("sources", [])]
    tier1 = [source for source in sources if source.get("source_tier") == 1]
    tier2 = [source for source in sources if source.get("source_tier") == 2]
    tier3_unknown = [source for source in sources if source.get("source_tier") not in {1, 2}]
    rejected_leaks = [source for source in sources if is_rejected_domain(source.get("domain", ""))]
    privacy_passes = [
        result
        for result in results
        if result.get("privacy_report", {}).get("deidentified") is True
        and _privacy_query_pass(result.get("sanitized_query", ""))
    ]
    summaries = [str(result.get("summary", "")) for result in results]
    safety_warnings = [
        result
        for result in results
        if result.get("council_review", {}).get("safety_critic", {}).get("decision") != "accept"
        or _safety_penalty(str(result.get("summary", ""))) > 0
    ]
    conflicts = [result for result in results if result.get("conflicts")]
    claims = [claim for result in results for claim in result.get("key_findings", [])]
    latencies = [float(record.get("latency_seconds", 0)) for record in records]
    llm_failures = sum(int(result.get("execution_metadata", {}).get("llm_failure_count", 0)) for result in results)
    fallbacks = sum(int(result.get("execution_metadata", {}).get("deterministic_fallback_count", 0)) for result in results)

    source_quality_score = 0.0
    if sources:
        source_quality_score = max(0.0, min(1.0, (len(tier1) * 1.0 + len(tier2) * 0.8 + len(tier3_unknown) * 0.35) / len(sources)))

    avg_claims = len(claims) / total if total else 0
    reliable_support = len([claim for claim in claims if claim.get("supporting_sources")]) / max(len(claims), 1)
    evidence_completeness_score = max(0.0, min(1.0, (min(avg_claims / 4, 1.0) * 0.65) + (reliable_support * 0.35)))

    safety_score = 1.0
    if summaries:
        safety_score = max(0.0, min(1.0, 1.0 - (sum(_safety_penalty(summary) for summary in summaries) / len(summaries))))
    if safety_warnings:
        safety_score = max(0.0, safety_score - min(0.35, len(safety_warnings) / max(total, 1) * 0.2))

    privacy_score = 0.0 if len(privacy_passes) < total else 1.0
    if total and len(privacy_passes) != total:
        privacy_score = len(privacy_passes) / total

    overall_score = (
        source_quality_score * 0.30
        + evidence_completeness_score * 0.25
        + safety_score * 0.25
        + privacy_score * 0.20
    )

    return {
        "total_queries": total,
        "accepted_count": decisions.count("accept"),
        "accept_with_caution_count": decisions.count("accept_with_caution"),
        "rejected_count": decisions.count("reject"),
        "valid_json_rate": _round(len(results) / total * 100) if total else 0,
        "privacy_pass_rate": _round(len(privacy_passes) / total * 100) if total else 0,
        "average_sources_per_query": _round(len(sources) / total) if total else 0,
        "tier1_source_percentage": _round(len(tier1) / len(sources) * 100) if sources else 0,
        "tier2_source_percentage": _round(len(tier2) / len(sources) * 100) if sources else 0,
        "rejected_domain_leak_count": len(rejected_leaks),
        "average_claims_per_query": _round(avg_claims),
        "conflict_detection_count": len(conflicts),
        "safety_warning_count": len(safety_warnings),
        "average_latency_seconds": _round(sum(latencies) / len(latencies)) if latencies else 0,
        "llm_failure_count": llm_failures,
        "deterministic_fallback_count": fallbacks,
        "source_quality_score": _round(source_quality_score),
        "evidence_completeness_score": _round(evidence_completeness_score),
        "safety_score": _round(safety_score),
        "privacy_score": _round(privacy_score),
        "overall_score": _round(overall_score),
    }


def choose_recommended_approach(metrics_by_approach: dict[str, dict[str, Any]]) -> str:
    """Choose best approach by score, using medical-system tie-break order."""
    def sort_key(name: str) -> tuple[float, int]:
        score = float(metrics_by_approach[name].get("overall_score", 0))
        tie_index = TIE_BREAK_ORDER.index(name) if name in TIE_BREAK_ORDER else len(TIE_BREAK_ORDER)
        return (score, -tie_index)

    return max(metrics_by_approach, key=sort_key)


def run_approach_comparison(
    approach_names: list[str],
    provider: str = "mock",
    model: str | None = None,
    max_questions: int | None = None,
    max_sources: int = 5,
    mock_mode: bool = True,
    output_dir: Path | None = None,
    write_csv: bool = True,
) -> dict[str, Any]:
    """Run all selected approaches across the validation question set."""
    selected_questions = VALIDATION_QUESTIONS[:max_questions] if max_questions else VALIDATION_QUESTIONS
    records_by_approach: dict[str, list[dict[str, Any]]] = {}
    approach_configs: dict[str, dict[str, Any]] = {}
    search_client = StaticSearchClient() if mock_mode else None

    for approach_name in approach_names:
        effective_provider = "none" if approach_name == "deterministic_only" else provider
        approach: WebEvidenceApproach = get_approach(approach_name, provider=effective_provider)
        approach_configs[approach.name] = approach.to_dict()
        records: list[dict[str, Any]] = []
        for entry in selected_questions:
            request = WebEvidenceRequest(
                clinical_question=entry["question"],
                patient_context=None,
                reason_for_web="Guidelines or public medical evidence may have changed after textbook publication.",
            )
            start = time.perf_counter()
            try:
                result = run_web_evidence_agent(
                    request,
                    max_sources=max_sources,
                    search_client=search_client,
                    use_mock_fetch=mock_mode,
                    approach=approach,
                    llm_provider=effective_provider,
                    llm_model=model,
                )
                result_dict = result.to_dict()
                valid_json = True
                error = None
            except Exception as exc:  # noqa: BLE001
                result_dict = {"error": str(exc)}
                valid_json = False
                error = str(exc)
            latency = time.perf_counter() - start
            records.append(
                {
                    "approach": approach.name,
                    "category": entry["category"],
                    "question": entry["question"],
                    "valid_json": valid_json,
                    "latency_seconds": round(latency, 4),
                    "error": error,
                    "result": result_dict,
                }
            )
        records_by_approach[approach.name] = records

    metrics_by_approach = {
        approach_name: compute_approach_metrics(records)
        for approach_name, records in records_by_approach.items()
    }
    recommended = choose_recommended_approach(metrics_by_approach)
    comparison_table = [
        {"approach": name, **metrics_by_approach[name]}
        for name in sorted(metrics_by_approach, key=lambda item: TIE_BREAK_ORDER.index(item) if item in TIE_BREAK_ORDER else 99)
    ]
    payload = {
        "phase": "phase_6_web_evidence_approach_comparison",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "mock" if mock_mode else "searxng",
        "provider": provider,
        "model": model,
        "approach_configs": approach_configs,
        "per_question_results": records_by_approach,
        "aggregate_metrics": metrics_by_approach,
        "comparison_table": comparison_table,
        "recommended_approach": recommended,
        "recommendation_reason": (
            "Best overall_score with tie-break preference for privacy, safety, deterministic source validation, "
            "and hospital deployment reliability."
        ),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "web_evidence_approach_comparison.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if write_csv:
            csv_path = output_dir / "web_evidence_approach_comparison.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(comparison_table[0].keys()) if comparison_table else ["approach"])
                writer.writeheader()
                writer.writerows(comparison_table)
            payload["csv_output_path"] = str(csv_path)
        payload["json_output_path"] = str(json_path)

    return payload
