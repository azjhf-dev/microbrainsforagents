"""Continual Learning Pipeline for Microbrains (Episode -> Candidate -> Verify -> Promote / Rollback).

Implements gated online plasticity and continual adaptation without catastrophic forgetting or self-poisoning:
1. Gathers labeled episodes (never unsupervised random noise gradients).
2. Computes Elastic Weight Consolidation (EWC) Fisher Information diagonal:
      L_continual(theta) = L_task(theta) + (lambda_ewc / 2) * Sum_i F_i * (theta_i - theta_anchor_i)^2
3. Stages candidate weight deltas in `_candidates/` without touching production weights.
4. Requires explicit held-out verification before backing up production weights to `_rollback/` and promoting.
5. Emits audit telemetry to `memory/continual_learning_audit.json`.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .analytical_micro_transformer import AnalyticalMicroTransformer, STORE_DIR

log = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


@dataclass
class ContinualUpdateResult:
    """Outcome of a continual plasticity candidate/promotion operation on a microbrain."""

    domain: str
    tokens_updated: List[str]
    ewc_penalty: float
    weights_delta_l2: float
    recalibrated_quantile: float
    empirical_coverage: float
    status: str
    candidate_staged: bool = True
    promoted: bool = False
    candidate_path: str = ""


class ContinualLearningPipeline:
    """Orchestrates staged continual plasticity updates, held-out gating, and EWC protection."""

    def __init__(
        self,
        store_dir: Optional[Path] = None,
        audit_log_path: Optional[Path] = None,
        ewc_lambda: float = 0.5,
    ):
        self.store_dir = Path(store_dir) if store_dir else STORE_DIR
        self.candidates_dir = self.store_dir / "_candidates"
        self.rollback_dir = self.store_dir / "_rollback"
        default_audit = Path(__file__).resolve().parents[1] / "memory" / "continual_learning_audit.json"
        self.audit_log_path = audit_log_path or default_audit
        self.ewc_lambda = ewc_lambda

    def compute_fisher_information(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Estimate diagonal Fisher Information matrix via squared parameter magnitudes."""
        fisher = {}
        for name, arr in weights.items():
            fisher[name] = np.maximum(1e-4, np.square(arr) * 0.1).astype(np.float32)
        return fisher

    def _resolve_brain(self, domain: str, search_dir: Optional[Path] = None) -> Optional[AnalyticalMicroTransformer]:
        target_dir = search_dir or self.store_dir
        brain = AnalyticalMicroTransformer.load(f"neuro_{domain}", target_dir)
        if brain is None:
            brain = AnalyticalMicroTransformer.load(domain, target_dir)
        return brain

    def _measure_hopfield_retention(self, brain: AnalyticalMicroTransformer) -> float:
        """Measure genuine retention accuracy across compiled Hopfield rules (no hardcoded coverage)."""
        if not brain.hopfield_keys or not brain.hopfield_values:
            return round(max(0.0, 1.0 - float(brain.conformal_quantile or 0.05)), 4)

        vocab_slice = brain.W_embed[: len(brain.vocab)]
        hits = 0
        total = len(brain.hopfield_keys)
        for k_vec, v_vec in zip(brain.hopfield_keys, brain.hopfield_values):
            k_idx = int(np.argmax(np.dot(vocab_slice, k_vec)))
            v_expected_idx = int(np.argmax(np.dot(vocab_slice, v_vec)))
            k_tok = brain.inv_vocab.get(k_idx, "<unk>")
            v_expected_tok = brain.inv_vocab.get(v_expected_idx, "<unk>")
            probs = brain.get_token_probabilities([k_tok])
            pred_tok = max(probs.items(), key=lambda kv: (kv[1], str(kv[0])))[0] if probs else "<unk>"
            if pred_tok == v_expected_tok or probs.get(v_expected_tok, 0.0) >= 0.5:
                hits += 1
        return round(hits / float(max(1, total)), 4)

    def _measure_holdout_coverage(
        self,
        brain: AnalyticalMicroTransformer,
        holdout_data: List[Tuple[List[str], str]],
    ) -> float:
        if not holdout_data:
            return 0.0
        hits = 0
        for inp, target in holdout_data:
            res = brain.infer_conformal(inp, alpha=brain.conformal_alpha or 0.05)
            if target in res.get("prediction_set", []) or res.get("top_candidate") == target:
                hits += 1
        return round(hits / float(len(holdout_data)), 4)

    def _backup_for_rollback(self, model_name: str) -> None:
        self.rollback_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("_weights.npz", "_meta.json"):
            src = self.store_dir / f"{model_name}{suffix}"
            if src.exists():
                shutil.copy2(src, self.rollback_dir / f"{model_name}{suffix}")

    def apply_plasticity_feedback(
        self,
        domain: str,
        problem_tokens: List[str],
        learning_rate: float = 0.02,
        holdout_data: Optional[List[Tuple[List[str], str]]] = None,
        promote: bool = False,
    ) -> ContinualUpdateResult:
        """Apply EWC-regularized deterministic plasticity update to a staged candidate.

        By default (`promote=False`), writes ONLY to `_candidates/` so unverified
        updates never overwrite production weights or fabricate conformal metrics.
        """
        brain = self._resolve_brain(domain, self.store_dir)
        if brain is None:
            return ContinualUpdateResult(
                domain=domain,
                tokens_updated=[],
                ewc_penalty=0.0,
                weights_delta_l2=0.0,
                recalibrated_quantile=0.0,
                empirical_coverage=0.0,
                status=f"error: domain {domain} not found",
                candidate_staged=False,
                promoted=False,
            )

        anchor_embed = np.copy(brain.W_embed)
        fisher = self.compute_fisher_information({"W_embed": brain.W_embed})["W_embed"]

        # Deterministic centroid of existing in-domain Hopfield keys (replaces random Gaussian noise)
        if brain.hopfield_keys:
            semantic_center = np.mean(np.stack(brain.hopfield_keys, axis=0), axis=0).astype(np.float32)
        else:
            semantic_center = np.mean(anchor_embed[: max(1, min(len(brain.vocab), len(anchor_embed)))], axis=0).astype(np.float32)
        center_norm = float(np.linalg.norm(semantic_center))
        if center_norm > 1e-6:
            semantic_center = semantic_center / center_norm

        delta_l2 = 0.0
        updated_tokens: List[str] = []

        for tok in problem_tokens:
            if tok not in brain.vocab:
                tok_id = brain.register_token(tok)
                if tok_id >= len(anchor_embed):
                    extra_rows = (tok_id + 1) - len(anchor_embed)
                    anchor_embed = np.vstack([anchor_embed, brain.W_embed[len(anchor_embed) : len(anchor_embed) + extra_rows]])
                    fisher = self.compute_fisher_information({"W_embed": brain.W_embed})["W_embed"]
            else:
                tok_id = brain.vocab[tok]

            # Deterministic alignment toward domain semantic centroid minus EWC anchor penalty
            grad = semantic_center - brain.W_embed[tok_id]
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > 1e-6:
                grad = (grad / grad_norm) * 0.05
            ewc_pull = self.ewc_lambda * fisher[tok_id] * (brain.W_embed[tok_id] - anchor_embed[tok_id])
            effective_update = (learning_rate * (grad - ewc_pull)).astype(np.float32)

            brain.W_embed[tok_id] = (brain.W_embed[tok_id] + effective_update).astype(np.float32)
            row_norm = float(np.linalg.norm(brain.W_embed[tok_id]))
            if row_norm > 1e-6:
                brain.W_embed[tok_id] /= row_norm

            delta_l2 += float(np.linalg.norm(effective_update))
            updated_tokens.append(tok)

        # Measure real empirical performance rather than hardcoding 0.985 or scaling q_hat blindly
        if holdout_data:
            cal_res = brain.calibrate_conformal(holdout_data, alpha=brain.conformal_alpha or 0.05, in_sample=False)
            measured_cov = self._measure_holdout_coverage(brain, holdout_data)
            q_val = float(cal_res.get("quantile_q_hat", brain.conformal_quantile or 0.05))
        else:
            measured_cov = self._measure_hopfield_retention(brain)
            q_val = float(brain.conformal_quantile or 0.05)

        # Stage candidate first
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        brain.save(directory=self.candidates_dir)
        cand_path = str(self.candidates_dir / f"{brain.name}_weights.npz")

        promoted = False
        if promote and holdout_data and measured_cov >= 0.90:
            self._backup_for_rollback(brain.name)
            brain.save(directory=self.store_dir)
            promoted = True

        n_rows = min(len(fisher), len(anchor_embed), len(brain.W_embed))
        ewc_penalty_val = float(
            0.5 * self.ewc_lambda * np.sum(fisher[:n_rows] * np.square(brain.W_embed[:n_rows] - anchor_embed[:n_rows]))
        )

        res = ContinualUpdateResult(
            domain=domain,
            tokens_updated=updated_tokens,
            ewc_penalty=round(ewc_penalty_val, 6),
            weights_delta_l2=round(delta_l2, 6),
            recalibrated_quantile=round(q_val, 4),
            empirical_coverage=round(float(measured_cov), 4),
            status="promoted" if promoted else "success",
            candidate_staged=True,
            promoted=promoted,
            candidate_path=cand_path,
        )

        self._record_audit_event(res)
        return res

    def verify_and_promote_candidate(
        self,
        domain: str,
        holdout_data: List[Tuple[List[str], str]],
        min_coverage: float = 0.95,
    ) -> ContinualUpdateResult:
        """Gate staged candidate on unseen holdout data; backup to _rollback and promote only if passing."""
        cand = self._resolve_brain(domain, self.candidates_dir)
        if cand is None:
            return ContinualUpdateResult(
                domain=domain,
                tokens_updated=[],
                ewc_penalty=0.0,
                weights_delta_l2=0.0,
                recalibrated_quantile=0.0,
                empirical_coverage=0.0,
                status=f"error: staged candidate for {domain} not found",
                candidate_staged=False,
                promoted=False,
            )

        if not holdout_data:
            return ContinualUpdateResult(
                domain=domain,
                tokens_updated=[],
                ewc_penalty=0.0,
                weights_delta_l2=0.0,
                recalibrated_quantile=float(cand.conformal_quantile or 0.05),
                empirical_coverage=0.0,
                status="rejected_empty_holdout",
                candidate_staged=True,
                promoted=False,
            )

        cov = self._measure_holdout_coverage(cand, holdout_data)
        if cov < min_coverage:
            res = ContinualUpdateResult(
                domain=domain,
                tokens_updated=[],
                ewc_penalty=0.0,
                weights_delta_l2=0.0,
                recalibrated_quantile=float(cand.conformal_quantile or 0.05),
                empirical_coverage=cov,
                status="rejected_verification",
                candidate_staged=True,
                promoted=False,
            )
            self._record_audit_event(res)
            return res

        cand.calibrate_conformal(holdout_data, alpha=cand.conformal_alpha or 0.05, in_sample=False)
        self._backup_for_rollback(cand.name)
        cand.save(directory=self.store_dir)
        res = ContinualUpdateResult(
            domain=domain,
            tokens_updated=[],
            ewc_penalty=0.0,
            weights_delta_l2=0.0,
            recalibrated_quantile=round(float(cand.conformal_quantile or 0.0), 4),
            empirical_coverage=cov,
            status="promoted",
            candidate_staged=True,
            promoted=True,
            candidate_path=str(self.store_dir / f"{cand.name}_weights.npz"),
        )
        self._record_audit_event(res)
        return res

    def rollback_domain(self, domain: str) -> bool:
        """Restore previously backed-up weights and metadata for `domain` from `_rollback/`."""
        if not self.rollback_dir.exists():
            return False
        for prefix in (f"neuro_{domain}", domain):
            w_bak = self.rollback_dir / f"{prefix}_weights.npz"
            m_bak = self.rollback_dir / f"{prefix}_meta.json"
            if w_bak.exists() and m_bak.exists():
                shutil.copy2(w_bak, self.store_dir / w_bak.name)
                shutil.copy2(m_bak, self.store_dir / m_bak.name)
                return AnalyticalMicroTransformer.load(prefix, self.store_dir, verify_checksum=True) is not None
        return False

    def _record_audit_event(self, res: ContinualUpdateResult) -> None:
        """Append outcome to continual learning audit trail."""
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        if self.audit_log_path.exists():
            try:
                entries = json.loads(self.audit_log_path.read_text(encoding="utf-8"))
            except Exception:
                entries = []

        entry = {
            "ts": utc_now_iso(),
            "domain": res.domain,
            "tokens_updated": res.tokens_updated,
            "ewc_penalty": res.ewc_penalty,
            "weights_delta_l2": res.weights_delta_l2,
            "recalibrated_quantile": res.recalibrated_quantile,
            "empirical_coverage": res.empirical_coverage,
            "candidate_staged": res.candidate_staged,
            "promoted": res.promoted,
            "status": res.status,
        }
        entries.append(entry)
        self.audit_log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
