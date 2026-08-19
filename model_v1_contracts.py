"""Phase-0 contracts for the redesigned disease-conditioned model."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import model_v1_config as cfg
from .disease_stage import validate_reference_exclusion


@dataclass(frozen=True)
class Phase0Report:
    engineering_decision: str
    real_data_ready: bool
    real_fit_blockers: tuple[str, ...]
    sessions: int
    subjects: int
    session_counts: dict[str, int]
    subject_counts: dict[str, int]
    strict_npz_count: int
    no_censor_npz_count: int
    reference_subjects_with_any_endpoint: int
    input_sha256: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_npz_header(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate the non-outcome header needed to freeze ROI order and TR."""

    required = {"timeseries", "roi_ids", "roi_labels", "tr_seconds"}
    with np.load(path, allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path.name} missing keys: {sorted(missing)}")
        shape = np.asarray(payload["timeseries"]).shape
        roi_ids = np.asarray(payload["roi_ids"], int)
        roi_labels = np.asarray(payload["roi_labels"], str)
        tr_seconds = float(np.asarray(payload["tr_seconds"]).item())
    if len(shape) != 2 or shape[1] != cfg.ROI_COUNT:
        raise ValueError(f"{path.name} has invalid time-series shape {shape}")
    if roi_ids.shape != (cfg.ROI_COUNT,) or roi_labels.shape != (cfg.ROI_COUNT,):
        raise ValueError(f"{path.name} has invalid ROI metadata shapes")
    if not np.isclose(tr_seconds, cfg.TR_SECONDS, atol=1e-6):
        raise ValueError(f"{path.name} has TR={tr_seconds}, expected {cfg.TR_SECONDS}")
    return roi_ids, roi_labels, tr_seconds


def npz_session_identity(path: Path) -> tuple[str, str]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"subject", "session", "run", "clinical_group"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path.name} missing identity keys: {sorted(missing)}")
        subject = str(np.asarray(payload["subject"]).item())
        session = str(np.asarray(payload["session"]).item())
        run = str(np.asarray(payload["run"]).item())
        group = str(np.asarray(payload["clinical_group"]).item())
    return f"{subject}_{session}_run-{run}", group


def audit_phase0() -> tuple[Phase0Report, list[dict[str, str]], np.ndarray, np.ndarray]:
    """Audit frozen inputs without reading any model outcome."""

    required = (
        cfg.AUDIT_STATUS,
        cfg.FMRI_AUDIT_CSV,
        cfg.REFERENCE_SUMMARY_CSV,
        cfg.REFERENCE_VISIT_COUNTS_CSV,
        cfg.CONVERSION_AUDIT_CSV,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen audit inputs: {missing}")

    status = json.loads(cfg.AUDIT_STATUS.read_text(encoding="utf-8"))
    if status.get("decision") != "DESCRIPTIVE_REFERENCE_COHORT_AUDIT_COMPLETE":
        raise ValueError("corrected clinical/pathology audit is not complete")

    rows = _read_csv(cfg.FMRI_AUDIT_CSV)
    if len(rows) != cfg.EXPECTED_SESSIONS:
        raise ValueError(f"expected {cfg.EXPECTED_SESSIONS} audited sessions, found {len(rows)}")
    keys = [row["session_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate session_key in fMRI audit")
    session_counts = Counter(row["group"] for row in rows)
    subjects_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subjects_by_group[row["group"]].add(row["subject"])
    subject_counts = {group: len(subjects_by_group[group]) for group in cfg.EXPECTED_SUBJECT_COUNTS}
    if dict(session_counts) != cfg.EXPECTED_SESSION_COUNTS:
        raise ValueError(f"session-count mismatch: {dict(session_counts)}")
    if subject_counts != cfg.EXPECTED_SUBJECT_COUNTS:
        raise ValueError(f"subject-count mismatch: {subject_counts}")
    subjects = {row["subject"] for row in rows}
    reference_rows = _read_csv(cfg.REFERENCE_VISIT_COUNTS_CSV)
    if len(reference_rows) != cfg.EXPECTED_REFERENCE_SUBJECTS_WITH_ANY_ENDPOINT:
        raise ValueError(
            f"expected {cfg.EXPECTED_REFERENCE_SUBJECTS_WITH_ANY_ENDPOINT} reference subjects, "
            f"found {len(reference_rows)}"
        )
    validate_reference_exclusion([row["PTID"] for row in reference_rows], subjects)

    strict_files = sorted(cfg.STRICT_TIMESERIES_ROOT.glob("*.npz"))
    no_censor_files = sorted(cfg.NO_CENSOR_TIMESERIES_ROOT.glob("*.npz"))
    if len(strict_files) != cfg.EXPECTED_SESSIONS:
        raise ValueError(f"strict-censor NPZ count is {len(strict_files)}, expected {cfg.EXPECTED_SESSIONS}")
    if len(no_censor_files) < cfg.EXPECTED_SESSIONS:
        raise ValueError("no-censor directory cannot cover the frozen session set")
    roi_ids, roi_labels, _ = validate_npz_header(strict_files[0])
    strict_identities = {}
    for path in strict_files:
        key, group = npz_session_identity(path)
        if key in strict_identities:
            raise ValueError(f"duplicate strict-censor session identity: {key}")
        strict_identities[key] = group
    audit_groups = {row["session_key"]: row["group"] for row in rows}
    if strict_identities != audit_groups:
        missing_keys = sorted(set(audit_groups).difference(strict_identities))
        extra_keys = sorted(set(strict_identities).difference(audit_groups))
        raise ValueError(f"strict-censor/audit identity mismatch: missing={missing_keys[:3]}, extra={extra_keys[:3]}")
    no_censor_identities = [npz_session_identity(path)[0] for path in no_censor_files]
    if len(set(no_censor_identities)) != len(no_censor_identities):
        raise ValueError("duplicate no-censor session identity")
    no_censor_keys = set(no_censor_identities)
    if not set(audit_groups).issubset(no_censor_keys):
        raise ValueError("no-censor directory does not cover the exact frozen 214-session intersection")
    for path in strict_files[1:]:
        other_ids, other_labels, _ = validate_npz_header(path)
        if not np.array_equal(other_ids, roi_ids) or not np.array_equal(other_labels, roi_labels):
            raise ValueError(f"ROI order mismatch in {path.name}")

    hashes = {path.name: sha256_file(path) for path in required}
    blockers = tuple(cfg.real_fit_blockers())
    report = Phase0Report(
        engineering_decision="ENGINEERING_CONTRACTS_GO",
        real_data_ready=not blockers,
        real_fit_blockers=blockers,
        sessions=len(rows),
        subjects=len(subjects),
        session_counts=dict(session_counts),
        subject_counts=subject_counts,
        strict_npz_count=len(strict_files),
        no_censor_npz_count=len(no_censor_files),
        reference_subjects_with_any_endpoint=len(reference_rows),
        input_sha256=hashes,
    )
    return report, rows, roi_ids, roi_labels
