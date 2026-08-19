"""Frozen paths and unresolved scientific choices for the redesigned model.

Entry scripts take no command-line arguments.  Edit this file only after the
corresponding scientific decision has been recorded and approved.
"""

from pathlib import Path


PROJECT_ROOT = Path(r"D:\wiucas\Alzheimer\程序\codex_code")
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "adni_clinical_pathology_audit_20260814"
AUDIT_STATUS = AUDIT_ROOT / "ADNI_CLINICAL_PATHOLOGY_AUDIT_STATUS_20260814.json"
FMRI_AUDIT_CSV = AUDIT_ROOT / "ADNI130_CLINICAL_PATHOLOGY_AUDIT_20260814.csv"
REFERENCE_SUMMARY_CSV = AUDIT_ROOT / "ADNIMERGE2_REFERENCE_COHORT_SUMMARY_EXCL_FMRI130_20260814.csv"
REFERENCE_VISIT_COUNTS_CSV = AUDIT_ROOT / "ADNIMERGE2_REFERENCE_COHORT_VISIT_COUNTS_EXCL_FMRI130_20260814.csv"
CONVERSION_AUDIT_CSV = AUDIT_ROOT / "ADNIMERGE2_AMYLOID_CONVERSION_ANCHOR_AUDIT_EXCL_FMRI130_20260814.csv"

STRICT_TIMESERIES_ROOT = Path(r"D:\wiucas\Alzheimer\fmri_output_NPIprep_revised\strict_censor")
NO_CENSOR_TIMESERIES_ROOT = Path(r"D:\wiucas\Alzheimer\fmri_output_NPIprep_revised\no_censor_motion_regression")
NPI_ROOT = Path(r"D:\wiucas\Alzheimer\NPI_EC_exploration_20260728\confirm_5seed_two_modes\compact06_ridge_n3")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ad_model_v1_20260818"
PHASE0_ROOT = OUTPUT_ROOT / "phase0_contracts"

ROI_COUNT = 90
NETWORK_COUNT = 9
N_LAGS = 3
TR_SECONDS = 0.607
EXPECTED_SESSION_COUNTS = {"CN": 117, "MCI": 83, "AD": 14}
EXPECTED_SUBJECT_COUNTS = {"CN": 75, "MCI": 45, "AD": 10}
EXPECTED_SESSIONS = 214
EXPECTED_SUBJECTS = 130
EXPECTED_REFERENCE_SUBJECTS_WITH_ANY_ENDPOINT = 4628
GLOBAL_SEED = 20260818

# These are deliberately unresolved.  None means real-data fitting must stop.
AMYLOID_POSITIVITY_THRESHOLD_CENTILOID = None
AAL90_TO_NETWORK9_MAPPING = None
FMRI_SUBJECT_SPLIT_FILE = None
RECOVERY_GATE_FILE = None
LOSS_WEIGHT_PROTOCOL_FILE = None
HRF_PROTOCOL_FILE = None
NPI_JAX_MIGRATION_GATE_FILE = None
PET_REGIONAL_MAPPING_FILE = None  # optional until regional pathology is used
STRUCTURAL_CONNECTIVITY_FILE = None  # required only for future M3, not M0/M1

# Synthetic-only engineering defaults.  They are not empirical model choices.
SYNTHETIC_STAGE_GRID = (-4.0, 4.0, 1601)
SYNTHETIC_STAGE_PRIOR_SD = 2.0
SYNTHETIC_N_MODES = 3
SYNTHETIC_M0_TAU_SECONDS = 1.0
SYNTHETIC_M1_TAU_SECONDS = 1.0
SYNTHETIC_M1_A0 = 0.4
SYNTHETIC_M1_B0 = -0.2
SYNTHETIC_M1_SIGMA_PER_SQRT_SECOND = 0.05


def real_fit_blockers() -> list[str]:
    """Return unresolved choices that block a primary real-data fit."""

    choices = {
        "amyloid_positivity_threshold_centiloid": AMYLOID_POSITIVITY_THRESHOLD_CENTILOID,
        "aal90_to_network9_mapping": AAL90_TO_NETWORK9_MAPPING,
        "fmri_subject_split_file": FMRI_SUBJECT_SPLIT_FILE,
        "recovery_gate_file": RECOVERY_GATE_FILE,
        "loss_weight_protocol_file": LOSS_WEIGHT_PROTOCOL_FILE,
        "hrf_protocol_file": HRF_PROTOCOL_FILE,
        "npi_jax_migration_gate_file": NPI_JAX_MIGRATION_GATE_FILE,
    }
    return [name for name, value in choices.items() if value is None]
