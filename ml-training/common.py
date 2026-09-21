"""Shared code for the training scripts and the inference service."""

from pathlib import Path

VALIDATION_SEASON = 2023
TEST_SEASONS = (2024, 2025)

TRACKING_DIR = Path(__file__).resolve().parent / "mlruns"
TRACKING_URI = f"sqlite:///{(TRACKING_DIR / 'mlflow.db').as_posix()}"
ARTIFACT_URI = (TRACKING_DIR / "artifacts").as_uri()

PER_SIDE_FEATURES = [
    "ROLL5_WIN_PCT",
    "ROLL5_PTS",
    "ROLL5_PLUS_MINUS",
    "ROLL5_FG_PCT",
    "ROLL5_REB",
    "ROLL5_AST",
    "ROLL5_TOV",
    "ROLL10_WIN_PCT",
    "ROLL10_PTS",
    "ROLL10_PLUS_MINUS",
    "ROLL10_FG_PCT",
    "ROLL10_REB",
    "ROLL10_AST",
    "ROLL10_TOV",
    "REST_DAYS",
    "IS_BACK_TO_BACK",
    "TEAM_ELO",
    "ABSENT_COUNT",
    "WEIGHTED_ABSENT_MIN",
]

# The contract between training and serving. Changing this list means
# retraining every model and refreshing their frozen tree counts.
FEATURE_COLUMNS = [f"{side}_{feat}" for side in ("HOME", "AWAY") for feat in PER_SIDE_FEATURES]

ROLLING_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if "ROLL5_" in c or "ROLL10_" in c]

def setup_mlflow(experiment_name: str):
    """Point MLflow at the SQLite store, creating the experiment if needed."""
    import mlflow

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(experiment_name)

def split_three_way(df):
    """Chronological train / validation / test split by season."""
    train = df[df["SEASON"] < VALIDATION_SEASON]
    validation = df[df["SEASON"] == VALIDATION_SEASON]
    test = df[df["SEASON"].isin(TEST_SEASONS)]

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            f"Empty split - seasons present: {sorted(df['SEASON'].unique())}, "
            f"expected data before {VALIDATION_SEASON} and in {TEST_SEASONS}."
        )

    train_seasons = sorted(train["SEASON"].unique())
    print(f"Train:      seasons {train_seasons[0]}-{train_seasons[-1]} ({len(train)} games)")
    print(f"Validation: season  {VALIDATION_SEASON} ({len(validation)} games)")
    print(f"Test:       seasons {TEST_SEASONS[0]}-{TEST_SEASONS[-1]} ({len(test)} games)")

    return train, validation, test
