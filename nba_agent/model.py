"""Models, time-aware evaluation and diagnostics."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_columns

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"

try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost is optional
    XGBClassifier = None


def make_models(seed: int = 42) -> dict:
    models = {
        # Predicts the home team's historical win rate for every game.
        "baseline_home_rate": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=seed, n_jobs=-1),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=seed)
    return models


def _scores(y, p) -> dict:
    return {
        "accuracy": accuracy_score(y, p >= 0.5),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
    }


def cross_validate(feats: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Expanding-window time-series CV: every fold trains on the past, tests on the future."""
    feats = feats.sort_values("GAME_DATE")
    X, y = feats[feature_columns(feats)].values, feats["HOME_WIN"].values
    rows = []
    for name, model in make_models().items():
        for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X)):
            model.fit(X[tr], y[tr])
            rows.append({"model": name, "fold": fold, **_scores(y[te], model.predict_proba(X[te])[:, 1])})
    res = pd.DataFrame(rows)
    summary = res.groupby("model")[["accuracy", "log_loss", "brier", "auc"]].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    return summary.sort_values("log_loss_mean")


def holdout_season(feats: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train on all earlier seasons, test on the most recent one. Saves a calibration plot."""
    test_season = feats["SEASON"].max()
    train, test = feats[feats["SEASON"] < test_season], feats[feats["SEASON"] == test_season]
    if train.empty:
        raise ValueError("Holdout needs at least two seasons of data.")
    cols = feature_columns(feats)
    rows, probs = [], {}
    for name, model in make_models().items():
        model.fit(train[cols].values, train["HOME_WIN"].values)
        p = model.predict_proba(test[cols].values)[:, 1]
        probs[name] = p
        rows.append({"model": name, **_scores(test["HOME_WIN"].values, p)})
    table = pd.DataFrame(rows).set_index("model").sort_values("log_loss")
    _plot_calibration(test["HOME_WIN"].values, probs, test_season)
    return table, {"test_season": test_season, "n_train": len(train), "n_test": len(test)}


def _plot_calibration(y, probs: dict, season: str) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    for name, p in probs.items():
        if name.startswith("baseline"):
            continue
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", label=name)
    ax.set(xlabel="Predicted home-win probability", ylabel="Observed home-win rate",
           title=f"Calibration on held-out {season} season")
    ax.legend()
    path = REPORT_DIR / "calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def data_diagnostics(feats: pd.DataFrame, model_name: str = "logistic_regression") -> dict:
    """Facts about the training data that point to concrete improvements."""
    cols = feature_columns(feats)
    test_season = feats["SEASON"].max()
    train, test = feats[feats["SEASON"] < test_season], feats[feats["SEASON"] == test_season]
    model = make_models()[model_name]
    model.fit(train[cols], train["HOME_WIN"])
    imp = permutation_importance(model, test[cols], test["HOME_WIN"], scoring="neg_log_loss",
                                 n_repeats=5, random_state=0)
    importance = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
    corr = feats[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    return {
        "n_games": len(feats),
        "games_per_season": feats.groupby("SEASON").size().to_dict(),
        "home_win_rate_by_season": feats.groupby("SEASON")["HOME_WIN"].mean().round(3).to_dict(),
        "permutation_importance_logloss": importance.round(4).to_dict(),
        "highly_correlated_feature_pairs": {f"{a} ~ {b}": round(v, 2) for (a, b), v in upper[upper > 0.85].items()},
        "features_with_near_zero_importance": importance[importance.abs() < 1e-4].index.tolist(),
    }


def _per_game_log_loss(y, p, eps: float = 1e-15) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_cv_compare(current: pd.DataFrame, trial: pd.DataFrame, n_splits: int = 5,
                      n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Compare two feature setups on the *same* games and the *same* time-series folds.

    Both setups are restricted to the games they share, sorted by date, and split
    with one TimeSeriesSplit, so every difference comes from the features, not from
    a different test set. For each model it reports:
      - mean log loss of each setup over the out-of-fold games,
      - improvement (current minus trial; positive means the trial is better),
      - how many folds the trial won,
      - a 95% bootstrap confidence interval for the per-game improvement,
      - a verdict: 'trial better', 'trial worse', or 'no clear difference'.
    """
    common = current["GAME_ID"][current["GAME_ID"].isin(trial["GAME_ID"])]
    a = current.set_index("GAME_ID").loc[common].sort_values("GAME_DATE")
    b = trial.set_index("GAME_ID").loc[a.index]
    y = a["HOME_WIN"].values
    Xa, Xb = a[feature_columns(a)].values, b[feature_columns(b)].values
    splits = list(TimeSeriesSplit(n_splits=n_splits).split(Xa))
    rng = np.random.default_rng(seed)

    rows = []
    for name in make_models():
        loss_a, loss_b, fold_wins = [], [], 0
        for tr, te in splits:
            ma, mb = make_models()[name], make_models()[name]
            ma.fit(Xa[tr], y[tr]); mb.fit(Xb[tr], y[tr])
            la = _per_game_log_loss(y[te], ma.predict_proba(Xa[te])[:, 1])
            lb = _per_game_log_loss(y[te], mb.predict_proba(Xb[te])[:, 1])
            loss_a.append(la); loss_b.append(lb)
            fold_wins += int(lb.mean() < la.mean())
        diff = np.concatenate(loss_a) - np.concatenate(loss_b)  # positive = trial better
        diff[np.abs(diff) < 1e-9] = 0.0  # ignore floating-point noise between identical predictions
        boot = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = "trial better" if lo > 0 else "trial worse" if hi < 0 else "no clear difference"
        rows.append({"model": name,
                     "current_log_loss": np.concatenate(loss_a).mean(),
                     "trial_log_loss": np.concatenate(loss_b).mean(),
                     "improvement": diff.mean(), "ci95_low": lo, "ci95_high": hi,
                     "folds_trial_won": f"{fold_wins}/{n_splits}", "verdict": verdict})
    return pd.DataFrame(rows).set_index("model")


def fit_final(feats: pd.DataFrame, model_name: str):
    model = make_models()[model_name]
    model.fit(feats[feature_columns(feats)], feats["HOME_WIN"])
    return model