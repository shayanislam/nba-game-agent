"""Run with: python -m pytest tests/"""
import pandas as pd

from nba_agent import data, features
from nba_agent.agent import Toolbox

SEASONS = ["2022-23", "2023-24"]


def test_no_leakage():
    """Changing a game's own result must not change that game's features."""
    games = data.make_synthetic_games(SEASONS, seed=1)
    before = features.build_game_features(games).set_index("GAME_ID")
    gid = before.index[len(before) // 2]
    flipped = games.copy()
    rows = flipped["GAME_ID"] == gid
    flipped.loc[rows, "WL"] = flipped.loc[rows, "WL"].map({"W": "L", "L": "W"})
    flipped.loc[rows, ["PTS", "PLUS_MINUS"]] *= -1
    after = features.build_game_features(flipped).set_index("GAME_ID")
    cols = features.feature_columns(before)
    assert (before.loc[gid, cols] == after.loc[gid, cols]).all()
    assert before.loc[gid, "HOME_WIN"] != after.loc[gid, "HOME_WIN"]


def test_tools_run():
    tb = Toolbox(synthetic=True)
    assert tb.load_data(SEASONS)["games_with_features"] > 500
    exp = tb.run_experiment(window=5)
    assert "comparison" in exp
    pred = tb.predict_game("T01", "T02", home_rest=2, away_rest=1)
    assert 0 < pred["home_win_probability"] < 1
    assert pred["rest"]["away"]["back_to_back"]
    assert "error" in tb.call("predict_game", {"home_team": "XXX", "away_team": "T02"})
    assert "error" in tb.call("predict_game", {"home_team": "T01", "away_team": "T02"})  # no date, no rest


def test_paired_compare():
    """Identical setups must tie; removing every useful feature must lose clearly."""
    from nba_agent import model
    feats = features.build_game_features(data.make_synthetic_games(SEASONS, seed=3))
    same = model.paired_cv_compare(feats, feats.copy())
    assert (same["improvement"].abs() < 1e-9).all()
    assert (same.drop(index="baseline_home_rate")["verdict"] == "no clear difference").all()
    keep = ["GAME_ID", "GAME_DATE", "SEASON", "HOME_WIN", "HOME_REST", "AWAY_REST"]
    gutted = feats[keep]
    worse = model.paired_cv_compare(feats, gutted)
    assert worse.loc["logistic_regression", "verdict"] == "trial worse"


def test_rest_computed_from_schedule():
    """With a game date, rest = days since each team's last game before that date."""
    games = data.make_synthetic_games(SEASONS, seed=2)
    team = "T05"
    dates = sorted(games.loc[games["TEAM_ABBREVIATION"] == team, "GAME_DATE"].unique())
    last, target = pd.Timestamp(dates[-5]), pd.Timestamp(dates[-5]) + pd.Timedelta(days=1)
    games = games[~((games["GAME_DATE"] > last) & (games["GAME_DATE"] <= target))]  # nothing in between
    _, info = features.matchup_features(games, team, "T06", game_date=str(target.date()))
    assert info["home"]["rest_days"] == 1
    assert info["home"]["back_to_back"]
    assert info["home"]["source"] == "computed from schedule"
    # A user-provided value overrides the schedule.
    _, info = features.matchup_features(games, team, "T06", home_rest=3, game_date=str(target.date()))
    assert info["home"]["rest_days"] == 3 and info["home"]["source"] == "provided by user"