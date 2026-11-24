"""Run with: python -m pytest tests/"""
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
    pred = tb.predict_game("T01", "T02")
    assert 0 < pred["home_win_probability"] < 1
    assert "error" in tb.call("predict_game", {"home_team": "XXX", "away_team": "T02"})
