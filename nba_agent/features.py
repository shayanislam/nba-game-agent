"""Feature engineering.

Every feature describes a team *before* tip-off: rolling stats are shifted by one
game, so a game's own result never leaks into its features.
"""
from __future__ import annotations

import pandas as pd

ROLL_STATS = ["PTS", "OPP_PTS", "PLUS_MINUS", "FG_PCT", "FG3_PCT", "TOV", "REB", "AST", "WIN"]


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Add opponent points, home flag and win flag to the team-game rows."""
    df = games.copy()
    opp = df[["GAME_ID", "TEAM_ID", "PTS"]].rename(columns={"TEAM_ID": "OPP_ID", "PTS": "OPP_PTS"})
    df = df.merge(opp, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["OPP_ID"]]
    df["HOME"] = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    df["WIN"] = (df["WL"] == "W").astype(int)
    return df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)


def _rolling_form(df: pd.DataFrame, window: int, shift: int) -> pd.DataFrame:
    """Rolling means and schedule features per team within each season.

    shift=1 gives pre-game features (for training); shift=0 gives the form
    *after* the latest game (for predicting the next one).
    """
    g = df.groupby(["TEAM_ID", "SEASON"], group_keys=False)
    out = df[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "SEASON", "GAME_DATE", "HOME", "WIN"]].copy()
    for col in ROLL_STATS:
        out[f"R{window}_{col}"] = g[col].transform(
            lambda s: s.shift(shift).rolling(window, min_periods=1).mean())
    out["SEASON_WIN_PCT"] = g["WIN"].transform(lambda s: s.shift(shift).expanding().mean())
    out["GAMES_PLAYED"] = g.cumcount() + (1 - shift)
    rest = g["GAME_DATE"].transform(lambda s: s.diff().dt.days)
    out["REST_DAYS"] = rest.fillna(7).clip(upper=7)
    out["BACK_TO_BACK"] = (out["REST_DAYS"] == 1).astype(int)
    return out


def build_game_features(games: pd.DataFrame, window: int = 10, min_games: int = 5) -> pd.DataFrame:
    """One row per game: home-minus-away differences plus each side's rest.

    Target column HOME_WIN is 1 if the home team won.
    Games where either team has played fewer than `min_games` that season are
    dropped, because early-season rolling averages are mostly noise.
    """
    form = _rolling_form(_team_games(games), window, shift=1)
    home = form[form["HOME"] == 1].set_index("GAME_ID")
    away = form[form["HOME"] == 0].set_index("GAME_ID")
    common = home.index.intersection(away.index)
    home, away = home.loc[common], away.loc[common]

    feats = pd.DataFrame(index=common)
    feats["GAME_DATE"] = home["GAME_DATE"]
    feats["SEASON"] = home["SEASON"]
    feats["HOME_TEAM"] = home["TEAM_ABBREVIATION"]
    feats["AWAY_TEAM"] = away["TEAM_ABBREVIATION"]
    for col in [f"R{window}_{c}" for c in ROLL_STATS] + ["SEASON_WIN_PCT"]:
        feats[f"DIFF_{col}"] = home[col] - away[col]
    feats["HOME_REST"] = home["REST_DAYS"]
    feats["AWAY_REST"] = away["REST_DAYS"]
    feats["HOME_B2B"] = home["BACK_TO_BACK"]
    feats["AWAY_B2B"] = away["BACK_TO_BACK"]
    feats["HOME_WIN"] = home["WIN"]

    enough = (home["GAMES_PLAYED"] >= min_games) & (away["GAMES_PLAYED"] >= min_games)
    feats = feats[enough].dropna()
    return feats.sort_values("GAME_DATE").reset_index(names="GAME_ID")


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c.startswith(("DIFF_", "HOME_REST", "AWAY_REST", "HOME_B2B", "AWAY_B2B"))]


def matchup_features(games: pd.DataFrame, home_team: str, away_team: str,
                     home_rest: int | None = None, away_rest: int | None = None,
                     window: int = 10, game_date: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Features for an upcoming game, from each team's form after its latest game.

    If `game_date` is given, only games before that date are used, and any rest
    value left as None is computed from the team's last game before that date.
    Rest is days between games, so 1 means a back-to-back. Returns the feature
    row and a dict describing where the rest values came from.
    """
    if game_date is not None:
        cutoff = pd.Timestamp(game_date)
        games = games[games["GAME_DATE"] < cutoff]
    form = _rolling_form(_team_games(games), window, shift=0)
    latest = form[form["SEASON"] == form["SEASON"].max()]
    latest = latest.sort_values("GAME_DATE").groupby("TEAM_ABBREVIATION").tail(1).set_index("TEAM_ABBREVIATION")
    for team in (home_team, away_team):
        if team not in latest.index:
            raise ValueError(f"Unknown team '{team}'. Known: {', '.join(sorted(latest.index))}")
    h, a = latest.loc[home_team], latest.loc[away_team]

    info = {}
    rests = {}
    for side, team, given, row_ in (("home", home_team, home_rest, h), ("away", away_team, away_rest, a)):
        last = row_["GAME_DATE"].date()
        if given is not None:
            rests[side] = int(given)
            info[side] = {"team": team, "rest_days": rests[side], "source": "provided by user",
                          "last_game_in_data": str(last)}
        elif game_date is not None:
            rests[side] = int((cutoff - row_["GAME_DATE"]).days)
            info[side] = {"team": team, "rest_days": rests[side], "source": "computed from schedule",
                          "last_game_in_data": str(last)}
        else:
            raise ValueError("Give a game_date so rest can be computed, or pass home_rest and away_rest.")

    row = {f"DIFF_{c}": h[c] - a[c] for c in [f"R{window}_{s}" for s in ROLL_STATS] + ["SEASON_WIN_PCT"]}
    row.update(HOME_REST=min(rests["home"], 7), AWAY_REST=min(rests["away"], 7),
               HOME_B2B=int(rests["home"] == 1), AWAY_B2B=int(rests["away"] == 1))
    for side in ("home", "away"):
        info[side]["back_to_back"] = rests[side] == 1
    if game_date is not None and max(rests.values()) > 10:
        info["warning"] = ("A team's last game in the data is more than 10 days before this date. "
                           "The current season may not be loaded, so form and rest may be stale.")
    return pd.DataFrame([row]), info