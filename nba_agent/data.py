"""Data access: pull NBA game logs from stats.nba.com via nba_api, with a local CSV cache.

Each row returned is one team's box score for one game (two rows per game),
matching the schema of nba_api's LeagueGameFinder endpoint.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"

KEEP_COLS = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP",
    "WL", "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
]


def fetch_season(season: str, pause: float = 1.0) -> pd.DataFrame:
    """Fetch regular-season team game logs for one season, e.g. '2023-24'."""
    from nba_api.stats.endpoints import leaguegamefinder

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
        timeout=60,
    )
    df = finder.get_data_frames()[0]
    time.sleep(pause)  # be polite to stats.nba.com
    return df[KEEP_COLS]


def load_games(seasons: list[str], refresh: bool = False) -> pd.DataFrame:
    """Load several seasons, using cached CSVs when available."""
    CACHE_DIR.mkdir(exist_ok=True)
    frames = []
    for season in seasons:
        path = CACHE_DIR / f"games_{season}.csv"
        if path.exists() and not refresh:
            df = pd.read_csv(path, dtype={"GAME_ID": str})
        else:
            df = fetch_season(season)
            df.to_csv(path, index=False)
        df["SEASON"] = season
        frames.append(df)
    games = pd.concat(frames, ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["GAME_ID"] = games["GAME_ID"].astype(str)
    return games.dropna(subset=["WL"]).sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def make_synthetic_games(seasons: list[str], n_teams: int = 30, seed: int = 0) -> pd.DataFrame:
    """Generate fake games in the same schema, for testing the pipeline offline.

    Teams get a hidden strength that drifts over time, plus home-court advantage,
    so models have real (but noisy) signal to learn. Never report results from this data.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    rows, gid = [], 0
    for s_idx, season in enumerate(seasons):
        strength = rng.normal(0, 4, n_teams)
        start = pd.Timestamp(f"{2020 + s_idx}-10-20")
        for day in range(170):
            date = start + pd.Timedelta(days=day)
            strength += rng.normal(0, 0.15, n_teams)
            order = rng.permutation(n_teams)[: 2 * rng.integers(3, 8)]
            for h, a in zip(order[::2], order[1::2]):
                margin = strength[h] - strength[a] + 2.5 + rng.normal(0, 12)
                base = rng.normal(112, 8)
                pts = {h: int(base + margin / 2), a: int(base - margin / 2)}
                if pts[h] == pts[a]:  # NBA games can't end tied
                    pts[h] += 1
                gid += 1
                for t, opp, home in ((h, a, True), (a, h, False)):
                    fga = int(rng.normal(88, 5)); fgm = int(fga * rng.normal(0.47, 0.03))
                    f3a = int(rng.normal(35, 5)); f3m = int(f3a * rng.normal(0.36, 0.04))
                    rows.append({
                        "SEASON_ID": f"2{2020 + s_idx}", "TEAM_ID": 1610612700 + t,
                        "TEAM_ABBREVIATION": teams[t], "GAME_ID": f"{gid:010d}",
                        "GAME_DATE": date,
                        "MATCHUP": f"{teams[t]} vs. {teams[opp]}" if home else f"{teams[t]} @ {teams[opp]}",
                        "WL": "W" if pts[t] > pts[opp] else "L", "PTS": pts[t],
                        "FGM": fgm, "FGA": fga, "FG_PCT": fgm / fga, "FG3M": f3m, "FG3A": f3a,
                        "FG3_PCT": f3m / max(f3a, 1), "FTM": 17, "FTA": 22,
                        "OREB": int(rng.normal(10, 3)), "DREB": int(rng.normal(34, 4)),
                        "REB": 44, "AST": int(rng.normal(25, 4)), "STL": int(rng.normal(7, 2)),
                        "BLK": int(rng.normal(5, 2)), "TOV": int(rng.normal(13, 3)), "PF": 19,
                        "PLUS_MINUS": pts[t] - pts[opp], "SEASON": season,
                    })
    df = pd.DataFrame(rows)
    return df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
