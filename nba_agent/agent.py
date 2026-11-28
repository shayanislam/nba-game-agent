"""An LLM agent that runs the pipeline through tool calls.

The model never sees raw data or invents numbers: it calls tools that load data,
evaluate models, run experiments and make predictions, then explains the results.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from . import data, features, model

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are a careful sports data scientist. You answer questions about NBA games \
by calling tools, never by guessing. Rules:
- Load data before anything else if it is not loaded yet.
- Always compare models against the baseline_home_rate model, and judge them mainly by log loss \
and calibration, not accuracy alone.
- When asked how to improve the data, call diagnose_data, then test concrete changes with \
run_experiment before recommending them. Only call a change an improvement when its verdict is \
'trial better' (the 95% confidence interval excludes zero). Treat 'no clear difference' as noise, \
and say so. Report what actually helped and what did not.
- To predict a game, make sure the season containing the game date is loaded, and pass game_date \
so rest days are computed from the schedule. Report each team's rest and whether either is on a \
back-to-back. If the tool returns a warning, tell the user.
- When predicting a game, give the probability, how confident the model is, and the main factors. \
Remind the user that single NBA games are highly uncertain.
- Be concise. Use numbers from tool results only."""

TOOLS = [
    {
        "name": "load_data",
        "description": ("Load NBA regular-season team game logs and build pre-game features. "
                        "The settings given here become the active setup used by every other tool, "
                        "so call it again to adopt settings that run_experiment showed were better."),
        "input_schema": {
            "type": "object",
            "properties": {
                "seasons": {"type": "array", "items": {"type": "string"},
                            "description": "Seasons like '2022-23'. Default: the last three completed seasons."},
                "window": {"type": "integer", "description": "Rolling window in games. Default 10."},
                "min_games": {"type": "integer", "description": "Minimum games played before a game is used. Default 5."},
                "drop_features": {"type": "array", "items": {"type": "string"},
                                  "description": "Feature columns to leave out of every model."},
            },
        },
    },
    {
        "name": "run_cross_validation",
        "description": "Expanding-window time-series cross-validation of every model.",
        "input_schema": {"type": "object", "properties": {"n_splits": {"type": "integer"}}},
    },
    {
        "name": "evaluate_holdout_season",
        "description": "Train on earlier seasons, test on the latest season; saves reports/calibration.png.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "diagnose_data",
        "description": "Feature importance, correlated features, class balance and season drift.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_experiment",
        "description": ("Rebuild features with different settings and compare them with the current setup "
                        "using paired time-series cross-validation (same games, same folds). Returns each "
                        "model's log loss change, folds won, a 95% confidence interval and a verdict. "
                        "Use to test data improvements."),
        "input_schema": {
            "type": "object",
            "properties": {
                "window": {"type": "integer", "description": "Rolling window in games."},
                "min_games": {"type": "integer", "description": "Minimum games played before a game is used."},
                "drop_features": {"type": "array", "items": {"type": "string"},
                                  "description": "Feature columns to remove."},
            },
        },
    },
    {
        "name": "predict_game",
        "description": ("Home-win probability for an upcoming game from each team's form before it. "
                        "Prefer passing game_date: rest days and back-to-backs are then computed from "
                        "each team's last game in the data. Only pass rest values if the user gives them."),
        "input_schema": {
            "type": "object",
            "properties": {
                "home_team": {"type": "string", "description": "Team abbreviation, e.g. 'LAL'."},
                "away_team": {"type": "string"},
                "game_date": {"type": "string", "description": "Date of the game, YYYY-MM-DD."},
                "home_rest": {"type": "integer",
                              "description": "Optional override: days since home team's last game (1 = back-to-back)."},
                "away_rest": {"type": "integer", "description": "Optional override, same as home_rest."},
                "model_name": {"type": "string", "description": "Default: logistic_regression."},
            },
            "required": ["home_team", "away_team"],
        },
    },
]

DEFAULT_SEASONS = ["2022-23", "2023-24", "2024-25"]


class Toolbox:
    """Holds loaded data and implements each tool."""

    def __init__(self, synthetic: bool = False):
        self.synthetic = synthetic
        self.games: pd.DataFrame | None = None
        self.feats: pd.DataFrame | None = None
        self.window = 10
        self.min_games = 5
        self.dropped: list[str] = []

    def _require(self):
        if self.feats is None:
            self.load_data()

    def _active(self, feats: pd.DataFrame, dropped: list[str]) -> pd.DataFrame:
        return feats.drop(columns=[c for c in dropped if c in feats.columns])

    def load_data(self, seasons: list[str] | None = None, window: int = 10, min_games: int = 5,
                  drop_features: list[str] | None = None) -> dict:
        seasons = seasons or DEFAULT_SEASONS
        self.games = (data.make_synthetic_games(seasons) if self.synthetic
                      else data.load_games(seasons))
        self.window, self.min_games = window, min_games
        # Rolling feature names include the window (R10_, R20_...), so match them to this window.
        self.dropped = [re.sub(r"R\d+_", f"R{window}_", c) for c in (drop_features or [])]
        self.feats = features.build_game_features(self.games, window, min_games)
        active = features.feature_columns(self._active(self.feats, self.dropped))
        return {"seasons": seasons, "synthetic_data": self.synthetic, "games_with_features": len(self.feats),
                "home_win_rate": round(self.feats["HOME_WIN"].mean(), 3),
                "window": window, "min_games": min_games, "dropped": self.dropped, "active_features": active}

    def run_cross_validation(self, n_splits: int = 5) -> dict:
        self._require()
        return model.cross_validate(self._active(self.feats, self.dropped), n_splits).round(4).to_dict("index")

    def evaluate_holdout_season(self) -> dict:
        self._require()
        table, info = model.holdout_season(self._active(self.feats, self.dropped))
        return {**info, "results": table.round(4).to_dict("index"), "calibration_plot": "reports/calibration.png"}

    def diagnose_data(self) -> dict:
        self._require()
        return model.data_diagnostics(self._active(self.feats, self.dropped))

    def run_experiment(self, window: int | None = None, min_games: int | None = None,
                       drop_features: list[str] | None = None) -> dict:
        self._require()
        window, min_games = window or self.window, min_games or self.min_games
        # Rolling columns are named by window, so map dropped names onto the new window.
        dropped = [re.sub(r"R\d+_", f"R{window}_", c) for c in (drop_features or [])]
        trial_feats = features.build_game_features(self.games, window, min_games)
        compare = model.paired_cv_compare(self._active(self.feats, self.dropped),
                                          self._active(trial_feats, dropped))
        return {"current_settings": {"window": self.window, "min_games": self.min_games, "dropped": self.dropped},
                "trial_settings": {"window": window, "min_games": min_games, "dropped": dropped},
                "method": ("Paired time-series cross-validation: both setups are scored on the same games "
                           "and folds. improvement = current minus trial log loss (positive = trial better). "
                           "The 95% CI is a bootstrap over per-game differences. 'no clear difference' "
                           "means the CI includes zero."),
                "comparison": compare.round(4).to_dict("index")}

    def predict_game(self, home_team: str, away_team: str, game_date: str | None = None,
                     home_rest: int | None = None, away_rest: int | None = None,
                     model_name: str = "logistic_regression") -> dict:
        self._require()
        feats = self._active(self.feats, self.dropped)
        if game_date is not None:
            # Train only on games before the date, so past dates can be backtested without leakage.
            feats = feats[feats["GAME_DATE"] < pd.Timestamp(game_date)]
        fitted = model.fit_final(feats, model_name)
        x, rest_info = features.matchup_features(self.games, home_team.upper(), away_team.upper(),
                                                 home_rest, away_rest, self.window, game_date)
        x = x[features.feature_columns(feats)]
        p = float(fitted.predict_proba(x)[0, 1])
        return {"home_team": home_team.upper(), "away_team": away_team.upper(), "game_date": game_date,
                "model": model_name, "home_win_probability": round(p, 3), "rest": rest_info,
                "inputs": x.round(3).iloc[0].to_dict()}

    def call(self, name: str, args: dict) -> str:
        try:
            result = getattr(self, name)(**args)
        except Exception as e:  # report tool errors back to the model
            result = {"error": f"{type(e).__name__}: {e}"}
        return json.dumps(result, default=str)


def run_agent(question: str, toolbox: Toolbox, history: list | None = None,
              model_id: str = DEFAULT_MODEL, verbose: bool = True) -> tuple[str, list]:
    """Run one user turn through the tool-use loop. Returns (answer, updated history)."""
    import anthropic

    client = anthropic.Anthropic()
    messages = (history or []) + [{"role": "user", "content": question}]
    while True:
        resp = client.messages.create(model=model_id, max_tokens=2000, system=SYSTEM_PROMPT,
                                      tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            answer = "".join(b.text for b in resp.content if b.type == "text")
            return answer, messages
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                if verbose:
                    print(f"  [tool] {block.name}({json.dumps(block.input)})")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": toolbox.call(block.name, block.input)})
        messages.append({"role": "user", "content": results})