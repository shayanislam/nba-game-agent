NBA Game Agent

An AI agent that predicts NBA game outcomes by calling a real machine learning pipeline as tools. It pulls game data from the NBA stats API, engineers pre-game features, evaluates models with time-aware validation, tests its own ideas for improving the training data, and explains each prediction along with its uncertainty.

Why this project

Sports data makes it easy to fool yourself. A random train/test split lets a model "see the future," and raw accuracy hides poorly calibrated probabilities. This project is built around avoiding both:

No leakage: every feature is computed from games played before tip-off (verified by a test in tests/test_pipeline.py).
Time-aware evaluation: expanding-window time-series cross-validation, plus a holdout where models train on earlier seasons and are tested on an unseen season.
Honest metrics: log loss, Brier score, AUC and a calibration plot, always compared against a baseline that simply predicts the home team's historical win rate.
How it works
Step	File	What happens
Data	nba_agent/data.py	Pulls regular-season team game logs with nba_api, cached as CSV
Features	nba_agent/features.py	Rolling form over the last 10 games (points, points allowed, +/-, shooting, turnovers, rebounds, assists, wins), season win %, rest days and back-to-backs, as home-minus-away differences
Models	nba_agent/model.py	Baseline, logistic regression, random forest and XGBoost; time-series CV, season holdout, permutation importance, calibration plot
Agent	nba_agent/agent.py	LLM with tool calling (Claude API). Tools: load_data, run_cross_validation, evaluate_holdout_season, diagnose_data, run_experiment, predict_game

The agent never invents numbers. It answers by calling tools, and its run_experiment tool lets it test data changes (rolling window size, minimum games played, dropping features) against the current setup before recommending them.

Results

Holdout: trained on the 2022-23 and 2023-24 seasons (2,300 games), tested on the unseen 2024-25 season (1,148 games).

Model	Accuracy	Log loss	Brier	AUC
Baseline (home win rate)	0.550	0.688	0.248	0.500
Logistic regression	0.673	0.602	0.208	0.731
Random forest	0.660	0.615	0.213	0.718
XGBoost	0.653	0.622	0.216	0.706

Time-series cross-validation gave the same ranking (logistic regression: 0.649 accuracy, 0.620 log loss, 0.707 AUC).

Key findings

The best model beats the home-court baseline by about 12 points of accuracy, in line with published NBA prediction models.
The simplest model performed best. With noisy outcomes and a few thousand games, the tree models gained nothing from their extra flexibility.
Season win % was by far the strongest predictor, followed by recent form (last-10 wins and +/-). An away team on a back-to-back was a measurable disadvantage.
Calibration

Calibration

All three models are well calibrated on the unseen 2024-25 season: predicted probabilities closely match observed home-win rates, with logistic regression slightly overconfident at the high end (~86% predicted vs. ~81% observed). No model predicts below ~20% or above ~88%, which is appropriate for a sport with frequent upsets.

Agent-driven data experiments

Asked how to improve the training data, the agent diagnosed feature importance and correlations, then ran 8 experiments (rolling windows of 5 to 30 games, minimum games played, dropping low-value features). The best holdout result, a 20-game window with 5 features dropped, looked like an improvement. The agent then checked it with time-series cross-validation and found the gain (about 0.004 log loss) was well within fold-to-fold noise (±0.02 to 0.05).

Conclusion: the simpler 9-feature setup performs as well as the original 14 features, but is not measurably better. Choosing a winner on the same data used to test it can overstate an improvement, so every apparent gain should be confirmed with cross-validation.

Setup
bash
pip install -r requirements.txt

Run the full pipeline (no API key needed):

bash
python main.py evaluate

Chat with the agent:

bash
# macOS / Linux
export ANTHROPIC_API_KEY=your-key
# Windows PowerShell
$env:ANTHROPIC_API_KEY="your-key"

python main.py agent

Example questions:

"Evaluate the models. Which one should I trust?"
"How could the training data be improved? Test your ideas."
"Load the 2024-25, 2025-26 and 2026-27 seasons, then predict LAL at SAS. The Spurs last played yesterday and the Lakers have had one day off."

To predict current games, load the current season so each team's form is up to date. Rest is counted as days between games: 1 means a back-to-back.

--synthetic runs everything on generated fake data, for testing without network access. Never report results from synthetic data.

Tests
bash
python -m pytest tests/
Limitations
No injury, lineup or betting-line data, which are among the strongest real predictors.
Rolling averages reset each season, so the first few games of each season are dropped, and early-season predictions rest on few games.
Single NBA games are noisy. Even a 65% favorite loses about one game in three.