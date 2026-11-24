"""Command-line entry point.

  python main.py evaluate              # run the full pipeline, no LLM needed
  python main.py agent                 # chat with the agent (needs ANTHROPIC_API_KEY)
  add --synthetic to either to test offline with fake data
"""
import argparse

import pandas as pd

from nba_agent.agent import DEFAULT_SEASONS, Toolbox, run_agent

pd.set_option("display.width", 140)


def evaluate(tb: Toolbox, seasons):
    info = tb.load_data(seasons)
    print(f"Loaded {info['games_with_features']} games (synthetic={info['synthetic_data']}), "
          f"home win rate {info['home_win_rate']}\n")
    print("Time-series cross-validation (mean over folds):")
    cv = pd.DataFrame(tb.run_cross_validation()).T
    print(cv[[c for c in cv.columns if c.endswith("_mean")]].round(4), "\n")
    hold = tb.evaluate_holdout_season()
    print(f"Holdout: train {hold['n_train']} games, test {hold['n_test']} games ({hold['test_season']})")
    print(pd.DataFrame(hold["results"]).T.round(4), "\n")
    diag = tb.diagnose_data()
    print("Top features by permutation importance (log loss):")
    for k, v in list(diag["permutation_importance_logloss"].items())[:6]:
        print(f"  {k:28s} {v:.4f}")
    print(f"\nCalibration plot saved to {hold['calibration_plot']}")


def chat(tb: Toolbox):
    print("NBA agent ready. Ask a question, or 'quit'.")
    history = []
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        answer, history = run_agent(q, tb, history)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["evaluate", "agent"])
    ap.add_argument("--synthetic", action="store_true", help="use fake data (offline testing only)")
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = ap.parse_args()
    tb = Toolbox(synthetic=args.synthetic)
    evaluate(tb, args.seasons) if args.command == "evaluate" else chat(tb)
