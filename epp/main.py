import numpy as np
import pandas as pd
import statsmodels.api as sm

from .utils import calculate_matches_results


def calculate_epp_leaderboard(
    tournament_df: pd.DataFrame,
    player_col: str = "player",
    round_col: str = "round",
    score_col: str = "score",
    greater_score_is_better: bool = True,
    reference_player: str | None = None,
) -> pd.DataFrame:
    results_df = calculate_matches_results(
        tournament_df, player_col, round_col, score_col, greater_score_is_better
    )

    if reference_player is None:
        reference_player = results_df["winner"].iloc[0]

    endog = results_df[["n_wins", "n_loses"]]
    exog = (
        (results_df["winner"].str.get_dummies() - results_df["loser"].str.get_dummies())
        .drop(reference_player, axis=1)
        .astype(float)
    )

    model_results = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

    epps = model_results.params
    epps[reference_player] = 0.0

    p_values = model_results.pvalues
    p_values[reference_player] = np.nan

    leaderboard_df = (
        pd.concat({"epp": epps, "p_value": p_values}, axis=1)
        .sort_values("epp", ascending=False)
        .reset_index(names="player")
    )

    return leaderboard_df
