import numpy as np
import pandas as pd
import statsmodels.api as sm


def calculate_epp_leaderboard(
    tournament_df: pd.DataFrame,
    player_col: str = "player",
    round_col: str = "round",
    score_col: str = "score",
    greater_score_is_better: bool = True,
    reference_player: str | None = None,
) -> pd.DataFrame:
    tournament_df = tournament_df[[player_col, round_col, score_col]].sort_values(
        [round_col, player_col]
    )

    all_unique_players = tournament_df[player_col].unique()
    each_round_players = tournament_df.groupby(round_col)[player_col]

    if any(each_round_players.unique().apply(set) != set(all_unique_players)):
        raise ValueError("Some players did not score in every round.")
    if any(each_round_players.value_counts() > 1):
        raise ValueError("Some players scored more than once in a round.")

    comparison_operator = np.greater if greater_score_is_better else np.less

    if reference_player is not None:
        reference_mask = all_unique_players == reference_player
        if not any(reference_mask):
            raise ValueError(f"Player '{reference_player}' not found.")
        reference_index = np.arange(n_players)[reference_mask][0]
    else:
        reference_index = 0

    n_players = len(all_unique_players)
    n_rounds = len(each_round_players)

    scores = tournament_df[score_col].to_numpy(dtype=float).reshape((1, n_rounds, -1))
    n_wins = comparison_operator(scores.T, scores).sum(axis=1, dtype=float)

    # adjust for ties
    n_wins += (n_rounds - (n_wins + n_wins.T)) / 2

    endog = np.delete(
        np.column_stack([n_wins.flatten(), n_rounds - n_wins.flatten()]),
        np.arange(n_players) * (n_players + 1),
        axis=0,
    )

    exog = np.delete(
        np.delete(
            np.repeat(np.eye(n_players), n_players, axis=0)
            - np.tile(np.eye(n_players), (n_players, 1)),
            np.arange(n_players) * (n_players + 1),
            axis=0,
        ),
        reference_index,
        axis=1,
    )

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    fit_results = model.fit()

    leaderboard_df = pd.DataFrame(
        {
            "player": all_unique_players,
            "epp": np.insert(fit_results.params, reference_index, 0.0),
            "p_value": np.insert(fit_results.pvalues, reference_index, np.nan),
        }
    ).sort_values("epp", ascending=False, ignore_index=True)

    return leaderboard_df
