import numpy as np
import pandas as pd


def calculate_matches_results(
    tournament_df: pd.DataFrame,
    player_col: str = "player",
    round_col: str = "round",
    score_col: str = "score",
    greater_score_is_better: bool = True,
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

    n_rounds = tournament_df[round_col].nunique()
    comparison_operator = np.greater if greater_score_is_better else np.less

    scores = tournament_df[score_col].to_numpy(dtype=float).reshape((1, n_rounds, -1))
    n_wins = comparison_operator(scores.T, scores).sum(axis=1, dtype=float)

    # adjust for ties
    n_wins += (n_rounds - (n_wins.T + n_wins)) / 2

    results_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            2 * [all_unique_players], names=["winner", "loser"]
        )
    )

    results_df["n_wins"] = n_wins.flatten()
    results_df["n_loses"] = n_rounds - n_wins.flatten()

    # drop matches with self
    results_df = results_df.drop(
        [2 * (player,) for player in all_unique_players]
    ).reset_index()

    return results_df
