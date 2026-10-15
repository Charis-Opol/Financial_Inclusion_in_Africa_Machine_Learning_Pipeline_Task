import numpy as np
import pytest

from fin_inclusion.interpretability.plots import (
    plot_attribution_heatmap,
    plot_embedding_projection,
    plot_interaction_facets,
    plot_matrix_panels,
    plot_rank_table,
    plot_share_grid,
)


def test_embedding_rows_map_every_raw_level_to_a_trained_row(mlp_fitted):
    mlp = mlp_fitted[0]

    for column in mlp.categorical_columns:
        rows = mlp.embedding_rows(column)
        assert set(rows) == set(mlp.categorical_levels[column])
        assert 0 not in rows.values()  # row 0 is the never-seen bucket
        assert max(rows.values()) < mlp.net.embeddings[mlp.categorical_columns.index(column)].num_embeddings


def test_phase5_plots_write_non_empty_files(tmp_path):
    rng = np.random.default_rng(0)
    names = ["country", "job_type", "age"]
    shares = {f"method {k}": rng.dirichlet(np.ones(3)) for k in range(4)}
    ranks = np.column_stack([(-v).argsort().argsort() + 1 for v in shares.values()]).astype(float)

    plot_share_grid(shares, names, [0, 1, 2], tmp_path / "grid.png", "t", "caption.", ncols=2)
    plot_rank_table(ranks, np.column_stack(list(shares.values())), names, list(shares), tmp_path / "ranks.png",
                    "t", "caption.")
    plot_attribution_heatmap(rng.normal(size=(60, 3)), names, rng.random(60), rng.integers(0, 2, 60), 0.5, 1.0,
                             tmp_path / "heat.png", "t", "caption.")
    plot_matrix_panels({"pooled": rng.normal(size=(3, 3)), "uganda": rng.normal(size=(3, 3))}, names,
                       ["TP", "TN", "TP - TN"], tmp_path / "matrix.png", "t", "caption.", diverging=True, vmax=2.0)
    panel = {"coords": rng.normal(size=(4, 2)), "labels": list("abcd"), "colour": rng.normal(size=4),
             "unknown": np.zeros(2), "explained": 0.8}
    plot_embedding_projection({"country": panel, "job_type": panel}, 1.0, tmp_path / "embed.png", "t", "caption.")
    plot_interaction_facets("job", rng.choice(["a", "b", "c"], 90), rng.normal(size=90), "country",
                            rng.choice(["Kenya", "Uganda"], 90), tmp_path / "facets.png", "t", "caption.")

    for name in ("grid", "ranks", "heat", "matrix", "embed", "facets"):
        assert (tmp_path / f"{name}.png").stat().st_size > 5000


def test_matrix_panels_leave_missing_cells_blank_instead_of_failing(tmp_path):
    matrix = np.array([[np.nan, 0.2], [0.2, np.nan]])

    plot_matrix_panels({"m": matrix}, ["a", "b"], ["a", "b"], tmp_path / "m.png", "t", "caption.",
                       diverging=False, vmax=0.3, fmt="{:.3f}")

    assert (tmp_path / "m.png").exists()


@pytest.mark.parametrize("column", ["country", "job_type"])
def test_embedding_projection_inputs_come_from_the_learned_table(mlp_fitted, column):
    """The projection reads the trained embedding rows; check they differ from
    the untrained unknown row's position in the table (row 0)."""
    mlp = mlp_fitted[0]
    weights = mlp.net.embeddings[mlp.categorical_columns.index(column)].weight.detach().numpy()
    rows = sorted(mlp.embedding_rows(column).values())

    assert weights[rows].shape == (len(rows), weights.shape[1])
    assert not np.allclose(weights[rows], weights[0])
