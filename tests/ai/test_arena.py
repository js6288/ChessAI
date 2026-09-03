from chessai.training.arena import RandomPlayer, quick_opening_fens, run_arena


def test_quick_openings_are_unique_legal_positions() -> None:
    openings = quick_opening_fens(10)
    assert len(openings) == len(set(openings)) == 10


def test_lightweight_arena_reports_only_product_metrics() -> None:
    summary, games = run_arena(
        RandomPlayer(1),
        RandomPlayer(2),
        games=2,
        opening_fens=quick_opening_fens(1),
        max_ply=2,
    )
    assert len(games) == 2
    assert summary.games == 2
    assert summary.wins + summary.draws + summary.losses == 2
    assert 0.0 <= summary.score_rate <= 1.0
