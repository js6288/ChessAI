from fastapi.testclient import TestClient

from chessai.server.app import create_app


def test_health_models_and_game_contract(tmp_path) -> None:
    app = create_app(model_dir=tmp_path / "models")
    with TestClient(app) as client:
        assert client.get("/api/v1/health").json()["status"] == "ok"
        models = client.get("/api/v1/models").json()["models"]
        assert models[0]["id"] == "heuristic"
        assert set(models[0]) == {
            "id",
            "name",
            "kind",
            "compatible",
            "error",
            "compatibility",
        }
        assert client.get("/api/v1/runs").status_code == 404

        response = client.post(
            "/api/v1/games",
            json={"human_side": "red", "difficulty": "beginner", "model_id": "heuristic"},
        )
        assert response.status_code == 201
        game = response.json()
        assert game["side_to_move"] == "red"
        assert len(game["legal_moves"]) == 44

        stale = client.post(
            f"/api/v1/games/{game['game_id']}/moves",
            json={"move": "h2e2", "expected_ply": 9},
        )
        assert stale.status_code == 409

        illegal = client.post(
            f"/api/v1/games/{game['game_id']}/moves",
            json={"move": "a0a9", "expected_ply": 0},
        )
        assert illegal.status_code == 422

        played = client.post(
            f"/api/v1/games/{game['game_id']}/moves",
            json={"move": "h2e2", "expected_ply": 0},
        )
        assert played.status_code == 200
        assert played.json()["ply"] == 1


def test_restart_resign_and_pgn(tmp_path) -> None:
    app = create_app(model_dir=tmp_path / "models")
    with TestClient(app) as client:
        game = client.post(
            "/api/v1/games",
            json={"human_side": "red", "difficulty": "beginner", "model_id": "heuristic"},
        ).json()
        game_id = game["game_id"]
        resigned = client.post(f"/api/v1/games/{game_id}/resign")
        assert resigned.json()["outcome"]["reason"] == "resignation"
        restarted = client.post(f"/api/v1/games/{game_id}/restart", json={})
        assert restarted.json()["ply"] == 0
        pgn = client.get(f"/api/v1/games/{game_id}/pgn")
        assert '[Game "Chinese Chess"]' in pgn.text
