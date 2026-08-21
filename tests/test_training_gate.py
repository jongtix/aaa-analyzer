"""맥 측 게이트 CLI — 챔피언 경로 해석 + 동결 하이퍼파라미터 리더 + verdict
직렬화 테스트 (SPEC-ANALYZER-TRAIN-GATE-001 M1, REQ-ATG-008/009/010).
"""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from analyzer.orchestration import activation as activation_module
from analyzer.orchestration.promotion_gate import PromotionVerdict
from analyzer.training import campaign_metrics as campaign_metrics_module
from analyzer.training import gate as gate_module
from analyzer.training import persistence as persistence_module
from analyzer.training.gate import (
    deserialize_verdicts,
    parse_args,
    read_frozen_hyperparameters,
    resolve_champion_model_paths,
    run_gate,
    serialize_verdicts,
    warn_dangling_champions,
)


def _write_champion_manifest(
    models_root: Path, market: str, horizon: int, algorithm: str, trained_date: date
) -> None:
    activation_module.write_activation_manifest(
        models_root,
        activation_module.ActivationManifest(
            market=market,
            horizon=horizon,
            algorithm=algorithm,
            trained_date=trained_date,
            sidecar_sha256="deadbeef",
            promoted_at="2026-08-19T00:00:00+00:00",
            promotion_basis={},
        ),
    )


def _write_champion_artifact(
    models_root: Path, market: str, horizon: int, algorithm: str, trained_date: date
) -> Path:
    model_dir = persistence_module.model_dir(models_root, market, horizon, algorithm)
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = persistence_module.model_filename(market, horizon, algorithm, trained_date)
    model_path = model_dir / filename
    model_path.write_text("dummy", encoding="utf-8")
    return model_path


class TestResolveChampionModelPaths:
    """AC-ATG-008: 활성화 매니페스트 기반으로 챔피언 아티팩트 경로 매핑을 산출한다."""

    def test_resolves_active_champion_combos_only(self, tmp_path: Path):
        models_root = tmp_path / "models"
        _write_champion_manifest(models_root, "domestic", 60, "xgboost", date(2026, 8, 19))
        _write_champion_manifest(models_root, "overseas", 20, "xgboost", date(2026, 8, 19))
        _write_champion_manifest(models_root, "overseas", 60, "xgboost", date(2026, 8, 19))
        # domestic/20 매니페스트 없음(deployment_prohibited) — 매핑에서 제외되어야 한다.

        paths = resolve_champion_model_paths(models_root)

        assert set(paths.keys()) == {
            ("domestic", 60, "xgboost"),
            ("overseas", 20, "xgboost"),
            ("overseas", 60, "xgboost"),
        }
        expected = persistence_module.model_dir(
            models_root, "domestic", 60, "xgboost"
        ) / persistence_module.model_filename("domestic", 60, "xgboost", date(2026, 8, 19))
        assert paths[("domestic", 60, "xgboost")] == expected

    def test_returns_empty_mapping_when_no_manifests_exist(self, tmp_path: Path):
        assert resolve_champion_model_paths(tmp_path / "models") == {}


class TestWarnDanglingChampions:
    """AC-ATG-008: 댕글링 매니페스트(2단계 보존 정책으로 아카이브 이동된 trained_date)
    감지 시 detect_dangling_manifest() 경유 경고 로그를 발행한다."""

    def test_warns_when_manifest_trained_date_not_in_active_directory(self, tmp_path: Path, caplog):
        models_root = tmp_path / "models"
        _write_champion_manifest(models_root, "domestic", 60, "xgboost", date(2025, 1, 1))
        # active 경로에는 다른(더 최신) trained_date만 존재 — 매니페스트가 가리키는
        # 2025-01-01은 아카이브로 이동된 상태(댕글링).
        _write_champion_artifact(models_root, "domestic", 60, "xgboost", date(2026, 8, 10))

        paths = {("domestic", 60, "xgboost"): tmp_path / "champion.json"}
        with caplog.at_level("WARNING"):
            warn_dangling_champions(models_root, paths)

        assert any("dangling" in record.message for record in caplog.records)

    def test_no_warning_when_trained_date_still_active(self, tmp_path: Path, caplog):
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 19)
        _write_champion_manifest(models_root, "domestic", 60, "xgboost", trained_date)
        _write_champion_artifact(models_root, "domestic", 60, "xgboost", trained_date)

        paths = {("domestic", 60, "xgboost"): tmp_path / "champion.json"}
        with caplog.at_level("WARNING"):
            warn_dangling_champions(models_root, paths)

        assert not any("dangling" in record.message for record in caplog.records)


class TestReadFrozenHyperparameters:
    """AC-ATG-010: 챔피언 .meta.json 사이드카에서 frozen_hyperparameters를 읽는다."""

    def test_reads_frozen_hyperparameters_from_sidecar(self, tmp_path: Path):
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 19)
        _write_champion_manifest(models_root, "domestic", 60, "xgboost", trained_date)
        model_path = _write_champion_artifact(models_root, "domestic", 60, "xgboost", trained_date)
        sidecar_path = campaign_metrics_module.sidecar_path_for(model_path)
        sidecar_path.write_text(
            json.dumps(
                {
                    "frozen_hyperparameters": {
                        "n_estimators": 38,
                        "learning_rate": 0.0151,
                        "max_depth": 3,
                    }
                }
            ),
            encoding="utf-8",
        )

        frozen = read_frozen_hyperparameters(models_root, "domestic", 60, "xgboost")

        assert frozen == {"n_estimators": 38, "learning_rate": 0.0151, "max_depth": 3}

    def test_returns_none_and_warns_when_sidecar_missing(self, tmp_path: Path, caplog):
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 19)
        _write_champion_manifest(models_root, "domestic", 60, "xgboost", trained_date)
        _write_champion_artifact(models_root, "domestic", 60, "xgboost", trained_date)

        with caplog.at_level("WARNING"):
            frozen = read_frozen_hyperparameters(models_root, "domestic", 60, "xgboost")

        assert frozen is None
        assert any("missing" in record.message for record in caplog.records)

    def test_returns_none_and_warns_when_field_absent_in_sidecar(self, tmp_path: Path, caplog):
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 19)
        _write_champion_manifest(models_root, "domestic", 60, "xgboost", trained_date)
        model_path = _write_champion_artifact(models_root, "domestic", 60, "xgboost", trained_date)
        sidecar_path = campaign_metrics_module.sidecar_path_for(model_path)
        sidecar_path.write_text(json.dumps({"other_field": 1}), encoding="utf-8")

        with caplog.at_level("WARNING"):
            frozen = read_frozen_hyperparameters(models_root, "domestic", 60, "xgboost")

        assert frozen is None

    def test_returns_none_when_no_champion_manifest(self, tmp_path: Path):
        assert read_frozen_hyperparameters(tmp_path / "models", "domestic", 60, "xgboost") is None


class TestVerdictSerializationRoundtrip:
    """AC-ATG-007: PromotionVerdict 전 필드가 stdout JSON 라운드트립에서 무손실이다."""

    def test_roundtrip_preserves_all_fields(self):
        verdicts = {
            ("domestic", 60, "xgboost"): PromotionVerdict(
                market="domestic",
                horizon=60,
                algorithm="xgboost",
                promoted=True,
                challenger_rank_ic=0.0512,
                champion_rank_ic=0.0301,
                challenger_trained_date=date(2026, 8, 22),
            ),
            ("overseas", 20, "xgboost"): PromotionVerdict(
                market="overseas",
                horizon=20,
                algorithm="xgboost",
                promoted=False,
                challenger_rank_ic=0.01,
                champion_rank_ic=0.02,
                challenger_trained_date=date(2026, 8, 22),
            ),
        }

        raw = serialize_verdicts(verdicts)
        restored = deserialize_verdicts(raw)

        assert restored == verdicts

    def test_serializes_to_single_json_document(self):
        raw = serialize_verdicts({})
        # 파싱 가능한 단일 JSON 문서(리스트)여야 한다 — 잡음 텍스트가 섞이지 않음.
        assert json.loads(raw) == []

    def test_roundtrip_empty_mapping(self):
        assert deserialize_verdicts(serialize_verdicts({})) == {}


class TestParseArgs:
    """REQ-ATG-008: 게이트 CLI 인자 관례 — train.py CLI 인자 관례 계승."""

    def test_parses_required_arguments(self):
        args = parse_args(
            [
                "--models-root",
                "/tmp/models",
                "--cache-dir",
                "/tmp/cache",
                "--data-as-of",
                "2026-08-22",
                "--feature-code-version",
                "v1",
            ]
        )

        assert args.models_root == Path("/tmp/models")
        assert args.cache_dir == Path("/tmp/cache")
        assert args.data_as_of == date(2026, 8, 22)
        assert args.feature_code_version == "v1"
        assert args.calendar_code == "KRX"
        assert args.merged_to_active is True

    def test_accepts_no_merged_to_active_flag(self):
        args = parse_args(
            [
                "--models-root",
                "/tmp/models",
                "--cache-dir",
                "/tmp/cache",
                "--data-as-of",
                "2026-08-22",
                "--feature-code-version",
                "v1",
                "--no-merged-to-active",
            ]
        )

        assert args.merged_to_active is False


class TestRunGate:
    """M2: 게이트 CLI 본체 — 시장별 패널 조립(캐시 재사용) + (market,horizon) 단위
    반복 evaluate_and_promote() 호출(§B 리스크 1) + verdict 병합."""

    def test_calls_evaluate_and_promote_once_per_combo_with_champion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 22)
        # domestic/60·overseas/20·overseas/60 xgboost만 챔피언 보유(라이브 실측 정합).
        for market, horizon in (("domestic", 60), ("overseas", 20), ("overseas", 60)):
            _write_champion_manifest(models_root, market, horizon, "xgboost", trained_date)
            model_path = _write_champion_artifact(
                models_root, market, horizon, "xgboost", trained_date
            )
            sidecar_path = campaign_metrics_module.sidecar_path_for(model_path)
            sidecar_path.write_text(
                json.dumps({"frozen_hyperparameters": {"n_estimators": 5}}), encoding="utf-8"
            )

        fake_panel = pd.DataFrame(
            {"stock_code": ["S0"], "trade_date": [pd.Timestamp("2026-01-01")]}
        )
        monkeypatch.setattr(
            gate_module.campaign_module,
            "_assemble_campaign_dataset",
            lambda *args, **kwargs: fake_panel,
        )

        call_log: list[dict] = []

        def fake_evaluate_and_promote(**kwargs):
            call_log.append(kwargs)
            (market, horizon, algorithm), _path = next(iter(kwargs["champion_model_paths"].items()))
            verdict = PromotionVerdict(
                market=market,
                horizon=horizon,
                algorithm=algorithm,
                promoted=True,
                challenger_rank_ic=0.05,
                champion_rank_ic=0.01,
                challenger_trained_date=kwargs["challenger_trained_date"],
            )
            return {(market, horizon, algorithm): verdict}

        monkeypatch.setattr(gate_module, "evaluate_and_promote", fake_evaluate_and_promote)

        verdicts = run_gate(
            trainer_engine=MagicMock(),
            models_root=models_root,
            cache_dir=tmp_path / "cache",
            data_as_of=trained_date,
            feature_code_version="v1",
            calendar_code="KRX",
            merged_to_active=True,
        )

        # domestic/60, overseas/20, overseas/60 — 챔피언 보유 3개 조합만 호출된다.
        assert len(call_log) == 3
        assert set(verdicts.keys()) == {
            ("domestic", 60, "xgboost"),
            ("overseas", 20, "xgboost"),
            ("overseas", 60, "xgboost"),
        }
        for kwargs in call_log:
            assert kwargs["challenger_trained_date"] == trained_date
            assert kwargs["xgb_params"] == {"n_estimators": 5}

    def test_skips_combo_without_any_champion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        models_root = tmp_path / "models"
        fake_panel = pd.DataFrame(
            {"stock_code": ["S0"], "trade_date": [pd.Timestamp("2026-01-01")]}
        )
        monkeypatch.setattr(
            gate_module.campaign_module,
            "_assemble_campaign_dataset",
            lambda *args, **kwargs: fake_panel,
        )
        call_log: list[dict] = []
        monkeypatch.setattr(
            gate_module,
            "evaluate_and_promote",
            lambda **kwargs: call_log.append(kwargs) or {},
        )

        verdicts = run_gate(
            trainer_engine=MagicMock(),
            models_root=models_root,
            cache_dir=tmp_path / "cache",
            data_as_of=date(2026, 8, 22),
            feature_code_version="v1",
            calendar_code="KRX",
            merged_to_active=True,
        )

        assert call_log == []
        assert verdicts == {}

    def test_one_combo_failure_does_not_discard_other_combo_verdicts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """High-2 수정: (market,horizon) 한 조합의 evaluate_and_promote() 실패
        (예: 매니페스트는 있으나 아티팩트가 디스크에서 사라진 경우)가 나머지
        조합의 verdict까지 전부 폐기시키지 않는다 — 해당 조합만 건너뛰고
        나머지는 정상 판정된다."""
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 22)
        # domestic/60은 실패하게 하고, overseas/20은 정상 성공하게 한다.
        for market, horizon in (("domestic", 60), ("overseas", 20)):
            _write_champion_manifest(models_root, market, horizon, "xgboost", trained_date)
            model_path = _write_champion_artifact(
                models_root, market, horizon, "xgboost", trained_date
            )
            sidecar_path = campaign_metrics_module.sidecar_path_for(model_path)
            sidecar_path.write_text(
                json.dumps({"frozen_hyperparameters": {"n_estimators": 5}}), encoding="utf-8"
            )

        fake_panel = pd.DataFrame(
            {"stock_code": ["S0"], "trade_date": [pd.Timestamp("2026-01-01")]}
        )
        monkeypatch.setattr(
            gate_module.campaign_module,
            "_assemble_campaign_dataset",
            lambda *args, **kwargs: fake_panel,
        )

        def fake_evaluate_and_promote(**kwargs):
            (market, horizon, algorithm), _path = next(iter(kwargs["champion_model_paths"].items()))
            if (market, horizon) == ("domestic", 60):
                raise FileNotFoundError("champion artifact missing on disk")
            verdict = PromotionVerdict(
                market=market,
                horizon=horizon,
                algorithm=algorithm,
                promoted=True,
                challenger_rank_ic=0.05,
                champion_rank_ic=0.01,
                challenger_trained_date=kwargs["challenger_trained_date"],
            )
            return {(market, horizon, algorithm): verdict}

        monkeypatch.setattr(gate_module, "evaluate_and_promote", fake_evaluate_and_promote)

        with caplog.at_level("WARNING"):
            verdicts = run_gate(
                trainer_engine=MagicMock(),
                models_root=models_root,
                cache_dir=tmp_path / "cache",
                data_as_of=trained_date,
                feature_code_version="v1",
                calendar_code="KRX",
                merged_to_active=True,
            )

        assert set(verdicts.keys()) == {("overseas", 20, "xgboost")}
        assert any("gate combo evaluation failed" in record.message for record in caplog.records)


class TestGateMain:
    def test_returns_zero_and_prints_verdict_json_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        monkeypatch.setattr(gate_module, "build_trainer_engine", lambda: MagicMock())
        verdict = PromotionVerdict(
            market="domestic",
            horizon=60,
            algorithm="xgboost",
            promoted=True,
            challenger_rank_ic=0.05,
            champion_rank_ic=0.01,
            challenger_trained_date=date(2026, 8, 22),
        )
        monkeypatch.setattr(
            gate_module, "run_gate", lambda **kwargs: {("domestic", 60, "xgboost"): verdict}
        )

        exit_code = gate_module.main(
            [
                "--models-root",
                str(tmp_path / "models"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--data-as-of",
                "2026-08-22",
                "--feature-code-version",
                "v1",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out) == [
            {
                "market": "domestic",
                "horizon": 60,
                "algorithm": "xgboost",
                "promoted": True,
                "challenger_rank_ic": 0.05,
                "champion_rank_ic": 0.01,
                "challenger_trained_date": "2026-08-22",
            }
        ]

    def test_returns_one_on_exception_and_logs_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        monkeypatch.setattr(gate_module, "build_trainer_engine", lambda: MagicMock())

        def _raise(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(gate_module, "run_gate", _raise)

        exit_code = gate_module.main(
            [
                "--models-root",
                str(tmp_path / "models"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--data-as-of",
                "2026-08-22",
                "--feature-code-version",
                "v1",
            ]
        )

        assert exit_code == 1
        # stdout에는 JSON 외 텍스트가 섞이지 않아야 한다(§B 리스크 2).
        captured = capsys.readouterr()
        assert captured.out == ""


class TestGateCliSubprocessInvocation:
    def test_module_invocation_exits_nonzero_without_required_args(self):
        result = subprocess.run(
            [sys.executable, "-m", "analyzer.training.gate"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
