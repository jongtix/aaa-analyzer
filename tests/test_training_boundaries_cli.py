"""등급 경계 실측 산출 CLI 테스트 (SPEC-ANALYZER-INFER-001 M4, REQ-AIF-080/111, AC-AIF-014).

실 DB 없이 조립 경로(`train._assemble_market_dataset`)와 캘린더 조회를 mock으로
대체해 fetch → assemble → 분위수 역산 → δ 유도 → JSON 기록 흐름만 검증한다.
모든 수치는 합성 입력이다 — 실측 경계값은 테스트에 등장하지 않는다.
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from analyzer.inference.boundaries_store import BOUNDARY_KEYS, load_grade_boundaries
from analyzer.training import boundaries_cli
from analyzer.training.boundaries_cli import (
    DEFAULT_TARGET_RATIOS,
    build_artifact_payload,
    collect_realized_returns,
    main,
)


def _synthetic_assembled(market: str, n_rows: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scale = 0.05 if market == "domestic" else 0.08
    frame = pd.DataFrame(
        {
            "stock_code": ["A"] * n_rows,
            "trade_date": pd.to_datetime(pd.date_range("2024-01-01", periods=n_rows, freq="B")),
            "label_D20": rng.normal(0.0, scale, n_rows),
            "label_D20_exclude_reason": pd.Series([None] * n_rows, dtype="object"),
            "label_D60": rng.normal(0.0, scale * 1.7, n_rows),
            "label_D60_exclude_reason": pd.Series([None] * n_rows, dtype="object"),
        }
    )
    # 마지막 10행은 제외 사유가 있고 레이블이 NaN — 행 수 집계에서 빠져야 한다.
    frame.loc[frame.index[-10:], "label_D20"] = np.nan
    frame.loc[frame.index[-10:], "label_D20_exclude_reason"] = "INSUFFICIENT_FORWARD_DAYS"
    # D60은 exclude_reason만 있고 레이블 값은 남아 있는 행 5개 — 역시 제외돼야 한다.
    frame.loc[frame.index[-5:], "label_D60_exclude_reason"] = "HALT_IN_WINDOW"
    return frame


@pytest.fixture
def mocked_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    engine = MagicMock(name="trainer_engine")
    build_engine = MagicMock(name="build_trainer_engine", return_value=engine)
    fetch_calendar = MagicMock(name="fetch_market_calendar", return_value=object())
    assemble = MagicMock(
        name="_assemble_market_dataset",
        side_effect=lambda _engine, _cal, market, *_: _synthetic_assembled(market),
    )
    monkeypatch.setattr(boundaries_cli, "build_trainer_engine", build_engine)
    monkeypatch.setattr(boundaries_cli, "fetch_market_calendar", fetch_calendar)
    monkeypatch.setattr(boundaries_cli.train_module, "_assemble_market_dataset", assemble)
    return {
        "engine": engine,
        "build_engine": build_engine,
        "fetch_calendar": fetch_calendar,
        "assemble": assemble,
    }


class TestCollectRealizedReturns:
    def test_returns_four_combinations_excluding_flagged_rows(
        self, mocked_pipeline: dict[str, MagicMock], tmp_path: Path
    ):
        collected = collect_realized_returns(
            mocked_pipeline["engine"],
            calendar_code="KRX",
            cache_dir=tmp_path,
            data_as_of=date(2026, 1, 31),
            feature_code_version="v-test",
        )

        assert set(collected.returns_by_combination) == {
            ("domestic", 20),
            ("domestic", 60),
            ("overseas", 20),
            ("overseas", 60),
        }
        assert len(collected.returns_by_combination[("domestic", 20)]) == 490
        assert len(collected.returns_by_combination[("domestic", 60)]) == 495
        assert collected.row_counts[("overseas", 20)] == 490
        assert not collected.returns_by_combination[("overseas", 60)].isna().any()

    def test_data_as_of_is_latest_trade_date_actually_used(
        self, mocked_pipeline: dict[str, MagicMock], tmp_path: Path
    ):
        collected = collect_realized_returns(
            mocked_pipeline["engine"],
            calendar_code="KRX",
            cache_dir=tmp_path,
            data_as_of=date(2026, 1, 31),
            feature_code_version="v-test",
        )

        expected = _synthetic_assembled("domestic")["trade_date"].max().date()
        assert collected.data_as_of == expected

    def test_applies_overseas_calendar_override_like_train_and_campaign(
        self, mocked_pipeline: dict[str, MagicMock], tmp_path: Path
    ):
        collect_realized_returns(
            mocked_pipeline["engine"],
            calendar_code="KRX",
            cache_dir=tmp_path,
            data_as_of=date(2026, 1, 31),
            feature_code_version="v-test",
        )

        calendar_codes = [c.args[1] for c in mocked_pipeline["fetch_calendar"].call_args_list]
        assert calendar_codes == ["KRX", "NYSE"]

    def test_fetches_markets_sequentially_over_the_single_engine(
        self, mocked_pipeline: dict[str, MagicMock], tmp_path: Path
    ):
        collect_realized_returns(
            mocked_pipeline["engine"],
            calendar_code="KRX",
            cache_dir=tmp_path,
            data_as_of=date(2026, 1, 31),
            feature_code_version="v-test",
        )

        markets = [c.args[2] for c in mocked_pipeline["assemble"].call_args_list]
        assert markets == ["domestic", "overseas"]
        assert all(
            c.args[0] is mocked_pipeline["engine"]
            for c in mocked_pipeline["assemble"].call_args_list
        )

    def test_rejects_combination_with_no_rows(
        self, mocked_pipeline: dict[str, MagicMock], tmp_path: Path
    ):
        empty = _synthetic_assembled("domestic").iloc[0:0]
        mocked_pipeline["assemble"].side_effect = lambda *_: empty

        with pytest.raises(ValueError, match="실현 수익률"):
            collect_realized_returns(
                mocked_pipeline["engine"],
                calendar_code="KRX",
                cache_dir=tmp_path,
                data_as_of=date(2026, 1, 31),
                feature_code_version="v-test",
            )


class TestBuildArtifactPayload:
    def test_payload_has_boundaries_delta_and_metadata(self):
        returns = {
            (market, horizon): pd.Series(np.linspace(-0.3, 0.3, 601) * (horizon / 20))
            for market in ("domestic", "overseas")
            for horizon in (20, 60)
        }
        row_counts = {combo: len(series) for combo, series in returns.items()}

        payload = build_artifact_payload(
            returns,
            row_counts=row_counts,
            target_ratios=DEFAULT_TARGET_RATIOS,
            data_as_of=date(2026, 1, 30),
            generated_at="2026-02-01T09:00:00+09:00",
        )

        assert payload["schema_version"] == 1
        assert payload["data_as_of"] == "2026-01-30"
        assert payload["target_ratios"] == DEFAULT_TARGET_RATIOS
        assert len(payload["combinations"]) == 4
        for entry in payload["combinations"]:
            assert list(entry["boundaries"]) == list(BOUNDARY_KEYS)
            assert entry["row_count"] == 601
        delta = payload["grade_margin_delta"]
        assert delta["provisional"] is True
        assert delta["value"] == pytest.approx(0.10 * delta["min_adjacent_gap"])
        assert "NOTIFIER-FILTER-001" in delta["replace_by"]

    def test_delta_is_ten_percent_of_global_minimum_gap(self):
        # D20 조합의 간격이 D60보다 좁으므로 전역 최소 간격은 D20에서 나온다.
        returns = {
            (market, horizon): pd.Series(np.linspace(-0.3, 0.3, 601) * (horizon / 20))
            for market in ("domestic", "overseas")
            for horizon in (20, 60)
        }
        payload = build_artifact_payload(
            returns,
            row_counts={c: 601 for c in returns},
            target_ratios=DEFAULT_TARGET_RATIOS,
            data_as_of=date(2026, 1, 30),
            generated_at="x",
        )

        d20 = next(e for e in payload["combinations"] if e["horizon"] == 20)["boundaries"]
        d20_values = list(d20.values())
        d20_min_gap = min(b - a for a, b in zip(d20_values, d20_values[1:], strict=False))
        assert payload["grade_margin_delta"]["min_adjacent_gap"] == pytest.approx(d20_min_gap)

    def test_payload_round_trips_through_store_loader(self, tmp_path: Path):
        returns = {
            (market, horizon): pd.Series(np.linspace(-0.3, 0.3, 601) * (horizon / 20))
            for market in ("domestic", "overseas")
            for horizon in (20, 60)
        }
        payload = build_artifact_payload(
            returns,
            row_counts={c: 601 for c in returns},
            target_ratios=DEFAULT_TARGET_RATIOS,
            data_as_of=date(2026, 1, 30),
            generated_at="x",
        )
        path = tmp_path / "gb.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        artifact = load_grade_boundaries(path)

        assert artifact.grade_margin_delta == pytest.approx(payload["grade_margin_delta"]["value"])
        assert artifact.row_counts[("overseas", 60)] == 601


class TestMain:
    def test_writes_artifact_and_returns_zero(
        self, mocked_pipeline: dict[str, MagicMock], tmp_path: Path
    ):
        output = tmp_path / "out" / "grade_boundaries.json"

        exit_code = main(
            [
                "--cache-dir",
                str(tmp_path / "cache"),
                "--data-as-of",
                "2026-01-31",
                "--feature-code-version",
                "v-test",
                "--output",
                str(output),
            ]
        )

        assert exit_code == 0
        assert mocked_pipeline["build_engine"].call_count == 1
        artifact = load_grade_boundaries(output)
        assert artifact.row_counts[("domestic", 20)] == 490
        assert artifact.target_ratios == DEFAULT_TARGET_RATIOS
        assert artifact.grade_margin_delta_provisional is True

    def test_returns_one_and_reports_failure(
        self,
        mocked_pipeline: dict[str, MagicMock],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        mocked_pipeline["assemble"].side_effect = RuntimeError("boom")

        exit_code = main(
            [
                "--cache-dir",
                str(tmp_path),
                "--data-as-of",
                "2026-01-31",
                "--feature-code-version",
                "v-test",
                "--output",
                str(tmp_path / "gb.json"),
            ]
        )

        assert exit_code == 1
        assert "boom" in capsys.readouterr().err
        assert not (tmp_path / "gb.json").exists()

    def test_default_output_is_packaged_store_path(self):
        from analyzer.inference.boundaries_store import default_artifact_path

        parser = boundaries_cli.build_parser()
        args = parser.parse_args(
            ["--cache-dir", "c", "--data-as-of", "2026-01-31", "--feature-code-version", "v"]
        )

        assert args.output == default_artifact_path()
