"""모델 영속화 — 네이티브 포맷 저장 + SHA-256 검증 + 2단계 보존 (SPEC-ANALYZER-TRAIN-001 M6).

REQ-AT-090/091/092: LightGBM/XGBoost 모델을 각 프레임워크의 네이티브
API·네이티브 포맷으로만 저장한다(pickle 등 언어/버전 종속 직렬화
미사용). 경로 관례는 `models/{market}/{horizon}/{algorithm}/`, 파일명은
`{market}_{horizon}_{algorithm}_{trained_date}`다 — `trading_signals.
model_version`(SCHEMA-001, 영구·비-FK 참조)이 소비하는 계약이므로 임의로
변경하지 않는다. 저장과 동시에 SHA-256 사이드카 해시 파일을 기록하고,
저장 직후 재계산 해시를 사이드카와 대조해 자체 검증한다.

REQ-AT-093/094/095: 동일 (시장, horizon, algorithm) 조합 내 최근 12개
버전은 "active"(models/{...} 경로에 압축 없이 유지), 그보다 오래된
버전은 월별 `archive/{YYYY-MM}.tar.zst`(임베디드 매니페스트 포함) 번들로
이동한다. tar 무결성 검증(멤버 목록 + 각 파일의 SHA-256 재대조)을
통과한 뒤에만 스테이징 원본을 삭제한다 — 검증 실패 시 원본을 보존한다
(shall not 삭제). 어떤 버전도 어느 보존 단계에서도 영구 삭제되지 않는다
— 아카이브를 통해 무기한 보존된다.
"""

import hashlib
import io
import json
import tarfile
from collections.abc import Sequence
from compression import zstd
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb

_NATIVE_EXTENSION: dict[str, str] = {"lightgbm": "txt", "xgboost": "json"}


@dataclass(frozen=True, slots=True)
class SavedModel:
    """저장 직후 산출물 — 모델 파일, 사이드카 해시 파일, 재검증된 SHA-256."""

    model_path: Path
    sidecar_path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """보존 정책 입력 — 이미 저장된 모델 버전 1개."""

    trained_date: date
    model_path: Path
    sidecar_path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class RetentionResult:
    """보존 정책 실행 결과 — active로 남은 버전 + 아카이브된 월(YYYY-MM) 목록."""

    active: list[ModelVersion]
    archived_months: list[str]


def model_dir(models_root: Path, market: str, horizon: int, algorithm: str) -> Path:
    """저장 경로 관례: `models/{market}/{horizon}/{algorithm}/`(REQ-AT-091)."""
    return models_root / market / str(horizon) / algorithm


def model_filename(market: str, horizon: int, algorithm: str, trained_date: date) -> str:
    """파일명 관례: `{market}_{horizon}_{algorithm}_{trained_date}.{native_ext}`(REQ-AT-091)."""
    if algorithm not in _NATIVE_EXTENSION:
        raise ValueError(f"지원하지 않는 algorithm: {algorithm}")
    ext = _NATIVE_EXTENSION[algorithm]
    return f"{market}_{horizon}_{algorithm}_{trained_date.isoformat()}.{ext}"


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_model_native(
    model: lgb.LGBMRegressor | xgb.XGBRegressor,
    models_root: Path,
    market: str,
    horizon: int,
    algorithm: str,
    trained_date: date,
) -> SavedModel:
    """모델을 네이티브 포맷으로 저장하고 SHA-256 사이드카를 기록 + 라운드트립 검증한다
    (REQ-AT-090/091/092, AC-AT-008).
    """
    if algorithm not in _NATIVE_EXTENSION:
        raise ValueError(f"지원하지 않는 algorithm: {algorithm}")

    target_dir = model_dir(models_root, market, horizon, algorithm)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / model_filename(market, horizon, algorithm, trained_date)

    if algorithm == "lightgbm":
        assert isinstance(model, lgb.LGBMRegressor)
        model.booster_.save_model(str(model_path))
    else:
        assert isinstance(model, xgb.XGBRegressor)
        model.get_booster().save_model(str(model_path))

    sha256 = _sha256_of_file(model_path)
    sidecar_path = model_path.with_suffix(model_path.suffix + ".sha256")
    sidecar_path.write_text(sha256, encoding="utf-8")

    if _sha256_of_file(model_path) != sha256:
        raise ValueError(f"{model_path} 저장 직후 SHA-256 라운드트립 검증 실패")

    return SavedModel(model_path=model_path, sidecar_path=sidecar_path, sha256=sha256)


def verify_model_integrity(model_path: Path, sidecar_path: Path) -> bool:
    """모델 파일의 현재 SHA-256이 사이드카 값과 일치하는지 검증한다."""
    expected = sidecar_path.read_text(encoding="utf-8").strip()
    return _sha256_of_file(model_path) == expected


def _create_archive_bundle(archive_path: Path, versions: Sequence[ModelVersion]) -> None:
    """월별 아카이브 tar.zst 번들을 생성한다 — 임베디드 매니페스트 포함(REQ-AT-093)."""
    manifest = [
        {
            "model": v.model_path.name,
            "sidecar": v.sidecar_path.name,
            "trained_date": v.trained_date.isoformat(),
            "sha256": v.sha256,
        }
        for v in versions
    ]
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zstd.open(archive_path, "wb") as zf, tarfile.open(fileobj=zf, mode="w|") as tar:
        manifest_info = tarfile.TarInfo(name="manifest.json")
        manifest_info.size = len(manifest_bytes)
        tar.addfile(manifest_info, io.BytesIO(manifest_bytes))
        for v in versions:
            tar.add(v.model_path, arcname=v.model_path.name)
            tar.add(v.sidecar_path, arcname=v.sidecar_path.name)


def _verify_archive_integrity(archive_path: Path, versions: Sequence[ModelVersion]) -> bool:
    """아카이브를 끝까지 읽어 각 모델 파일의 SHA-256이 원본과 일치하는지 검증한다(REQ-AT-094)."""
    expected_by_name = {v.model_path.name: v.sha256 for v in versions}
    found_names: set[str] = set()

    try:
        with zstd.open(archive_path, "rb") as zf, tarfile.open(fileobj=zf, mode="r|") as tar:
            for member in tar:
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                data = extracted.read()
                if member.name == "manifest.json":
                    found_names.add(member.name)
                    continue
                if member.name in expected_by_name:
                    if _sha256_of_bytes(data) != expected_by_name[member.name]:
                        return False
                    found_names.add(member.name)
    except tarfile.TarError, OSError:
        return False

    required = {"manifest.json", *expected_by_name.keys()}
    return required.issubset(found_names)


def apply_retention_policy(
    versions: Sequence[ModelVersion],
    archive_root: Path,
    active_count: int = 12,
) -> RetentionResult:
    """2단계 보존 정책을 실행한다(REQ-AT-093/094/095, AC-AT-009).

    최근 `active_count`개(기본 12)는 압축 없이 active로 유지하고, 그보다
    오래된 버전은 `trained_date` 기준 월별로 묶어 `archive/{YYYY-MM}.tar.zst`
    번들로 이동한다. 각 월 번들은 생성 직후 무결성을 검증하며, 검증을
    통과한 경우에만 스테이징 원본(모델 파일 + 사이드카)을 삭제한다 —
    검증 실패 시 `ValueError`를 발생시키고 원본은 그대로 보존된다(어떤
    버전도 영구 삭제되지 않는다).
    """
    sorted_versions = sorted(versions, key=lambda v: v.trained_date)
    if len(sorted_versions) <= active_count:
        return RetentionResult(active=list(sorted_versions), archived_months=[])

    boundary = len(sorted_versions) - active_count
    to_archive = sorted_versions[:boundary]
    active = sorted_versions[boundary:]

    by_month: dict[str, list[ModelVersion]] = {}
    for v in to_archive:
        by_month.setdefault(v.trained_date.strftime("%Y-%m"), []).append(v)

    archived_months: list[str] = []
    for month_key, month_versions in by_month.items():
        archive_path = archive_root / f"{month_key}.tar.zst"
        _create_archive_bundle(archive_path, month_versions)

        if not _verify_archive_integrity(archive_path, month_versions):
            raise ValueError(f"아카이브 {archive_path} 무결성 검증 실패 — 스테이징 원본 보존")

        for v in month_versions:
            v.model_path.unlink()
            v.sidecar_path.unlink()
        archived_months.append(month_key)

    return RetentionResult(active=active, archived_months=archived_months)
