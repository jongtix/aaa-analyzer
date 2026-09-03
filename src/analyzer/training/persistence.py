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
import shutil
import tarfile
import tempfile
from collections.abc import Sequence
from compression import zstd
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb

_NATIVE_EXTENSION: dict[str, str] = {"lightgbm": "txt", "xgboost": "json"}

MONTHLY_ACTIVE_COUNT: int = 36
"""REQ-ATT-017: 월간 성공 후처리가 `apply_retention_policy()`에 명시적으로 전달하는
active 유지 개수(2026-09-01 review-4 대응으로 12→36 상향). 주간(52)+월간(12) 합산
연 ~64회 프로모션 기준 약 6.75개월의 롤백 가능 윈도우에 해당한다.
`apply_retention_policy()` 자체의 파라미터 기본값 12는 무수정(PRESERVE)이다.
"""


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


def quantile_model_filename(market: str, horizon: int, alpha: float, trained_date: date) -> str:
    """분위수 보조 모델의 파일명 — 포인트 LightGBM 모델과 동일한 algorithm
    세그먼트("lightgbm")·디렉토리를 공유하되, 파일명 세그먼트에 alpha 구분자를
    추가해 충돌을 피한다(REQ-ATE-007/008/010).

    `model_filename()`은 algorithm을 {"lightgbm","xgboost"} 두 키로만
    검증하므로, 여기서는 그 함수가 만드는 포인트 모델용 이름을 그대로 얻은 뒤
    alpha 태그만 사후 삽입한다 — 확장자는 변경하지 않는다.

    `training/train.py`(주간 정기 재학습)와 `training/campaign.py`(월간
    캠페인 챔피언 배포)가 공유하는 헬퍼다 — 두 곳에 각자 구현하면 파일명
    관례가 서서히 갈라질 위험이 있어 이 모듈로 일원화했다.
    """
    base = model_filename(market, horizon, "lightgbm", trained_date)
    stem, _, ext = base.rpartition(".")
    alpha_tag = f"q{round(alpha * 100):02d}"
    return f"{stem}_{alpha_tag}.{ext}"


def save_quantile_model(
    model: lgb.LGBMRegressor,
    models_root: Path,
    market: str,
    horizon: int,
    alpha: float,
    trained_date: date,
) -> SavedModel:
    """분위수 보조 모델을 저장한다 — `save_model_native()`(algorithm="lightgbm")를
    임시 스테이징 디렉토리에서 호출해 SHA-256 라운드트립 검증(REQ-AT-092)을 그대로
    재사용한 뒤, 포인트 모델의 실경로를 절대 건드리지 않고 alpha 접미사가 붙은
    최종 파일명으로 옮긴다(REQ-ATE-007/008/009/010, AC-ATE-003).

    `training/train.py`(주간 정기 재학습)와 `training/campaign.py`(월간
    캠페인 챔피언 배포)가 공유하는 헬퍼다(위 함수와 동일한 일원화 취지).
    """
    models_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models_root) as staging:
        staged = save_model_native(model, Path(staging), market, horizon, "lightgbm", trained_date)
        final_dir = model_dir(models_root, market, horizon, "lightgbm")
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / quantile_model_filename(market, horizon, alpha, trained_date)
        final_sidecar = final_path.with_suffix(final_path.suffix + ".sha256")
        shutil.move(str(staged.model_path), str(final_path))
        shutil.move(str(staged.sidecar_path), str(final_sidecar))
        return SavedModel(model_path=final_path, sidecar_path=final_sidecar, sha256=staged.sha256)


def verify_model_integrity(model_path: Path, sidecar_path: Path) -> bool:
    """모델 파일의 현재 SHA-256이 사이드카 값과 일치하는지 검증한다."""
    expected = sidecar_path.read_text(encoding="utf-8").strip()
    return _sha256_of_file(model_path) == expected


def enumerate_model_versions(
    models_root: Path, market: str, horizon: int, algorithm: str
) -> list[ModelVersion]:
    """REQ-ATT-016: `model_dir()` 경로를 스캔해 기존 파일명 관례(REQ-AT-092)와
    일치하는 네이티브 모델 파일 + `.sha256` 사이드카 **쌍**으로부터
    `ModelVersion` 시퀀스를 구성한다(`trained_date` 오름차순).

    사이드카가 없는 모델 파일, 파일명 관례에서 벗어난 파일, 그리고 아카이브로
    이동된 버전은 결과에 포함되지 않는다 — 이 헬퍼는 active 경로만 스캔한다
    (spec.md §4 알려진 한계 5). `apply_retention_policy()`는 수정하지 않는다
    (PRESERVE).
    """
    if algorithm not in _NATIVE_EXTENSION:
        raise ValueError(f"지원하지 않는 algorithm: {algorithm}")

    target_dir = model_dir(models_root, market, horizon, algorithm)
    if not target_dir.is_dir():
        return []

    ext = _NATIVE_EXTENSION[algorithm]
    prefix = f"{market}_{horizon}_{algorithm}_"

    versions: list[ModelVersion] = []
    for model_path in target_dir.glob(f"{prefix}*.{ext}"):
        sidecar_path = model_path.with_suffix(model_path.suffix + ".sha256")
        if not sidecar_path.is_file():
            continue
        try:
            trained_date = date.fromisoformat(model_path.name[len(prefix) : -(len(ext) + 1)])
        except ValueError:
            continue
        versions.append(
            ModelVersion(
                trained_date=trained_date,
                model_path=model_path,
                sidecar_path=sidecar_path,
                sha256=sidecar_path.read_text(encoding="utf-8").strip(),
            )
        )

    return sorted(versions, key=lambda v: v.trained_date)


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
    except tarfile.TarError:
        return False
    except OSError:
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


def combo_archive_root(models_root: Path, market: str, horizon: int, algorithm: str) -> Path:
    """아카이브 경로 관례: `{models_root}/archive/{market}/{horizon}/{algorithm}/`.

    최상위 아카이브 루트는 기존 관례대로 `models_root/archive` 하나이며(신규
    최상위 경로 신설 없음), 그 아래를 `model_dir()`과 동일한 조합 세그먼트로
    분할한다 — `apply_retention_policy()`가 월 키(`{YYYY-MM}.tar.zst`)만으로
    번들 이름을 정하므로, 조합별로 분할하지 않으면 같은 달에 아카이브되는 서로
    다른 조합의 번들이 동일 경로에 덮어써진다.
    """
    return models_root / "archive" / market / str(horizon) / algorithm


def apply_retention_for_combos(
    models_root: Path,
    combos: Sequence[tuple[str, int, str]],
    active_count: int = MONTHLY_ACTIVE_COUNT,
) -> dict[tuple[str, int, str], RetentionResult]:
    """REQ-ATT-017: 조합별 보존 정책 실행 — 월간 성공 후처리의 호출 지점.

    각 (시장, horizon, algorithm) 조합에 대해 `enumerate_model_versions()`(M1)로
    active 버전을 열거하고, 그 결과를 `apply_retention_policy()`에 `active_count`
    (기본 `MONTHLY_ACTIVE_COUNT=36`)와 조합별 `archive_root`와 함께 전달한다.
    `apply_retention_policy()`는 수정하지 않고 소환만 한다(PRESERVE).

    조합 목록의 출처(캠페인 요약 리포트 또는 `POINT_COMBOS` 순회)는 이 함수의
    호출자(월간 후처리 훅) 소관이므로 여기서 결정하지 않는다.

    한 조합의 아카이브 무결성 검증이 실패하면 `apply_retention_policy()`가
    `ValueError`를 발생시키며, 그 예외는 그대로 전파된다 — 실패한 조합의 원본은
    보존되고 이후 조합은 처리되지 않는다.
    """
    results: dict[tuple[str, int, str], RetentionResult] = {}
    for market, horizon, algorithm in combos:
        versions = enumerate_model_versions(models_root, market, horizon, algorithm)
        results[(market, horizon, algorithm)] = apply_retention_policy(
            versions,
            archive_root=combo_archive_root(models_root, market, horizon, algorithm),
            active_count=active_count,
        )
    return results
