#!/usr/bin/env python3
"""semantic-release "무통보 릴리즈 누락" 방지 게이트.

`pyproject.toml`의 `[tool.semantic_release]`는 `commit_parser = "emoji"`로
설정돼 있다. python-semantic-release의 emoji 파서는 커밋 제목의 맨 앞에서
`commit_parser_options`에 등록된 gitmoji(또는 `:shortcode:`)를 찾지 못하면
그 커밋을 그냥 "Other"로 분류하고 조용히 버전 범프를 건너뛴다 — 에러도,
경고도 없다.

실제로 겪은 사고: `feat(SPEC-OBSV-ANALYZER-DEADMAN-001): ...` 커밋이 gitmoji
없이(Conventional Commits 형식만 지킨 채) main에 머지됐고, semantic-release가
"No release will be made"로 판단해 Docker 빌드·NAS 배포가 전혀 트리거되지
않았다. CI 어디에도 실패 신호가 없었다.

이 스크립트는 PR이 main에 반영하려는 커밋들(`git log origin/main..HEAD`) 중
semantic-release 버전 범프에 관여하는 conventional-commit 타입(`feat`/`fix`/
`perf`/`breaking`)을 가진 커밋이, 그 타입에 대응하는 bump-level의 gitmoji로
시작하는지 검사한다. `pyproject.toml`의 실제 태그 목록을 읽어서 비교하므로
태그 목록이 나중에 바뀌어도(드리프트) 이 게이트가 자동으로 따라간다.

사용법:
    python3 scripts/check_commit_gitmoji.py origin/main..HEAD
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Conventional Commits 타입 -> [tool.semantic_release.commit_parser_options]
# 레벨 키. feat=minor, fix/perf=patch, breaking=major은 Conventional Commits
# 자체의 의미이지 이 레포만의 설정이 아니므로 여기 고정해도 "드리프트"가
# 아니다. 실제 gitmoji/:shortcode: 토큰 목록은 pyproject.toml에서 읽는다
# (하드코딩 금지 — 태그 목록이 바뀌면 이 게이트도 자동으로 따라가야 한다).
TYPE_TO_LEVEL_KEY: dict[str, str] = {
    "feat": "minor_tags",
    "fix": "patch_tags",
    "perf": "patch_tags",
    "breaking": "major_tags",
}

# 게이트 대상 conventional-commit 타입 접두어를 찾는 정규식. 스코프
# `(scope)`와 breaking 표기 `!`는 선택적이며, 뒤에 반드시 콜론이 와야 한다.
_TYPE_RE = re.compile(r"^(?P<type>[a-zA-Z]+)(?:\([^)]*\))?!?:")

# git log 커밋 레코드 구분자(subject에 등장할 일이 없는 제어 문자).
_RECORD_SEP = "\x1f"


class GitmojiCheckError(RuntimeError):
    """pyproject.toml 설정이 기대한 형태가 아닐 때 발생."""


@dataclass(frozen=True)
class Violation:
    """게이트를 통과하지 못한 커밋 한 건."""

    sha: str
    subject: str
    commit_type: str
    expected_tags: tuple[str, ...]

    def format(self) -> str:
        expected = " 또는 ".join(self.expected_tags)
        sha_part = f"{self.sha} " if self.sha else ""
        return (
            f"{sha_part}커밋이 semantic-release 게이트를 통과하지 못했습니다: {self.subject!r}\n"
            f"  -> '{self.commit_type}' 타입 커밋은 다음 gitmoji 중 하나로 시작해야 합니다: "
            f"{expected}"
        )


def load_tags_by_level(pyproject_path: Path) -> dict[str, tuple[str, ...]]:
    """`pyproject.toml`에서 major/minor/patch 태그 목록을 그대로 읽는다.

    이 목록을 스크립트에 하드코딩하지 않고 매번 실제 설정 파일에서 읽어오는
    이유는, 나중에 누군가 `commit_parser_options`의 태그를 늘리거나 바꿔도
    이 게이트가 별도 수정 없이 자동으로 반영되게 하기 위해서다(드리프트 방지).
    """
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    try:
        options = data["tool"]["semantic_release"]["commit_parser_options"]
    except KeyError as exc:
        raise GitmojiCheckError(
            f"{pyproject_path}에 [tool.semantic_release.commit_parser_options] 섹션이 없습니다"
        ) from exc

    return {
        level_key: tuple(options.get(level_key, []))
        for level_key in ("major_tags", "minor_tags", "patch_tags")
    }


def extract_conventional_type(subject: str) -> str | None:
    """커밋 제목에서 conventional-commit 타입을 추출한다.

    선행 gitmoji가 있든("✨ feat(scope): ...") 없든("feat(scope): ...")
    타입 단어 자체는 정확히 찾아낸다 — gitmoji 존재 여부는 이 함수의
    관심사가 아니라 `check_subject`에서 원본 subject를 대상으로 별도
    검사한다. 콜론이 붙은 conventional-commit 접두어를 전혀 찾지 못하면
    (머지 커밋, 자유 형식 메시지 등) None을 반환 — 게이트 대상에서 면제된다.
    """
    rest = subject
    if subject and not subject[0].isascii():
        # 맨 앞 글자가 ASCII가 아니면 gitmoji로 보고 그 한 토큰만 걷어낸다.
        _emoji, sep, remainder = subject.partition(" ")
        if sep:
            rest = remainder

    match = _TYPE_RE.match(rest)
    return match.group("type") if match else None


def check_subject(
    subject: str,
    tags_by_level: dict[str, tuple[str, ...]],
    sha: str = "",
) -> Violation | None:
    """`subject`가 gitmoji 게이트를 통과하지 못하면 Violation을, 통과하면 None을 반환."""
    commit_type = extract_conventional_type(subject)
    if commit_type is None or commit_type not in TYPE_TO_LEVEL_KEY:
        return None  # docs/chore/test/ci/refactor/style/... 및 파싱 불가 커밋은 면제

    level_key = TYPE_TO_LEVEL_KEY[commit_type]
    expected_tags = tags_by_level.get(level_key, ())
    if not expected_tags:
        # 설정에 해당 레벨의 태그가 하나도 없으면 검사할 대상이 없다
        # (semantic-release 자신도 "Other"로 분류해 fail-open하는 것과 동일한 방향).
        return None

    if any(subject.startswith(tag) for tag in expected_tags):
        return None

    return Violation(
        sha=sha,
        subject=subject,
        commit_type=commit_type,
        expected_tags=expected_tags,
    )


def get_commit_subjects(commit_range: str, *, cwd: Path | None = None) -> list[tuple[str, str]]:
    """`commit_range` 구간의 (short-sha, subject) 목록을 오래된 순으로 반환.

    `--no-merges`는 semantic-release의 `ignore_merge_commits=True` 기본값과
    동일한 방향으로 머지 커밋을 검사 대상에서 제외한다.
    """
    # GIT_DIR/GIT_WORK_TREE 등 GIT_ 접두 환경변수가 앰비언트에 존재하면(예:
    # 이 프로세스가 git 훅 안에서 실행 중일 때) git이 `cwd`를 무시하고 그
    # 변수가 가리키는 저장소를 대상으로 동작한다. 명시적으로 지정한
    # `cwd`(테스트의 임시 저장소 포함)가 항상 우선하도록 GIT_ 환경변수를
    # 제거한 환경으로 호출한다.
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(
        ["git", "log", "--no-merges", f"--format=%h{_RECORD_SEP}%s", commit_range],
        cwd=cwd if cwd is not None else REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=clean_env,
    )
    commits: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        sha, _, subject = line.partition(_RECORD_SEP)
        commits.append((sha, subject))
    return commits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "commit_range",
        help="검사할 git 커밋 범위 (예: origin/main..HEAD)",
    )
    parser.add_argument(
        "--pyproject",
        default=str(REPO_ROOT / "pyproject.toml"),
        help="commit_parser_options를 읽을 pyproject.toml 경로",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="git log를 실행할 저장소 경로 (기본값: --pyproject의 상위 디렉터리)",
    )
    args = parser.parse_args(argv)

    try:
        tags_by_level = load_tags_by_level(Path(args.pyproject))
    except GitmojiCheckError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2

    repo_root = Path(args.repo) if args.repo else Path(args.pyproject).resolve().parent

    try:
        commits = get_commit_subjects(args.commit_range, cwd=repo_root)
    except subprocess.CalledProcessError as exc:
        print(f"오류: git log 실행 실패 ({exc})", file=sys.stderr)
        return 2

    violations = [
        v
        for sha, subject in commits
        if (v := check_subject(subject, tags_by_level, sha=sha)) is not None
    ]

    if not violations:
        print(f"OK: 커밋 {len(commits)}건 검사 완료, gitmoji 누락 없음")
        return 0

    print(
        f"실패: 커밋 {len(violations)}건이 semantic-release gitmoji 게이트를 통과하지 못했습니다\n",
        file=sys.stderr,
    )
    for violation in violations:
        print(violation.format(), file=sys.stderr)
        print(file=sys.stderr)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
