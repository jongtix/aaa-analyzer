"""scripts/check_commit_gitmoji.py 단위 테스트.

semantic-release가 "No release will be made"로 조용히 판정해 Docker 빌드·
NAS 배포가 트리거되지 않는 사고(feat(SPEC-OBSV-ANALYZER-DEADMAN-001) 커밋이
gitmoji 없이 main에 머지된 실제 사례)를 PR 시점에 차단하는 게이트 스크립트를
검증한다.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import check_commit_gitmoji  # noqa: E402

# 이 레포의 실제 pyproject.toml [tool.semantic_release.commit_parser_options]와
# 동일한 태그 목록 — 실제 사고 재현 테스트에 사용한다.
REPO_TAGS_BY_LEVEL = {
    "major_tags": ("breaking",),
    "minor_tags": ("✨", ":sparkles:"),
    "patch_tags": ("🐛", "⚡", ":bug:", ":zap:"),
}


class TestCheckSubject:
    def test_feat_commit_with_correct_gitmoji_passes(self):
        subject = "✨ feat(analyzer): stream:signal:{market} 신호 발행 추가"

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL)

        assert violation is None

    def test_feat_commit_missing_gitmoji_fails_reproduces_real_incident(self):
        # 2026-09 실제 사고: 이 정확한 제목의 커밋이 gitmoji 없이 main에
        # 머지되어 semantic-release가 릴리즈를 만들지 않았다.
        subject = (
            "feat(SPEC-OBSV-ANALYZER-DEADMAN-001): "
            "M1+M2 analyzer 게이지+Redis 영속화+warm-start 추가"
        )

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL)

        assert violation is not None
        assert violation.commit_type == "feat"
        assert violation.expected_tags == ("✨", ":sparkles:")

    def test_chore_deps_dependabot_commit_is_exempt(self):
        subject = "chore(deps): Bump the minor-and-patch group with 5 updates"

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL)

        assert violation is None

    def test_fix_commit_with_wrong_type_gitmoji_fails(self):
        # 📝(memo)는 patch_tags에 없다 — fix 타입에는 맞지 않는 gitmoji.
        subject = "📝 fix(analyzer): 잘못된 gitmoji로 시작하는 fix 커밋"

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL)

        assert violation is not None
        assert violation.commit_type == "fix"
        assert violation.expected_tags == ("🐛", "⚡", ":bug:", ":zap:")

    def test_fix_commit_with_correct_gitmoji_passes(self):
        subject = "🐛 fix(inference): Redis 소켓 타임아웃 결함 정정"

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL)

        assert violation is None

    def test_perf_commit_with_patch_level_gitmoji_passes(self):
        subject = "⚡ perf(analyzer): 추론 피처 조립 캐싱"

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL)

        assert violation is None

    def test_perf_commit_missing_gitmoji_fails(self):
        subject = "perf(analyzer): 추론 피처 조립 캐싱"

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL)

        assert violation is not None
        assert violation.commit_type == "perf"

    def test_docs_commit_is_exempt(self):
        subject = "docs(SPEC-ANALYZER-TRAIN-META-001): sync-phase 산출물 작성"

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL)

        assert violation is None

    def test_breaking_type_with_no_own_gitmoji_config_passes_trivially(self):
        # major_tags=["breaking"] 자체가 문자열 태그이므로, 타입 단어와
        # 태그가 같은 경우 gitmoji 프리픽스 없이도 조건을 만족한다.
        subject = "breaking(api): 공개 엔드포인트 제거"

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL)

        assert violation is None

    def test_merge_commit_is_exempt(self):
        subject = "Merge pull request #64 from jongtix/dependabot/uv/deps"

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL)

        assert violation is None

    def test_release_commit_is_exempt(self):
        subject = "🔖 chore(release): v1.2.3 [skip ci]"

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL)

        assert violation is None

    def test_level_with_no_configured_tags_is_fail_open(self):
        tags_by_level = {"major_tags": (), "minor_tags": (), "patch_tags": ()}
        subject = "feat(analyzer): 새 기능"

        violation = check_commit_gitmoji.check_subject(subject, tags_by_level)

        assert violation is None

    def test_violation_format_includes_sha_subject_and_expected_tags(self):
        subject = "feat(analyzer): gitmoji 누락"

        violation = check_commit_gitmoji.check_subject(subject, REPO_TAGS_BY_LEVEL, sha="abc1234")

        assert violation is not None
        message = violation.format()
        assert "abc1234" in message
        assert subject in message
        assert "✨" in message


class TestExtractConventionalType:
    def test_no_gitmoji_prefix(self):
        assert check_commit_gitmoji.extract_conventional_type("feat(scope): desc") == "feat"

    def test_with_gitmoji_prefix(self):
        assert check_commit_gitmoji.extract_conventional_type("✨ feat(scope): desc") == "feat"

    def test_no_scope(self):
        assert check_commit_gitmoji.extract_conventional_type("✨ ci: desc") == "ci"

    def test_breaking_bang_marker(self):
        assert check_commit_gitmoji.extract_conventional_type("feat!: desc") == "feat"

    def test_unparseable_subject_returns_none(self):
        assert check_commit_gitmoji.extract_conventional_type("Merge branch 'main'") is None

    def test_empty_subject_returns_none(self):
        assert check_commit_gitmoji.extract_conventional_type("") is None


class TestLoadTagsByLevel:
    def test_reads_actual_project_pyproject_toml(self):
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"

        tags_by_level = check_commit_gitmoji.load_tags_by_level(pyproject_path)

        assert tags_by_level["minor_tags"] == ("✨", ":sparkles:")
        assert tags_by_level["patch_tags"] == ("🐛", "⚡", ":bug:", ":zap:")
        assert tags_by_level["major_tags"] == ("breaking",)

    def test_missing_commit_parser_options_section_raises(self, tmp_path):
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text('[tool.other]\nkey = "value"\n', encoding="utf-8")

        try:
            check_commit_gitmoji.load_tags_by_level(pyproject_path)
        except check_commit_gitmoji.GitmojiCheckError:
            pass
        else:
            raise AssertionError("GitmojiCheckError를 기대했지만 발생하지 않았다")

    def test_missing_tag_key_defaults_to_empty_tuple(self, tmp_path):
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(
            '[tool.semantic_release.commit_parser_options]\nminor_tags = ["✨"]\n',
            encoding="utf-8",
        )

        tags_by_level = check_commit_gitmoji.load_tags_by_level(pyproject_path)

        assert tags_by_level["minor_tags"] == ("✨",)
        assert tags_by_level["patch_tags"] == ()
        assert tags_by_level["major_tags"] == ()


class TestGetCommitSubjects:
    def test_returns_sha_and_subject_pairs(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        (repo / "a.txt").write_text("1", encoding="utf-8")
        subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "✨ feat(x): first"],
            cwd=repo,
            check=True,
        )
        subprocess.run(["git", "branch", "base"], cwd=repo, check=True)
        (repo / "a.txt").write_text("2", encoding="utf-8")
        subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "feat(y): second, no gitmoji"],
            cwd=repo,
            check=True,
        )

        commits = check_commit_gitmoji.get_commit_subjects("base..HEAD", cwd=repo)

        assert len(commits) == 1
        sha, subject = commits[0]
        assert len(sha) > 0
        assert subject == "feat(y): second, no gitmoji"


class TestMain:
    def test_main_reports_failure_exit_code_on_violation(self, tmp_path, capsys):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        (repo / "a.txt").write_text("1", encoding="utf-8")
        subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
        subprocess.run(["git", "branch", "base"], cwd=repo, check=True)
        (repo / "a.txt").write_text("2", encoding="utf-8")
        subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "feat(SPEC-X): gitmoji 없는 feat"],
            cwd=repo,
            check=True,
        )
        pyproject_path = repo / "pyproject.toml"
        pyproject_path.write_text(
            "[tool.semantic_release.commit_parser_options]\n"
            'minor_tags = ["✨"]\n'
            'patch_tags = ["🐛", "⚡"]\n'
            'major_tags = ["breaking"]\n',
            encoding="utf-8",
        )

        exit_code = check_commit_gitmoji.main(["base..HEAD", "--pyproject", str(pyproject_path)])

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "gitmoji 없는 feat" in captured.err

    def test_main_reports_success_exit_code_when_clean(self, tmp_path, capsys):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
        (repo / "a.txt").write_text("1", encoding="utf-8")
        subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
        subprocess.run(["git", "branch", "base"], cwd=repo, check=True)
        (repo / "a.txt").write_text("2", encoding="utf-8")
        subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "✨ feat(SPEC-X): gitmoji 있는 feat"],
            cwd=repo,
            check=True,
        )
        pyproject_path = repo / "pyproject.toml"
        pyproject_path.write_text(
            "[tool.semantic_release.commit_parser_options]\n"
            'minor_tags = ["✨"]\n'
            'patch_tags = ["🐛", "⚡"]\n'
            'major_tags = ["breaking"]\n',
            encoding="utf-8",
        )

        exit_code = check_commit_gitmoji.main(["base..HEAD", "--pyproject", str(pyproject_path)])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "OK" in captured.out

    def test_main_returns_error_exit_code_on_missing_config_section(self, tmp_path):
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text('[tool.other]\nkey = "value"\n', encoding="utf-8")

        exit_code = check_commit_gitmoji.main(
            ["origin/main..HEAD", "--pyproject", str(pyproject_path)]
        )

        assert exit_code == 2
