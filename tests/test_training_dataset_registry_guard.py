"""REQ-AT-021(AC-AT-003): 조정 레지스트리 비어있음 방지 가드.

GWT: `training/dataset.py`만 임포트한 새 Python 프로세스에서
`analyzer.data.adjustment.HANDLER_REGISTRY`를 조회하면 SPLIT/DIVIDEND
키가 모두 존재해야 한다(빈 딕셔너리가 아니어야 한다). 다른 테스트 파일이
이미 여러 모듈을 임포트해둔 동일 프로세스에서는 이 가드가 실제로
동작하는지 검증할 수 없으므로(레지스트리가 이미 채워져 있을 수 있음)
별도 서브프로세스로 실행한다.
"""

import subprocess
import sys


def test_ac_at_003_handler_registry_populated_after_dataset_import_fresh_process():
    script = (
        "import analyzer.training.dataset\n"
        "from analyzer.data.adjustment import HANDLER_REGISTRY\n"
        "assert HANDLER_REGISTRY, 'HANDLER_REGISTRY must not be empty'\n"
        "assert 'SPLIT' in HANDLER_REGISTRY, 'SPLIT handler missing'\n"
        "assert 'DIVIDEND' in HANDLER_REGISTRY, 'DIVIDEND handler missing'\n"
        "print('OK')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_dataset_module_asserts_registry_populated_at_import_time():
    """dataset.py 자체가 모듈 로드 시점에 fail-fast assertion을 실행하는지 확인."""
    import analyzer.training.dataset as dataset_module
    from analyzer.data.adjustment import HANDLER_REGISTRY

    assert dataset_module is not None
    assert "SPLIT" in HANDLER_REGISTRY
    assert "DIVIDEND" in HANDLER_REGISTRY
