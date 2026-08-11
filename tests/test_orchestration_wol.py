"""Wake-on-LAN 매직패킷 송신 테스트 (SPEC-ANALYZER-TRAIN-AUTOMATION-001, REQ-ATA-010/011/014).

`WolSender`는 `typing.Protocol`로 정의되어 실 네트워크 없이 페이크 구현으로 단위
테스트 가능하다(spec.md §4.1 설계 근거 — NAS 네트워크 접근 없이도 재시도/멱등성
로직을 개발·검증할 수 있도록 실 송신 부분을 인터페이스 뒤로 격리).
"""

import socket
from unittest.mock import MagicMock, patch

import pytest

from analyzer.orchestration.wol import (
    InvalidMacAddressError,
    UdpBroadcastWolSender,
    WolResult,
    build_magic_packet,
    send_with_retry,
)


class TestBuildMagicPacket:
    """REQ-ATA-010: WoL 매직패킷 바이트열 구성(0xFF*6 + MAC*16)."""

    def test_builds_packet_with_correct_structure(self):
        packet = build_magic_packet("AA:BB:CC:DD:EE:FF")
        mac_bytes = bytes.fromhex("AABBCCDDEEFF")

        assert packet == b"\xff" * 6 + mac_bytes * 16
        assert len(packet) == 102

    def test_accepts_hyphen_separated_mac(self):
        assert build_magic_packet("AA-BB-CC-DD-EE-FF") == build_magic_packet("AA:BB:CC:DD:EE:FF")

    def test_accepts_lowercase_mac(self):
        assert build_magic_packet("aa:bb:cc:dd:ee:ff") == build_magic_packet("AA:BB:CC:DD:EE:FF")

    def test_rejects_invalid_mac_format(self):
        with pytest.raises(InvalidMacAddressError):
            build_magic_packet("not-a-mac")

    def test_rejects_too_short_mac(self):
        with pytest.raises(InvalidMacAddressError):
            build_magic_packet("AA:BB:CC")


class TestUdpBroadcastWolSender:
    """구체 구현 — UDP 브로드캐스트로 매직패킷을 송신한다.

    [plan.md §D 제약] 이 클래스는 전달받은 MAC 주소를 검증 없이 그대로 송신할
    뿐이다 — 하드웨어(번인) MAC 확보 책임은 호출자(config/운영 문서)에 있다.
    """

    def test_sends_packet_via_broadcast_socket(self):
        sender = UdpBroadcastWolSender(broadcast_address="192.168.0.255", port=9)
        with patch("analyzer.orchestration.wol.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value.__enter__.return_value = mock_sock

            result = sender.send("AA:BB:CC:DD:EE:FF")

        assert result is True
        mock_sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        assert mock_sock.sendto.call_count == 1
        (_packet, address) = mock_sock.sendto.call_args[0]
        assert address == ("192.168.0.255", 9)

    def test_returns_false_on_socket_error(self):
        sender = UdpBroadcastWolSender()
        with patch("analyzer.orchestration.wol.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.sendto.side_effect = OSError("network unreachable")
            mock_socket_cls.return_value.__enter__.return_value = mock_sock

            result = sender.send("AA:BB:CC:DD:EE:FF")

        assert result is False

    def test_returns_false_on_invalid_mac(self):
        sender = UdpBroadcastWolSender()

        assert sender.send("invalid") is False


class _RecordingWolSender:
    """단위 테스트 전용 페이크 — 호출 시퀀스를 기록하고 미리 구성된 결과를 순서대로 반환한다."""

    def __init__(self, outcomes: list[bool]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    def send(self, mac_address: str) -> bool:
        self.calls.append(mac_address)
        return self._outcomes.pop(0)


class _RaisingThenSucceedingSender:
    """예외를 던지다가 결국 성공하는 페이크 — 재시도 루프의 예외 포착 경로를 검증한다."""

    def __init__(self, fail_count: int) -> None:
        self._fail_count = fail_count
        self.calls = 0

    def send(self, mac_address: str) -> bool:
        self.calls += 1
        if self.calls <= self._fail_count:
            raise OSError("temporary failure")
        return True


class TestSendWithRetry:
    """REQ-ATA-011: 실패 시 최대 3회까지 재시도.
    AC-ATA-012(REQ-ATA-014): 이미 깨어있는 대상에도 부가 효과 없이 정상 진행."""

    def test_succeeds_on_first_attempt(self):
        sender = _RecordingWolSender([True])

        result = send_with_retry(sender, "AA:BB:CC:DD:EE:FF", max_retries=3)

        assert result == WolResult(success=True, attempts=1)
        assert sender.calls == ["AA:BB:CC:DD:EE:FF"]

    def test_succeeds_after_retries(self):
        sender = _RecordingWolSender([False, False, True])

        result = send_with_retry(sender, "AA:BB:CC:DD:EE:FF", max_retries=3)

        assert result == WolResult(success=True, attempts=3)
        assert len(sender.calls) == 3

    def test_fails_after_max_retries_exhausted(self):
        sender = _RecordingWolSender([False, False, False])

        result = send_with_retry(sender, "AA:BB:CC:DD:EE:FF", max_retries=3)

        assert result.success is False
        assert result.attempts == 3
        assert result.error is not None
        assert len(sender.calls) == 3

    def test_does_not_retry_beyond_max(self):
        sender = _RecordingWolSender([False, False, False])

        send_with_retry(sender, "AA:BB:CC:DD:EE:FF", max_retries=3)

        assert len(sender.calls) == 3  # 4번째 시도가 없어야 한다

    def test_captures_exception_as_error_and_continues_retrying(self):
        sender = _RaisingThenSucceedingSender(fail_count=2)

        result = send_with_retry(sender, "AA:BB:CC:DD:EE:FF", max_retries=3)

        assert result.success is True
        assert sender.calls == 3

    def test_idempotent_when_target_already_awake(self):
        """AC-ATA-012: 이미 깨어있는 대상 — WoL은 프로토콜 특성상 무시되고 send()는
        여전히 성공(True)을 반환한다(하드웨어/OS 수준에서 처리, 별도 코드 분기 불필요)."""
        sender = _RecordingWolSender([True])

        result = send_with_retry(sender, "AA:BB:CC:DD:EE:FF", max_retries=3)

        assert result.success is True
        assert result.attempts == 1
