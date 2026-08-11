"""Wake-on-LAN 매직패킷 송신 (SPEC-ANALYZER-TRAIN-AUTOMATION-001 §2.1, REQ-ATA-010/011/014).

`WolSender`를 `typing.Protocol`로 추상화해 특정 네트워킹 구현(컨테이너 네트워크
모드, 서드파티 WoL 라이브러리 등)에 하드커플링되지 않도록 한다(REQ-ATA-010). 실
송신 부분을 인터페이스 뒤로 격리함으로써, 이 모듈이 담당하는 재시도(REQ-ATA-011)와
멱등성(REQ-ATA-014) 로직은 실 네트워크 없이 페이크 구현으로 단위 테스트할 수 있다
(spec.md §4.1 설계 근거).

WoL 프로토콜 자체가 이미 깨어있는 대상에 대해 멱등적이므로(매직패킷은 OS/NIC
수준에서 무시됨), 이 모듈은 별도의 중복 방지 로직을 추가하지 않는다(REQ-ATA-014,
shall not).

[plan.md §D 제약, 2026-08-11] `send()`에 전달하는 MAC 주소는 반드시 하드웨어
(번인) MAC이어야 한다 — `ifconfig`류가 보고하는 활성 인터페이스 MAC(macOS Private
Wi-Fi Address로 순환할 수 있음)을 그대로 사용해서는 안 된다. 이 모듈 자체는 전달된
MAC 주소의 출처를 검증하지 않는다 — 하드웨어 MAC 확보 책임은 호출자
(`orchestration.config`)에 있다.
"""

import re
import socket
from dataclasses import dataclass
from typing import Protocol

_MAC_ADDRESS_PATTERN = re.compile(r"^([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$")

_DEFAULT_WOL_PORT = 9
_DEFAULT_BROADCAST_ADDRESS = "255.255.255.255"


class InvalidMacAddressError(ValueError):
    """MAC 주소 형식이 `XX:XX:XX:XX:XX:XX`(콜론 또는 하이픈 구분)와 일치하지 않는다."""


def _normalize_mac_bytes(mac_address: str) -> bytes:
    if not _MAC_ADDRESS_PATTERN.match(mac_address):
        raise InvalidMacAddressError(f"올바르지 않은 MAC 주소 형식: {mac_address!r}")
    hex_digits = mac_address.replace(":", "").replace("-", "")
    return bytes.fromhex(hex_digits)


def build_magic_packet(mac_address: str) -> bytes:
    """WoL 매직패킷 바이트열을 구성한다 — `0xFF` 6바이트 + 대상 MAC 16회 반복(REQ-ATA-010)."""
    mac_bytes = _normalize_mac_bytes(mac_address)
    return b"\xff" * 6 + mac_bytes * 16


@dataclass(frozen=True, slots=True)
class WolResult:
    """WoL 송신 재시도 루프의 최종 결과 — 실패 처리 경로(REQ-ATA-012)가 소비한다."""

    success: bool
    attempts: int
    error: str | None = None


class WolSender(Protocol):
    """WoL 매직패킷 송신 추상화(REQ-ATA-010) — 구체 네트워킹 구현에 하드커플링되지 않는다."""

    def send(self, mac_address: str) -> bool:
        """매직패킷 1회 송신을 시도한다.

        예외를 던지지 않고 성공 여부를 `bool`로 반환하는 것이 기본 계약이지만,
        `send_with_retry()`는 구현체가 예외를 던지는 경우도 재시도 대상 실패로
        포착한다(구현체의 방어적 계약 위반에 대비).
        """
        ...


class UdpBroadcastWolSender:
    """UDP 브로드캐스트로 매직패킷을 송신하는 구체 구현.

    대상 네트워크 세그먼트로 브로드캐스트 주소(기본 `255.255.255.255`, 운영 시
    서브넷 지정 브로드캐스트 사용 권장)에 UDP 데이터그램을 전송한다.
    `network_mode: host` 폴백(REQ-ATA-013)이 필요한지 여부와 무관하게 이
    클래스 자체는 컨테이너 네트워킹 방식에 의존하지 않는다 — 발신 소켓만 다룬다.
    """

    def __init__(
        self,
        broadcast_address: str = _DEFAULT_BROADCAST_ADDRESS,
        port: int = _DEFAULT_WOL_PORT,
    ) -> None:
        self._broadcast_address = broadcast_address
        self._port = port

    def send(self, mac_address: str) -> bool:
        try:
            packet = build_magic_packet(mac_address)
        except InvalidMacAddressError:
            return False

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.sendto(packet, (self._broadcast_address, self._port))
        except OSError:
            return False
        return True


def send_with_retry(sender: WolSender, mac_address: str, max_retries: int = 3) -> WolResult:
    """REQ-ATA-011: WoL 송신 실패 시 최대 `max_retries`회까지 재시도한다.

    "최대 3회까지 재시도"는 총 시도 횟수 상한(초기 시도 포함)으로 해석한다 —
    SSH 재시도(REQ-ATA-021, "10초 간격 최대 6회까지 재시도")와 동일한 해석
    관례를 따른다.
    """
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if sender.send(mac_address):
                return WolResult(success=True, attempts=attempt)
            last_error = "WoL 매직패킷 송신 실패"
        except Exception as exc:  # noqa: BLE001 — 재시도 경계에서 포착, 마지막 오류만 보고
            last_error = str(exc)
    return WolResult(success=False, attempts=max_retries, error=last_error)
