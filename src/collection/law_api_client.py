"""국가법령정보 공동활용 Open API 클라이언트 (open.law.go.kr).

API 신청: https://open.law.go.kr → 회원가입 → API 활용신청 (1~2일 내 승인)
인증 방식: OC 파라미터에 가입 이메일 주소 사용

요청 패턴:
  법령 검색:  GET https://www.law.go.kr/DRF/lawSearch.do
               ?OC=이메일&target=law&type=XML&query=주택임대차보호법&display=20&page=1
  법령 본문:  GET https://www.law.go.kr/DRF/lawService.do
               ?OC=이메일&target=law&type=XML&MST={법령일련번호}
  판례 검색:  GET https://www.law.go.kr/DRF/lawSearch.do
               ?OC=이메일&target=prec&type=XML&query=임대차보증금반환&display=20&page=1
  판례 본문:  GET https://www.law.go.kr/DRF/lawService.do
               ?OC=이메일&target=prec&type=XML&ID={판례일련번호}   ← MST 아님, ID

응답 XML:
  법령 검색:  <LawSearch><law><법령일련번호>...</법령일련번호></law></LawSearch>
  법령 본문:  <법령><기본정보>...</기본정보><조문><조문단위>...</조문단위></조문></법령>
  판례 검색:  <PrecSearch><prec><판례일련번호>...</판례일련번호></prec></PrecSearch>
  판례 본문:  <PrecService>...</PrecService>

환경변수: LAW_OC=가입한이메일주소
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

BASE_SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
BASE_SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"

_HTML_MARKERS = ("<!DOCTYPE", "<html", "<HTML")
_XML_NOT_FOUND_MARKER = "일치하는"  # "<Law>일치하는 판례가 없습니다</Law>" 패턴


def _is_html(text: str) -> bool:
    """API가 오류 HTML을 반환했을 때 감지."""
    stripped = text.lstrip()
    return any(stripped.startswith(m) for m in _HTML_MARKERS)


def _is_error_response(text: str) -> bool:
    """HTML이 아니지만 오류인 XML 응답 감지 (일치하는 결과 없음 등)."""
    if _is_html(text):
        return True
    return _XML_NOT_FOUND_MARKER in text[:200]


@dataclass
class LawAPIConfig:
    oc: str                   # 가입 이메일 주소
    sleep_seconds: float = 0.8
    display: int = 20         # 페이지당 결과 수 (최대 100)
    timeout: int = 30
    max_retries: int = 2


class LawAPIClient:
    """open.law.go.kr 국가법령정보 공동활용 API 클라이언트."""

    def __init__(self, config: LawAPIConfig | None = None):
        if config is None:
            oc = os.environ.get("LAW_OC", "")
            if not oc:
                raise ValueError(
                    "LAW_OC 환경변수가 없습니다. "
                    ".env 파일에 LAW_OC=가입한이메일주소 를 설정하세요.\n"
                    "API 신청: https://open.law.go.kr"
                )
            config = LawAPIConfig(oc=oc)
        self.config = config

    # ------------------------------------------------------------------ #
    # 법령                                                                #
    # ------------------------------------------------------------------ #

    def search_laws(self, law_name: str, page: int = 1) -> str:
        """법령명으로 법령 목록 검색. XML 응답 텍스트 반환."""
        params = {
            "OC": self.config.oc,
            "target": "law",
            "type": "XML",
            "query": law_name,
            "display": self.config.display,
            "page": page,
        }
        return self._get(BASE_SEARCH_URL, params)

    def get_law_detail(self, law_serial_no: str) -> str:
        """법령일련번호로 법령 본문(조항호목) 전체 조회. XML 응답 텍스트 반환.

        파라미터명: MST (값은 <법령일련번호> 태그에서 가져온 번호)
        """
        params = {
            "OC": self.config.oc,
            "target": "law",
            "type": "XML",
            "MST": law_serial_no,
        }
        return self._get(BASE_SERVICE_URL, params)

    # ------------------------------------------------------------------ #
    # 판례                                                                #
    # ------------------------------------------------------------------ #

    def search_precedents(self, keyword: str, page: int = 1) -> str:
        """키워드로 판례 목록 검색. XML 응답 텍스트 반환."""
        params = {
            "OC": self.config.oc,
            "target": "prec",
            "type": "XML",
            "query": keyword,
            "display": self.config.display,
            "page": page,
        }
        return self._get(BASE_SEARCH_URL, params)

    def get_precedent_detail(self, prec_id: str) -> str:
        """판례일련번호로 판례 본문 조회. XML 응답 텍스트 반환.

        파라미터명: ID (MST 아님 — 잘못 사용하면 오류 HTML 반환됨)
        """
        params = {
            "OC": self.config.oc,
            "target": "prec",
            "type": "XML",
            "ID": prec_id,
        }
        return self._get(BASE_SERVICE_URL, params)

    # ------------------------------------------------------------------ #
    # 내부 유틸                                                           #
    # ------------------------------------------------------------------ #

    def _get(self, url: str, params: dict) -> str:
        """GET 요청. 재시도 포함. HTML 오류 응답은 예외 발생."""
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=self.config.timeout)
                resp.raise_for_status()
                text = resp.text
                if _is_html(text):
                    raise ValueError(
                        f"API가 HTML을 반환했습니다 (파라미터 오류 가능성). "
                        f"params={params}"
                    )
                time.sleep(self.config.sleep_seconds)
                return text
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < self.config.max_retries:
                    wait = self.config.sleep_seconds * (attempt + 2)
                    print(f"[RETRY {attempt+1}] {type(e).__name__} — {wait:.1f}초 후 재시도")
                    time.sleep(wait)
                else:
                    raise
