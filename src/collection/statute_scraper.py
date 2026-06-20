"""law.go.kr 법령 텍스트 웹 스크래핑 (API 키 불필요).

law.go.kr은 서버사이드 렌더링이므로 requests만으로 HTML을 가져올 수 있다.
개별 법령 페이지에서 조(條) 단위 텍스트를 추출한다.

URL 패턴:
  https://www.law.go.kr/법령/{법령명}
  예) https://www.law.go.kr/법령/주택임대차보호법

사용 시 주의:
- 서버 부하를 고려해 요청 간 sleep을 충분히 둔다 (기본 1초).
- robots.txt를 준수한다 (law.go.kr는 공공기관으로 수집 허용 범위 내).
- 수집한 원본 HTML을 raw/ 에 캐싱해 재수집을 방지한다.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .statute_collector import StatuteArticle

BASE_URL = "https://www.law.go.kr/법령"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}


def _fetch_law_html(law_name: str, raw_dir: Path, sleep_sec: float = 1.0) -> str:
    cache_path = raw_dir / f"scrape_{law_name.replace('/', '_')}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    url = f"{BASE_URL}/{law_name}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    html = resp.text
    cache_path.write_text(html, encoding="utf-8")
    time.sleep(sleep_sec)
    return html


def _normalize_article_no(raw: str) -> str:
    """'제3조', '제 3조', '3조' 등 → '제3조'."""
    raw = raw.strip()
    m = re.search(r"제?\s*(\d+)\s*조", raw)
    if m:
        return f"제{m.group(1)}조"
    return raw


def _parse_law_html(html: str, law_name: str) -> list[StatuteArticle]:
    """law.go.kr HTML → StatuteArticle 리스트.

    HTML 구조(2024년 기준):
      <div class="law_view"> 또는 <div id="lawMain">
        <p class="jo"> 또는 여러 형태의 조문 마커
      실제 구조가 다를 경우 fallback 로직으로 텍스트 추출.
    """
    soup = BeautifulSoup(html, "lxml")

    # 시행일자 파싱 시도
    enf_date = ""
    for text in soup.stripped_strings:
        m = re.search(r"시행\s+(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})", text)
        if m:
            enf_date = f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
            break

    # 법령ID 추출 시도 (페이지 내 lsiSeq 파라미터)
    law_id = ""
    m = re.search(r"lsiSeq[='\"][:=]?\s*['\"]?(\d+)", html)
    if m:
        law_id = m.group(1)

    articles: list[StatuteArticle] = []

    # --- 시도 1: 조문 전용 컨테이너 ---
    # law.go.kr는 <div class="law_view"> 안에 조문을 나열
    law_view = (
        soup.find("div", class_="law_view")
        or soup.find("div", id="lawMain")
        or soup.find("div", class_="cont_view")
    )
    if not law_view:
        law_view = soup.body or soup

    # 조문 패턴: "제N조" 또는 "제N조의M" 으로 시작하는 텍스트 노드를 기준으로 분할
    full_text = law_view.get_text(separator="\n")
    articles = _split_by_article_pattern(full_text, law_name, law_id, enf_date)

    if not articles:
        # Fallback: 전체 텍스트를 하나의 청크로 저장
        text = full_text.strip()
        if text:
            articles = [StatuteArticle(
                law_id=law_id or law_name,
                law_name=law_name,
                article_no="전문",
                article_title="",
                content=text[:5000],  # 너무 길면 앞부분만
                enforcement_date=enf_date,
            )]

    return articles


def _split_by_article_pattern(
    full_text: str, law_name: str, law_id: str, enf_date: str
) -> list[StatuteArticle]:
    """전체 텍스트를 '제N조' 패턴으로 분할해 조 단위 청크 생성."""
    # 조문 시작 패턴: "제1조", "제1조의2", "제 1 조" 등
    pattern = re.compile(r"(제\s*\d+\s*조(?:의\s*\d+)?)\s*([^\n]*)")

    # 모든 조문 시작 위치 찾기
    matches = list(pattern.finditer(full_text))
    if not matches:
        return []

    articles = []
    for i, m in enumerate(matches):
        art_no_raw = m.group(1).replace(" ", "")  # "제3조의2"
        # 제목: 괄호 안 텍스트 추출
        title_raw = m.group(2).strip()
        title_m = re.match(r"[\(（](.*?)[\)）]", title_raw)
        art_title = title_m.group(1) if title_m else ""

        # 조 번호 정규화
        art_no = _normalize_article_no(art_no_raw)

        # 이 조문의 텍스트 범위: 현재 위치 ~ 다음 조문 시작
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content_raw = full_text[start:end].strip()

        # 너무 짧으면 스킵 (삭제조문 등)
        if len(content_raw) < 5:
            continue

        # 항 단위 분리 (② ③ 등 원문자 기준)
        para_chunks = _split_paragraphs(content_raw)

        articles.append(StatuteArticle(
            law_id=law_id or law_name,
            law_name=law_name,
            article_no=art_no,
            article_title=art_title,
            content=content_raw,
            enforcement_date=enf_date,
            chunks=para_chunks,
        ))

    return articles


def _split_paragraphs(text: str) -> list[str]:
    """① ② ... 원문자를 기준으로 항 분할."""
    # 원문자 범위: ① ~ ㊿ (Unicode)
    para_pattern = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]")
    splits = list(para_pattern.finditer(text))
    if not splits:
        return []
    paras = []
    for i, s in enumerate(splits):
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        paras.append(text[s.start():end].strip())
    return paras


class StatuteScraper:
    """법령 웹 스크래퍼 (API 키 불필요)."""

    def __init__(self, raw_dir: Path, processed_dir: Path, sleep_sec: float = 1.0):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.sleep_sec = sleep_sec
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

    def scrape_statute(self, law_name: str) -> list[StatuteArticle]:
        print(f"  스크래핑: {law_name}")
        try:
            html = _fetch_law_html(law_name, self.raw_dir, self.sleep_sec)
            articles = _parse_law_html(html, law_name)
            print(f"    → {len(articles)}개 조문")
            return articles
        except Exception as e:
            print(f"    [WARN] 실패: {e}")
            return []

    def scrape_all(self, target_laws: list[dict]) -> list[StatuteArticle]:
        all_articles: list[StatuteArticle] = []
        for law_info in target_laws:
            name = law_info["name"]
            articles = self.scrape_statute(name)
            all_articles.extend(articles)

        out_path = self.processed_dir / "statutes.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for art in all_articles:
                f.write(json.dumps(asdict(art), ensure_ascii=False) + "\n")
        print(f"\n총 {len(all_articles)}개 조문 → {out_path}")
        return all_articles
