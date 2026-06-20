"""law.go.kr에서 다운로드한 법령 PDF 파일 파서 (API 키 불필요).

법령 PDF 다운로드 방법:
  1. https://www.law.go.kr 접속
  2. 검색창에 법령명 입력 (예: 주택임대차보호법)
  3. 법령 클릭 → 상단 [다운로드] 버튼 → PDF 선택
  4. 저장 위치: data/raw/pdf/ 디렉터리
     파일명 규칙: {법령명}.pdf  (예: 주택임대차보호법.pdf)

대상 법령 PDF 6개:
  - 주택임대차보호법.pdf
  - 상가건물임대차보호법.pdf         (상가건물 임대차보호법)
  - 근로기준법.pdf
  - 소비자기본법.pdf
  - 전자상거래법.pdf                 (전자상거래 등에서의 소비자보호에 관한 법률)
  - 민법.pdf                        (민법 전체 — 너무 크면 계약편만)

PDF 구조 특징:
  - 조(條) 번호: '제N조', '제N조의N' 형태
  - 항(項): ①②③... 원문자 또는 숫자+항
  - 법령명과 시행일자는 PDF 첫 페이지에 포함
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from .statute_collector import StatuteArticle

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False


PDF_DIR_DEFAULT = Path("data/raw/pdf")

# law name from filename mapping (파일명 → 법령명)
FILENAME_TO_LAWNAME: dict[str, str] = {
    "주택임대차보호법": "주택임대차보호법",
    "상가건물임대차보호법": "상가건물 임대차보호법",
    "상가건물 임대차보호법": "상가건물 임대차보호법",
    "근로기준법": "근로기준법",
    "소비자기본법": "소비자기본법",
    "전자상거래법": "전자상거래 등에서의 소비자보호에 관한 법률",
    "민법": "민법",
}


def _extract_pdf_text(pdf_path: Path) -> str:
    """PDF 전체 텍스트 추출."""
    if not _PYPDF_AVAILABLE:
        raise ImportError("pip install pypdf 를 먼저 실행하세요.")
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _normalize_text(text: str) -> str:
    """PDF 추출 텍스트 정제.
    - 줄바꿈 중복 제거
    - 페이지 번호 제거
    - 헤더/푸터 제거
    """
    # 연속된 공백/줄바꿈 정리
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 페이지 번호 패턴 제거 (예: "- 3 -", "3 / 15" 등)
    text = re.sub(r"[-‐]\s*\d+\s*[-‐]", "", text)
    text = re.sub(r"\d+\s*/\s*\d+", "", text)
    return text.strip()


def _parse_articles_from_text(text: str, law_name: str) -> list[StatuteArticle]:
    """법령 전체 텍스트 → 조(條) 단위 분할.

    패턴: '제N조' 또는 '제N조의N' (항목 경계)
    """
    text = _normalize_text(text)

    # 시행일자 추출
    enf_date = ""
    m = re.search(r"시행\s*[\[【]?\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})", text)
    if m:
        enf_date = f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"

    # 법령ID: 파일명 기반 임시 ID (API 없으므로)
    law_id = re.sub(r"\s+", "_", law_name)

    # 조문 시작 위치 찾기
    art_pattern = re.compile(
        r"제\s*(\d+)\s*조(?:의\s*(\d+))?\s*[\(（]?([^①②③④⑤\n\r]*?)[\)）]?"
    )
    matches = list(art_pattern.finditer(text))

    if not matches:
        # 조문 경계를 못 찾으면 전체를 하나의 청크로
        return [StatuteArticle(
            law_id=law_id,
            law_name=law_name,
            article_no="전문",
            article_title="",
            content=text[:3000],
            enforcement_date=enf_date,
        )]

    articles: list[StatuteArticle] = []
    for i, m in enumerate(matches):
        num1 = m.group(1)
        num2 = m.group(2)
        raw_title = (m.group(3) or "").strip().strip("()（）")

        art_no = f"제{int(num1)}조"
        if num2:
            art_no += f"의{int(num2)}"

        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if len(content) < 8:
            continue

        # 항 단위 청크 분리
        para_chunks = _split_paragraphs(content)

        articles.append(StatuteArticle(
            law_id=law_id,
            law_name=law_name,
            article_no=art_no,
            article_title=raw_title,
            content=content,
            enforcement_date=enf_date,
            chunks=para_chunks,
        ))

    return articles


def _split_paragraphs(text: str) -> list[str]:
    """①②③ 원문자 또는 '1.', '1)' 기준으로 항 분할."""
    para_pat = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]")
    splits = list(para_pat.finditer(text))
    if not splits:
        return []
    paras = []
    for i, s in enumerate(splits):
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        para = text[s.start():end].strip()
        if para:
            paras.append(para)
    return paras


class StatutePDFLoader:
    """law.go.kr 다운로드 PDF → StatuteArticle 리스트."""

    def __init__(self, pdf_dir: Path, processed_dir: Path):
        self.pdf_dir = pdf_dir
        self.processed_dir = processed_dir
        pdf_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

    def _law_name_from_stem(self, stem: str) -> str:
        return FILENAME_TO_LAWNAME.get(stem, stem)

    def load_all_pdfs(self, target_laws: list[dict] | None = None) -> list[StatuteArticle]:
        """pdf_dir의 모든 PDF 파일을 읽어 StatuteArticle 리스트 반환."""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[WARN] PDF 파일이 없습니다: {self.pdf_dir}")
            print("  law.go.kr에서 대상 법령 PDF를 다운로드해 해당 폴더에 저장하세요.")
            return []

        all_articles: list[StatuteArticle] = []
        for pdf_path in pdf_files:
            law_name = self._law_name_from_stem(pdf_path.stem)
            print(f"  PDF 파싱: {pdf_path.name} → {law_name}")
            try:
                text = _extract_pdf_text(pdf_path)
                articles = _parse_articles_from_text(text, law_name)
                print(f"    → {len(articles)}개 조문")
                all_articles.extend(articles)
            except Exception as e:
                print(f"    [오류] {e}")

        out_path = self.processed_dir / "statutes.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for art in all_articles:
                f.write(json.dumps(asdict(art), ensure_ascii=False) + "\n")
        print(f"총 {len(all_articles)}개 조문 → {out_path}")
        return all_articles
