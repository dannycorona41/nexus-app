"""
NEXUS White Paper Analyzer
===========================
NLP-powered quality scorer for crypto project white papers and developer reports.
Uses FinBERT (financial sentiment) + custom heuristics for:

  Technical depth     - Is there real engineering behind this?
  Tokenomics quality  - Are supply/distribution/utility well-designed?
  Team transparency   - Are the team, advisors, legal structure clear?
  Innovation score    - Genuine novel contribution vs copy/buzzword?
  Red flag detection  - Vague claims, plagiarism patterns, rug signals
  Final grade         - A through F with numeric score 0-100

Install:
  pip install transformers torch pdfplumber requests beautifulsoup4 scikit-learn tiktoken

Usage:
  from nexus_whitepaper_analyzer import WhitepaperAnalyzer
  analyzer = WhitepaperAnalyzer()
  result   = analyzer.analyze("https://example.com/whitepaper.pdf")
  print(result)
"""

import re
import math
import logging
import hashlib
import requests
from dataclasses import dataclass, field
from typing import Optional
from io import BytesIO
from urllib.parse import urlparse

import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

log = logging.getLogger("nexus.analyzer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class AnalysisResult:
    project_name: str
    grade: str                      # A, B, C, D, F
    overall_score: float            # 0-100
    technical_depth: float          # 0-100
    tokenomics_quality: float       # 0-100
    team_transparency: float        # 0-100
    innovation_score: float         # 0-100
    sentiment_score: float          # 0-100 (FinBERT positive sentiment)
    red_flags: list[str] = field(default_factory=list)
    green_flags: list[str] = field(default_factory=list)
    summary: str = ""
    word_count: int = 0
    section_scores: dict = field(default_factory=dict)

    def __str__(self):
        flags_str = ""
        if self.red_flags:
            flags_str += f"\n  Red flags:  {', '.join(self.red_flags[:5])}"
        if self.green_flags:
            flags_str += f"\n  Green flags:{', '.join(self.green_flags[:5])}"
        return (
            f"══════════════════════════════════════════\n"
            f"  NEXUS WHITE PAPER ANALYSIS\n"
            f"══════════════════════════════════════════\n"
            f"  Project   : {self.project_name}\n"
            f"  Grade     : {self.grade}  ({self.overall_score:.1f}/100)\n"
            f"  ──────────────────────────────────────\n"
            f"  Technical : {self.technical_depth:.1f}/100\n"
            f"  Tokenomics: {self.tokenomics_quality:.1f}/100\n"
            f"  Team      : {self.team_transparency:.1f}/100\n"
            f"  Innovation: {self.innovation_score:.1f}/100\n"
            f"  Sentiment : {self.sentiment_score:.1f}/100\n"
            f"  Words     : {self.word_count:,}\n"
            f"  ──────────────────────────────────────\n"
            f"  Summary: {self.summary[:300]}{flags_str}\n"
            f"══════════════════════════════════════════"
        )


# ─────────────────────────────────────────────
# Section Parser
# ─────────────────────────────────────────────

class DocumentParser:
    """Extract and structure text from PDF URLs, local PDFs, or raw text."""

    SECTION_KEYWORDS = {
        "abstract":     ["abstract", "overview", "executive summary", "introduction"],
        "technology":   ["technology", "architecture", "consensus", "protocol", "smart contract",
                         "blockchain", "network", "layer", "sharding", "zk", "zkp", "proof"],
        "tokenomics":   ["tokenomics", "token distribution", "token supply", "vesting",
                         "allocation", "emission", "burn", "staking", "rewards", "economics"],
        "team":         ["team", "founders", "advisors", "leadership", "core contributors", "about us"],
        "roadmap":      ["roadmap", "milestones", "timeline", "q1", "q2", "q3", "q4", "2024", "2025"],
        "use_case":     ["use case", "problem", "solution", "market", "applications", "utility"],
        "security":     ["security", "audit", "smart contract audit", "bug bounty", "formal verification"],
        "legal":        ["legal", "compliance", "regulation", "disclaimer", "jurisdiction"],
    }

    def from_url(self, url: str) -> dict[str, str]:
        """Download and parse a PDF from URL."""
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            if "pdf" in resp.headers.get("Content-Type", "").lower() or url.endswith(".pdf"):
                return self.from_pdf_bytes(BytesIO(resp.content))
            else:
                return self.from_html(resp.text)
        except Exception as e:
            log.error(f"Failed to fetch {url}: {e}")
            return {}

    def from_pdf_bytes(self, source) -> dict[str, str]:
        """Parse PDF into sections."""
        full_text = ""
        try:
            with pdfplumber.open(source) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
        except Exception as e:
            log.error(f"PDF parse error: {e}")
            return {"full": ""}
        return self._segment(full_text)

    def from_pdf_path(self, path: str) -> dict[str, str]:
        with open(path, "rb") as f:
            return self.from_pdf_bytes(BytesIO(f.read()))

    def from_html(self, html: str) -> dict[str, str]:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return self._segment(text)

    def from_text(self, text: str) -> dict[str, str]:
        return self._segment(text)

    def _segment(self, text: str) -> dict[str, str]:
        """Split full text into named sections."""
        sections = {"full": text}
        lower    = text.lower()

        for section, keywords in self.SECTION_KEYWORDS.items():
            for kw in keywords:
                idx = lower.find(kw)
                if idx != -1:
                    # Take 3000 chars after the keyword
                    sections[section] = text[idx: idx + 3000]
                    break

        return sections


# ─────────────────────────────────────────────
# FinBERT Sentiment Analyzer
# ─────────────────────────────────────────────

class FinBERTAnalyzer:
    """FinBERT-based sentiment analysis (positive/negative/neutral)."""

    MODEL_NAME = "ProsusAI/finbert"
    LABEL_MAP  = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self):
        log.info("Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        log.info("FinBERT loaded")

    def analyze_chunk(self, text: str) -> dict[str, float]:
        """Analyze a single chunk (max 512 tokens) → {positive, negative, neutral}."""
        inputs  = self.tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=512, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs   = F.softmax(logits, dim=-1)[0]
        return {
            "positive": float(probs[0]),
            "negative": float(probs[1]),
            "neutral":  float(probs[2]),
        }

    def analyze_long_text(self, text: str, max_chunks: int = 8) -> dict[str, float]:
        """Analyze long text by chunking and averaging sentiment."""
        words  = text.split()
        chunk_size = 350
        chunks = [" ".join(words[i: i + chunk_size])
                  for i in range(0, min(len(words), chunk_size * max_chunks), chunk_size)]
        if not chunks:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

        totals = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for chunk in chunks:
            result = self.analyze_chunk(chunk)
            for k in totals:
                totals[k] += result[k]
        n = len(chunks)
        return {k: v / n for k, v in totals.items()}


# ─────────────────────────────────────────────
# Scoring Heuristics
# ─────────────────────────────────────────────

class ScoringHeuristics:
    """Rule-based scoring for document characteristics."""

    # ── Red Flags ─────────────────────────────

    RED_FLAG_PATTERNS = [
        (r"\b100x\b|\b1000x\b|\bmoon\b|\blamborghini\b",    "Hype language / unrealistic claims",        -15),
        (r"\bguaranteed (returns|profits|gains)\b",           "Guaranteed returns (illegal claim)",        -25),
        (r"\bno whitepaper\b|\bno audit\b",                   "Missing whitepaper/audit",                  -20),
        (r"\banonymous team\b|\bno team\b",                   "Anonymous team",                            -15),
        (r"\bunlimited supply\b",                             "Unlimited supply tokenomics",               -10),
        (r"\b(team|founder|dev).*\b(100|90|80)\s*%",          "Team owns >80% of supply",                  -20),
        (r"\bcopy(cat|right|paste)\b",                        "Content appears copied",                    -15),
        (r"\bno (vesting|lock|lockup)\b",                     "No vesting schedule",                       -10),
        (r"\b(buy now|limited time|act fast)\b",              "Sales pressure language",                   -10),
        (r"\b(ponzi|pyramid)\b",                              "Ponzi/pyramid risk indicators",             -30),
        (r"\bfork of\b|\bclone of\b",                         "Simple fork/clone without innovation",      -10),
    ]

    # ── Green Flags ───────────────────────────

    GREEN_FLAG_PATTERNS = [
        (r"\baudit(ed|ing)?\b.*\b(certik|hacken|trail of bits|openzeppelin|quantstamp)\b",
                                                              "Reputable security audit",                   +20),
        (r"\bopen[- ]?source\b|\bgithub\.com\b",              "Open-source code",                          +10),
        (r"\bphd\b|\bprofessor\b|\bresearch(er)?\b",          "Academic team members",                     +10),
        (r"\bvesting\b.*\b(year|month|cliff)\b",              "Vesting schedule present",                  +10),
        (r"\bkpi\b|\bmetrics\b|\bbenchmark\b",                "KPI / measurable benchmarks",               +8),
        (r"\b(formal verification|zk[-]?proof|zkp)\b",        "Cryptographic innovation",                  +15),
        (r"\b(dao|decentralized governance)\b",               "Decentralized governance",                  +8),
        (r"\b(revenue|fee|sustainable)\b",                    "Revenue / sustainability model",            +10),
        (r"\b(partner|partnership)\b.*\b(microsoft|google|chainlink|polygon|ripple)\b",
                                                              "Credible institutional partnership",         +12),
        (r"\biso\b|\bsoc 2\b|\bcomplian\b",                   "Regulatory compliance",                     +8),
    ]

    def check_red_flags(self, text: str) -> tuple[list[str], float]:
        flags, penalty = [], 0.0
        lower = text.lower()
        for pattern, label, score in self.RED_FLAG_PATTERNS:
            if re.search(pattern, lower):
                flags.append(label)
                penalty += abs(score)
        return flags, penalty

    def check_green_flags(self, text: str) -> tuple[list[str], float]:
        flags, bonus = [], 0.0
        lower = text.lower()
        for pattern, label, score in self.GREEN_FLAG_PATTERNS:
            if re.search(pattern, lower):
                flags.append(label)
                bonus += score
        return flags, bonus

    def score_technical_depth(self, sections: dict) -> float:
        """Score based on technical section quality."""
        tech_text = sections.get("technology", "") + sections.get("security", "")
        if not tech_text:
            return 20.0

        score = 30.0
        indicators = [
            (r"\b(consensus|pos|pow|pbft|bft|tendermint)\b",       15),
            (r"\b(smart contract|solidity|rust|move|cairo)\b",     10),
            (r"\b(sharding|rollup|layer 2|state channel)\b",       12),
            (r"\b(tps|transactions per second|throughput|latency)\b", 8),
            (r"\b(merkle|patricia|hash|cryptograph)\b",             8),
            (r"\bformal proof\b|\bverif(y|ied|ication)\b",         12),
            (r"\b(peer[-\s]reviewed|arxiv|whitepaper)\b",          10),
            (r"\b(testnet|mainnet|devnet)\b",                       5),
        ]
        lower = tech_text.lower()
        for pattern, pts in indicators:
            if re.search(pattern, lower):
                score += pts

        # Penalize if it's all buzzwords, no substance
        word_count = len(tech_text.split())
        if word_count < 200:
            score -= 20
        return min(100.0, max(0.0, score))

    def score_tokenomics(self, sections: dict) -> float:
        """Score tokenomics quality: supply, distribution, utility, vesting."""
        tok_text = sections.get("tokenomics", "")
        if not tok_text:
            return 15.0

        score = 30.0
        lower = tok_text.lower()
        checks = [
            (r"\b(vesting|cliff|lock|lockup)\b",                   15),
            (r"\b(circulating supply|max supply|total supply)\b",  10),
            (r"\b(burn|deflationary|buyback)\b",                   10),
            (r"\b(utility|governance|staking|reward)\b",           10),
            (r"\b(treasury|ecosystem fund|foundation)\b",           8),
            (r"\b(distribution|allocation)\b.*\b(%|percent)\b",    8),
            (r"\b(audit|verified|transparent)\b",                  10),
        ]
        for pattern, pts in checks:
            if re.search(pattern, lower):
                score += pts

        # Negative: concentration risk
        team_alloc = re.search(r"team.*?(\d{1,3})\s*%", lower)
        if team_alloc:
            alloc = int(team_alloc.group(1))
            if alloc > 50:
                score -= 30
            elif alloc > 30:
                score -= 15

        return min(100.0, max(0.0, score))

    def score_team_transparency(self, sections: dict) -> float:
        """Score team section: named people, backgrounds, social links."""
        team_text = sections.get("team", "")
        legal_text = sections.get("legal", "")
        if not team_text:
            return 10.0

        score = 20.0
        lower = team_text.lower()
        checks = [
            (r"\blinkedin\b|\btwitter\b|\bgithub\b",           10),
            (r"\b(ceo|cto|cfo|founder|co-founder)\b",          10),
            (r"\b(university|mit|stanford|oxford|phd|msc)\b",  12),
            (r"\b(formerly|previously|experience)\b",           8),
            (r"\b(advisor|board)\b",                           8),
            (r"\b(legal|entity|incorporated|llc|ltd)\b",       10),
            (r"\b(kyc|doxxed)\b",                              15),
        ]
        for pattern, pts in checks:
            if re.search(pattern, lower + legal_text.lower()):
                score += pts

        return min(100.0, max(0.0, score))

    def score_innovation(self, sections: dict) -> float:
        """Is this genuinely novel or just a fork with buzzwords?"""
        full = sections.get("full", "")
        lower = full.lower()
        score = 30.0

        novel_indicators = [
            (r"\bnovel\b|\binnovation\b|\bfirst\s+(in|to)\b",    10),
            (r"\bpatent\b|\bip\b|\bproprietary\b",               10),
            (r"\bnew\s+(approach|algorithm|protocol|mechanism)\b", 12),
            (r"\bsolves?\b.*\b(problem|challenge|issue)\b",        8),
            (r"\b(cross[-\s]?chain|interoperab)\b",               8),
            (r"\bai\b|\bmachine learning\b|\bfederated\b",        8),
        ]
        copy_indicators = [
            (r"\bcopy of\b|\bfork of\b|\binspired by\b",         -20),
            (r"the same as\b",                                    -10),
            (r"\bexactly like\b",                                 -10),
        ]
        for pattern, pts in novel_indicators + copy_indicators:
            if re.search(pattern, lower):
                score += pts

        # Length heuristic: thin whitepapers lack depth
        words = len(full.split())
        if words < 1000:
            score -= 25
        elif words > 5000:
            score += 10

        return min(100.0, max(0.0, score))

    def extract_project_name(self, sections: dict) -> str:
        """Try to extract the project name from the document."""
        abstract = sections.get("abstract", sections.get("full", ""))[:500]
        # Look for capitalized project names
        match = re.search(r"(?:project|protocol|platform|token|network)[:\s]+([A-Z][A-Za-z0-9]+)", abstract)
        if match:
            return match.group(1)
        # Fallback: first title-cased phrase
        words = abstract.split()
        for w in words:
            if w[0].isupper() and len(w) > 3:
                return w
        return "Unknown Project"


# ─────────────────────────────────────────────
# Main Analyzer
# ─────────────────────────────────────────────

class WhitepaperAnalyzer:
    """
    Full white paper analysis pipeline.
    Combines FinBERT sentiment + rule-based heuristics → grade A–F.
    """

    GRADE_THRESHOLDS = [
        (85, "A"),
        (70, "B"),
        (55, "C"),
        (40, "D"),
        (0,  "F"),
    ]

    WEIGHTS = {
        "technical":   0.30,
        "tokenomics":  0.25,
        "team":        0.20,
        "innovation":  0.15,
        "sentiment":   0.10,
    }

    def __init__(self, use_finbert: bool = True):
        self.parser     = DocumentParser()
        self.heuristics = ScoringHeuristics()
        self.finbert    = FinBERTAnalyzer() if use_finbert else None

    def analyze(self, source: str) -> AnalysisResult:
        """
        Analyze from URL, local PDF path, or raw text string.
        Returns a full AnalysisResult.
        """
        log.info(f"Analyzing: {source[:80]}...")

        # Parse
        if source.startswith("http"):
            sections = self.parser.from_url(source)
        elif source.endswith(".pdf"):
            sections = self.parser.from_pdf_path(source)
        else:
            sections = self.parser.from_text(source)

        full_text  = sections.get("full", "")
        word_count = len(full_text.split())

        if word_count < 100:
            log.warning("Document too short — scoring as F")
            return AnalysisResult(
                project_name="Unknown",
                grade="F",
                overall_score=5.0,
                technical_depth=5.0,
                tokenomics_quality=5.0,
                team_transparency=5.0,
                innovation_score=5.0,
                sentiment_score=50.0,
                red_flags=["Document too short or inaccessible"],
                summary="Could not extract sufficient content for analysis.",
                word_count=word_count,
            )

        # Score components
        technical   = self.heuristics.score_technical_depth(sections)
        tokenomics  = self.heuristics.score_tokenomics(sections)
        team        = self.heuristics.score_team_transparency(sections)
        innovation  = self.heuristics.score_innovation(sections)

        # FinBERT sentiment
        if self.finbert:
            sentiment_raw = self.finbert.analyze_long_text(full_text[:8000])
            sentiment_score = sentiment_raw["positive"] * 100
        else:
            sentiment_score = 50.0

        # Flags
        red_flags,   penalty = self.heuristics.check_red_flags(full_text)
        green_flags, bonus   = self.heuristics.check_green_flags(full_text)

        # Weighted score
        raw_score = (
            technical   * self.WEIGHTS["technical"]  +
            tokenomics  * self.WEIGHTS["tokenomics"] +
            team        * self.WEIGHTS["team"]        +
            innovation  * self.WEIGHTS["innovation"]  +
            sentiment_score * self.WEIGHTS["sentiment"]
        )
        adjusted = max(0.0, min(100.0, raw_score - penalty * 0.3 + bonus * 0.2))

        # Grade
        grade = "F"
        for threshold, letter in self.GRADE_THRESHOLDS:
            if adjusted >= threshold:
                grade = letter
                break

        project_name = self.heuristics.extract_project_name(sections)
        summary      = self._build_summary(sections, grade, red_flags, green_flags)

        return AnalysisResult(
            project_name     = project_name,
            grade            = grade,
            overall_score    = round(adjusted, 2),
            technical_depth  = round(technical, 2),
            tokenomics_quality=round(tokenomics, 2),
            team_transparency= round(team, 2),
            innovation_score = round(innovation, 2),
            sentiment_score  = round(sentiment_score, 2),
            red_flags        = red_flags,
            green_flags      = green_flags,
            summary          = summary,
            word_count       = word_count,
            section_scores   = {
                "technical":   technical,
                "tokenomics":  tokenomics,
                "team":        team,
                "innovation":  innovation,
                "sentiment":   sentiment_score,
                "red_penalty": penalty,
                "green_bonus": bonus,
            }
        )

    def analyze_batch(self, sources: list[str]) -> list[AnalysisResult]:
        """Analyze multiple white papers and return sorted by score."""
        results = [self.analyze(src) for src in sources]
        return sorted(results, key=lambda r: r.overall_score, reverse=True)

    def _build_summary(self, sections: dict, grade: str, red_flags: list, green_flags: list) -> str:
        abstract = sections.get("abstract", sections.get("full", ""))[:400]
        summary  = f"Grade {grade}. "
        if abstract:
            summary += abstract[:200].strip() + "... "
        if red_flags:
            summary += f"Concerns: {'; '.join(red_flags[:3])}. "
        if green_flags:
            summary += f"Strengths: {'; '.join(green_flags[:3])}."
        return summary.strip()

    def score_to_nexus_factor(self, result: AnalysisResult) -> float:
        """
        Convert analysis result to NEXUS tokenomics/sentiment factor (0-100).
        Used by nexus_signal_engine.py.
        """
        return result.overall_score


# ─────────────────────────────────────────────
# Developer Report Analyzer
# ─────────────────────────────────────────────

class DevReportAnalyzer:
    """
    Analyze developer activity reports from developerreport.com
    and GitHub API for NEXUS developer activity scoring.
    """

    def __init__(self, github_token: str | None = None):
        self.github_token = github_token
        self.headers = {"Authorization": f"token {github_token}"} if github_token else {}

    def get_github_metrics(self, owner: str, repo: str) -> dict:
        """Fetch commit activity, contributors, and issue velocity from GitHub API."""
        base = f"https://api.github.com/repos/{owner}/{repo}"
        metrics = {}

        try:
            # Commit activity (last 52 weeks)
            resp = requests.get(f"{base}/stats/commit_activity",
                                headers=self.headers, timeout=15)
            if resp.status_code == 200:
                weeks = resp.json()
                recent_commits = sum(w.get("total", 0) for w in weeks[-12:])  # Last 3 months
                metrics["commits_90d"] = recent_commits

            # Contributors
            resp = requests.get(f"{base}/stats/contributors",
                                headers=self.headers, timeout=15)
            if resp.status_code == 200:
                contributors = resp.json()
                active = sum(1 for c in contributors
                             if sum(w.get("c", 0) for w in c.get("weeks", [])[-12:]) > 0)
                metrics["active_contributors"] = active

            # Repo info
            resp = requests.get(base, headers=self.headers, timeout=15)
            if resp.status_code == 200:
                info = resp.json()
                metrics["stars"]         = info.get("stargazers_count", 0)
                metrics["forks"]         = info.get("forks_count", 0)
                metrics["open_issues"]   = info.get("open_issues_count", 0)
                metrics["last_push"]     = info.get("pushed_at", "")

        except Exception as e:
            log.warning(f"GitHub API error: {e}")

        return metrics

    def compute_velocity_score(self, metrics: dict) -> float:
        """Score developer velocity 0-100."""
        score = 20.0
        commits = metrics.get("commits_90d", 0)
        devs    = metrics.get("active_contributors", 0)
        stars   = metrics.get("stars", 0)

        score += min(40, commits / 3)     # 120+ commits/90d = full 40 pts
        score += min(20, devs * 4)        # 5+ devs = full 20 pts
        score += min(10, math.log10(max(1, stars)) * 5)  # Log scale for stars
        return min(100.0, max(0.0, score))


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def main():
    import sys
    import os

    use_finbert = "--no-finbert" not in sys.argv
    analyzer    = WhitepaperAnalyzer(use_finbert=use_finbert)

    # Default test sources
    sources = [
        arg for arg in sys.argv[1:] if not arg.startswith("--")
    ]

    if not sources:
        # Demo mode: score a sample URL
        sources = [
            "https://ripple.com/files/ripple_consensus_whitepaper.pdf",
        ]

    results = analyzer.analyze_batch(sources)
    for r in results:
        print(r)
        print()

    # Export scores for NEXUS signal engine
    export = [
        {
            "project":        r.project_name,
            "grade":          r.grade,
            "score":          r.overall_score,
            "technical":      r.technical_depth,
            "tokenomics":     r.tokenomics_quality,
            "team":           r.team_transparency,
            "innovation":     r.innovation_score,
            "red_flags":      r.red_flags,
            "green_flags":    r.green_flags,
            "nexus_factor":   analyzer.score_to_nexus_factor(r),
        }
        for r in results
    ]

    import json
    out_path = os.path.join(os.path.dirname(__file__), "whitepaper_scores.json")
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    log.info(f"Scores exported to {out_path}")


if __name__ == "__main__":
    main()
