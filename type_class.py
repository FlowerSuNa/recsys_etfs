from enum import Enum
from decimal import Decimal
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field

# 상태 정보를 저장하는 State 클래스
class State(TypedDict):
    question: str          # 사용자 입력 질문
    user_profile: dict     # 사용자 프로필 정보
    entity_info: str       # 개체 검색 결과
    output:list            # 쿼리 생성 에이전트 내용 
    query: str             # 생성된 SQL 쿼리
    candidates: list       # 후보 ETF 목록
    rankings: list         # 순위가 매겨진 ETF 목록
    explanation: str       # 추천 이유 설명
    final_answer: str      # 최종 추천 답변

class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"   # 보수적
    MODERATE = "moderate"           # 중립적
    AGGRESSIVE = "aggressive"       # 공격적

class ExperienceLevel(str, Enum):
    BEGINNER = "beginner"           # 초보자
    INTERMEDIATE = "intermediate"   # 중수
    ADVANCED = "advanced"           # 고수

class InvestmentHorizon(str, Enum):
    SHORT = "short"     # 0~1년
    MEDIUM = "medium"   # 1~3년
    LONG = "long"       # 3년 이상


class InvestmentProfile(BaseModel):
    risk_tolerance: RiskTolerance = Field(
        description="Investor's risk tolerance (conservative/moderate/aggressive)"
    )
    experience_level: ExperienceLevel = Field(
        description="Investor's experience level (beginner/intermediate/advanced)"
    )
    investment_horizon: InvestmentHorizon = Field(
        description="투자 기간 (short/medium/long)"
    )
    investment_goal: str = Field(
        description="Main investment goal (e.g., retirement, capital growth, passive income)"
    )
    preferred_sectors: List[str] = Field(
        description="List of preferred or interested sectors/industries for investment"
    )
    excluded_sectors: List[str] = Field(
        description="List of sectors or industries to exclude from investment"
    )
    monthly_investment: int = Field(
        description="Available monthly investment amount in KRW"
    )

class QueryOutput(TypedDict):
    """ Structured output for a generated SQL query, its explanation, and the result set. """
    query: Annotated[
        str, ..., 
        "Syntactically valid SQL query."
    ]
    explanation: Annotated[
        str, ..., 
        "Explanation of how the query was constructed and the rationale behind the column and filter selections."
    ]
    candidates: Annotated[
        List[Dict[str, Any]], ..., 
        "The structured result set from executing the SQL query. Each dictionary represents a row with column-value pairs."
    ]

class ETFRanking(TypedDict):
    """Individual ETF ranking result"""
    rank: Annotated[int, ..., "Ranking position (1-5)"]
    etf_code: Annotated[str, ..., "ETF 종목코드 (6-digit)"]
    etf_name: Annotated[str, ..., "ETF 종목명"]
    score: Annotated[float, ..., "Composite score (0-100)"]
    ranking_reason: Annotated[str, ..., "Explanation for the ranking (in Korean)"]

class ETFRankingResult(TypedDict):
    """Ranked ETFs"""
    rankings: List[ETFRanking]

class ETFRecommendation(BaseModel):
   """ Details for each recommended ETF """
   etf_code: str = Field(..., description="ETF 종목코드 (6-digit)")
   etf_name: str = Field(..., description="ETF 종목명")
   allocation: Decimal = Field(..., description="Recommended allocation percentage (0-100%)")
   description: str = Field(..., description="Brief summary of the ETF and its strategy in Korean")
   key_points: List[str] = Field(..., description="Key investment highlights in Korean")
   risks: List[str] = Field(..., description="Main risk factors to consider in Korean")

class RecommendationExplanation(BaseModel):
   """ inal recommendation summary for the ETF portfolio """
   overview: str = Field(..., description="Overall investment strategy summary in Korean")
   recommendations: List[ETFRecommendation] = Field(..., description="List of ETF recommendations")
   considerations: List[str] = Field(..., description="General notes or caveats for the investor in Korean")
   
   # 마크다운 포맷으로 출력
   def to_markdown(self) -> str:
      """Convert explanation to markdown format"""
      markdown = [
            "# ETF 포트폴리오 추천",
            "",
            "## 투자 전략 개요",
            self.overview,
            "",
            "## 추천 ETF 포트폴리오",
            ""
      ]
      
      # 포트폴리오 구성 비율
      markdown.extend([
            "| ETF | 종목코드 | 추천비중 |",
            "|-----|----------|----------|"
      ])
      
      for rec in self.recommendations:
         markdown.append(
            f"| {rec.etf_name} | {rec.etf_code} | {rec.allocation}% |"
            )
      
      # ETF 상세 설명
      markdown.append("\n## ETF 상세 설명\n")
      
      for rec in self.recommendations:
         markdown.extend([
               f"### {rec.etf_name} ({rec.etf_code})",
               rec.description,
               "",
               "**주요 투자 포인트:**",
               "".join([f"\n* {point}" for point in rec.key_points]),
               "",
               "**투자 위험:**",
            "".join([f"\n* {risk}" for risk in rec.risks]),
         ""
         ])
      
      # 투자 리스크 고려사항
      markdown.extend([
            "## 투자 시 고려사항",
            "".join([f"\n* {item}" for item in self.considerations]),
            ""
      ])
      return "\n".join(markdown)