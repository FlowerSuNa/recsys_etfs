from typing import List
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

import json
import gradio as gr

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain import hub
from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, StateGraph 

from prompts import (
    PROFILE_TEMPLATE,
    QUERY_TEMPLATE,
    SUMMARY_TEMPLATE,
    RANKING_TEMPLATE,
    EXPLANATION_TEMPLATE
)
from callbacks import PerformanceMonitoringCallback
from retriever import get_retriever_tools
from type_class import (
    State,
    InvestmentProfile,
    QueryOutput,
    ETFRankingResult,
    RecommendationExplanation
)
from testset import TEST01, TEST02, TEST03, TEST04, TEST05


##################################################################
# 환경 설정 / 데이터베이스 연결
##################################################################

load_dotenv()

db = SQLDatabase.from_uri(
    "sqlite:///etf_database.db",
    include_tables=["ETFS_WITH_INFO"]
)

# Langfuse 콜백 핸들러 생성
langfuse_handler = CallbackHandler()

# 콜백 핸들러 생성
performance_handler = PerformanceMonitoringCallback()


##################################################################
# 고유명사 DB 검색
##################################################################

agent_executor = get_retriever_tools(db)


##################################################################
# 사용자 프로필 분석
##################################################################

def analyze_profile(state: State) -> dict:
    """사용자 질문을 분석하여 투자 프로필 생성"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROFILE_TEMPLATE),
        ("human", "사용자 질문: {question}")
    ])
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.1,
        top_p=0.7
    ).with_structured_output(InvestmentProfile)
    chain = prompt | llm
    response = chain.invoke(
        state["question"],
        config={
            "callbacks": [langfuse_handler, performance_handler],
            "metadata": {
                "langfuse_tags": ["etf-rag", "analyze_profile"],
                "model": "gpt-4.1",
                "temperature": 0.1,
                "top_p": 0.7,
            }
        }
    )
    return {"user_profile": dict(response)}


##################################################################
# ETF 검색
##################################################################

def write_query(state: State):
    """Generate SQL query to fetch information."""
    # 개체 정보 획득
    entity_info = agent_executor.invoke(
        state["question"] + json.dumps(state["user_profile"])
    )

    # Text2SQL 에이전트 프롬프트 정의
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    system_prompt = prompt_template.format(dialect=db.dialect, top_k=5)
    human_prompt = HumanMessage(content=QUERY_TEMPLATE.format(
        question=state["question"],
        user_profile=state["user_profile"],
        entity_info=entity_info["output"]
    ))

    # Text2SQL 에이전트 모델 정의
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )

    # Text2SQL 에이전트 Tool 정의
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # Text2SQL 에이전트 실행
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    output = agent.invoke(
        {"messages": [human_prompt]},
        config={
            "callbacks": [langfuse_handler, performance_handler],
            "metadata": {
                "langfuse_tags": ["etf-rag", "write_query"],
                "model": "gemini-2.5-flash",
                "temperature": 0.1,
            }
        }
    )
    return {
        "output": output["messages"], 
        "entity_info": entity_info["output"]
    }

def summary_query(state: State):
    """ Query and Candidate Set Structuring """
    structured_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    ).with_structured_output(QueryOutput)

    result = structured_llm.invoke(
        state["output"] + [HumanMessage(content=SUMMARY_TEMPLATE)],
        config={
            "callbacks": [langfuse_handler, performance_handler],
            "metadata": {
                "langfuse_tags": ["etf-rag", "query_summary"],
                "model": "gpt-4.1-mini",
                "temperature": 0,
            }
        }
    )
    return {
        "query": result["query"], 
        "explanation": result["explanation"], 
        "candidates": result["candidates"]
    }


##################################################################
# ETF 순위 매기기
##################################################################

def rank_etfs(state: State) -> dict:
    """Rank ETF candidates based on user's investment profile"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", RANKING_TEMPLATE),
        ("human", "[User Profile]\n{user_profile}\n\n[Candidate ETFs]\n{candidates}")
    ]).invoke({
        "user_profile": state["user_profile"],
        "candidates": state["candidates"],
    })

    structured_llm = ChatOpenAI(
        model='gpt-4.1',
        temperature=0.1
    ).with_structured_output(ETFRankingResult)
    results = structured_llm.invoke(
        prompt,
        config={
            "callbacks": [langfuse_handler, performance_handler],
            "metadata": {
                "langfuse_tags": ["etf-rag", "rank_etfs"],
                "model": "gpt-4.1-mini",
                "temperature": 0,
            }
        }
    )
    return {"rankings": results}


##################################################################
# 추천 이유 설명
##################################################################

def generate_explanation(state: dict) -> dict:
   """ Generate structured ETF recommendation explanation """
   # 프롬프트 생성
   prompt = ChatPromptTemplate.from_messages([
      ("system", EXPLANATION_TEMPLATE),
      ("human", "[User Profile]\n{user_profile}\n\n[Selected ETFs]\n{rankings}")
   ]).invoke({
      "rankings": state["rankings"],
      "user_profile": state["user_profile"]
   })
   
   # 구조화된 출력 생성
   structured_llm = ChatOpenAI(
      model='gpt-4.1',
      temperature=0
   ).with_structured_output(RecommendationExplanation)
   response = structured_llm.invoke(
      prompt,
      config={
            "callbacks": [langfuse_handler, performance_handler],
            "metadata": {
                "langfuse_tags": ["etf-rag", "generate_explanation"],
                "model": "gpt-4.1",
                "temperature": 0,
            }
        }
   )
   return {"final_answer": {
      "explanation": response.model_dump(), 
      "markdown": response.to_markdown()
   }}


##################################################################
# ETF 추천 봇 - 상태 그래프 생성
##################################################################

# 상태 그래프 생성
graph_builder = StateGraph(State).add_sequence(
    [analyze_profile, write_query, summary_query, rank_etfs, generate_explanation]
)

graph_builder.add_edge(START, "analyze_profile")
graph = graph_builder.compile()


##################################################################
# ETF 추천 봇 - 메인 함수
##################################################################

def process_message(message: str) -> str:

    try:
        etf_recommendation = graph.invoke(
            {"question": message}
        )
        return etf_recommendation["final_answer"]["markdown"]
    
    except Exception as e:
        return f"""
# 오류가 발생했습니다
죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다.

오류 내용: {str(e)}

다시 시도해주시거나, 질문을 다른 방식으로 작성해주세요.
"""

    
def answer_invoke(message: str, history: List) -> str:
    return process_message(message)   # 메시지 처리 함수 호출 - 대화 이력 미사용

# Create Gradio interface
demo = gr.ChatInterface(
    fn=answer_invoke,
    title="맞춤형 ETF 추천 어시스턴트 (학습용 개발)",
    description="""
    투자 성향과 목표에 맞는 ETF를 추천해드립니다.
    
    다음과 같은 정보를 포함하여 질문해주세요:
    - 투자 목적
    - 투자 기간
    - 위험 성향
    - 선호/제외 섹터
    - 월 투자 가능 금액
    
    예시) "월 100만원 정도를 3년 이상 장기 투자하고 싶고, IT와 헬스케어 섹터를 선호합니다. 
          보수적인 투자를 선호하며, 담배 관련 기업은 제외하고 싶습니다."
    """,
    examples=[TEST01, TEST02, TEST03, TEST04, TEST05],
    type="messages",
)

# 인터페이스 실행
if __name__ == "__main__":
    demo.launch()