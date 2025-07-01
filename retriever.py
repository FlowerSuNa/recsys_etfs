import ast
import re
import pickle

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from kiwipiepy import Kiwi

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents import initialize_agent, AgentType


def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

def get_retriever(embeddings, data, collection_name, data_add=False):
    """ 검색기 반환 함수 """
    chroma_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    kiwi = Kiwi()
    
    if data_add:
        # 데이터를 벡터스토어에 저장
        chroma_db.add_texts(data)       

    retriever = chroma_db.as_retriever(
        search_kwargs={"k": 10}
    )
    bm25_retriever = BM25Retriever.from_texts(
        texts=data,
        preprocess_func=lambda x: [t.form for t in kiwi.tokenize(x)]
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever, bm25_retriever], 
        weights=[0.5, 0.5]          
    )
    return ensemble_retriever

def create_toolkits(retriever, description):
    """ 검색 도구 생성 """
    entity_retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description=description,
    )
    return entity_retriever_tool

def get_retriever_tools(db):
    """ 검색기 도구 반환 함수 """
    # 고유명사 작업 리스트
    etfs_ko = query_as_list(db, "SELECT DISTINCT 종목명 FROM ETFS_WITH_INFO")
    etfs_en = query_as_list(db, "SELECT DISTINCT 영문명 FROM ETFS_WITH_INFO")
    fund_managers = query_as_list(db, "SELECT DISTINCT 운용사 FROM ETFS_WITH_INFO")
    underlying_assets = query_as_list(db, "SELECT DISTINCT 분류체계 FROM ETFS_WITH_INFO")

    participant = []
    data = query_as_list(db, "SELECT DISTINCT 지정참가회사 FROM ETFS_WITH_INFO")
    for row in data:
        cleaned = re.split(r',\s*', row)
        cleaned = [r.strip() for r in cleaned if r.strip() != '']
        participant += cleaned

    participant = list(set(participant))

    etf_summary = query_as_list(db, "SELECT 기본정보 FROM ETFS_WITH_INFO")
    investment_warning = query_as_list(db, "SELECT 투자유의사항 FROM ETFS_WITH_INFO")

    # 임베딩 모델 정의
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # 검색기 로드
    etfs_retriever = get_retriever(embeddings, etfs_ko + etfs_en, "etfs")
    fund_managers_retriever = get_retriever(embeddings, fund_managers, "fund_managers")
    underlying_assets_retriever = get_retriever(embeddings, underlying_assets, "underlying_assets")
    participant_retriever = get_retriever(embeddings, participant, "participant")
    etf_summary_retriever = get_retriever(embeddings, etf_summary, "etf_summary")
    investment_warning_retriever = get_retriever(embeddings, investment_warning, "investment_warning")

    # 검색기 툴 생성
    description = (
        "Use this tool to look up official ETF names in Korean or English. "
        "Input is an approximate or partial name, and the output includes valid ETF identifiers. "
        "Use when the user asks about a specific ETF, ticker, or wants to explore ETF options."
    )
    etfs_tool = create_toolkits(etfs_retriever, description)

    description = (
        "Use this tool to search for fund managers or asset management companies. "
        "Input can be a partial or misspelled company name. "
        "Use when the question refers to who manages the ETF or which company is responsible for it."
    )
    fund_managers_tool = create_toolkits(fund_managers_retriever, description)

    description = (
        "Use this tool to look up underlying assets or classification types of ETFs. "
        "Helpful when the user asks about what asset class or index an ETF tracks. "
        "Input can include asset types, sectors, or index names."
    )
    underlying_assets_tool = create_toolkits(underlying_assets_retriever, description)

    description = (
        "Use this tool to identify authorized participants (지정참가회사) for a given ETF. "
        "Input should be an approximate name or related company. "
        "Useful when the user inquires about liquidity providers or ETF creation/redemption participants."
    )
    participant_tool = create_toolkits(participant_retriever, description)

    description = (
        "Use this tool to retrieve general summary information about an ETF, "
        "such as its investment strategy, benchmark, or unique characteristics. "
        "Best used when the user asks for an overview or introduction to an ETF."
    )
    etf_summary_tool = create_toolkits(etf_summary_retriever, description)

    description = (
        "Use this tool to retrieve investment warnings or cautionary notes related to a specific ETF. "
        "Useful when the user is asking about potential risks, legal disclaimers, or what to be careful of before investing."
    )
    investment_warning_tool = create_toolkits(investment_warning_retriever, description)

    # 여러 검색기 통합
    retriever_tools = [
        etfs_tool, fund_managers_tool,
        underlying_assets_tool, participant_tool,
        etf_summary_tool, investment_warning_tool
    ]

    # 여러 검색기를 질의에 맞게 선택할 수 있는 에이전트 정의
    agent_executor = initialize_agent(
        tools=retriever_tools,
        agent=AgentType.OPENAI_FUNCTIONS,
        llm=ChatOpenAI(model="gpt-4.1"),
    )
    return agent_executor

if __name__ == "__main__":
    from langchain_community.utilities import SQLDatabase
    from dotenv import load_dotenv

    load_dotenv()

    # 디비 연결
    db = SQLDatabase.from_uri(
        "sqlite:///etf_database.db",
        include_tables=["ETFS_WITH_INFO"]
    )

    # 고유명사 작업 리스트
    etfs_ko = query_as_list(db, "SELECT DISTINCT 종목명 FROM ETFS_WITH_INFO")
    etfs_en = query_as_list(db, "SELECT DISTINCT 영문명 FROM ETFS_WITH_INFO")
    fund_managers = query_as_list(db, "SELECT DISTINCT 운용사 FROM ETFS_WITH_INFO")
    underlying_assets = query_as_list(db, "SELECT DISTINCT 분류체계 FROM ETFS_WITH_INFO")

    participant = []
    data = query_as_list(db, "SELECT DISTINCT 지정참가회사 FROM ETFS_WITH_INFO")
    for row in data:
        cleaned = re.split(r',\s*', row)
        cleaned = [r.strip() for r in cleaned if r.strip() != '']
        participant += cleaned

    participant = list(set(participant))

    etf_summary = query_as_list(db, "SELECT 기본정보 FROM ETFS_WITH_INFO")
    investment_warning = query_as_list(db, "SELECT 투자유의사항 FROM ETFS_WITH_INFO")

    # 임베딩 모델 정의
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # 임베딩 벡터 저장소 및 검색기 생성
    get_retriever(embeddings, etfs_ko + etfs_en, "etfs", data_add=True)
    get_retriever(embeddings, fund_managers, "fund_managers", data_add=True)
    get_retriever(embeddings, underlying_assets, "underlying_assets", data_add=True)
    get_retriever(embeddings, participant, "participant", data_add=True)
    get_retriever(embeddings, etf_summary, "etf_summary", data_add=True)
    get_retriever(embeddings, investment_warning, "investment_warning", data_add=True)