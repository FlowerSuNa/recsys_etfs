import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class PerformanceMonitoringCallback(BaseCallbackHandler):
    """LLM 호출 성능을 모니터링하는 콜백 핸들러"""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.token_usage: Dict[str, Any] = {}
        self.call_count: int = 0
        
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """LLM 호출이 시작될 때 호출"""
        self.start_time = time.time()
        self.call_count += 1
        print(f"🚀 LLM 호출 #{self.call_count} 시작 - {datetime.now().strftime('%H:%M:%S')}")
        
        # 첫 번째 프롬프트의 길이 확인
        if prompts:
            print(f"📝 프롬프트 길이: {len(prompts[0])} 문자")
        
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 호출이 완료될 때 호출"""
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"✅ LLM 호출 완료 - 소요시간: {duration:.2f}초")
            
            # 토큰 사용량 추적
            if response.generations:
                generation = response.generations[0][0]
                
                # usage_metadata를 우선 확인 
                if hasattr(generation, 'usage_metadata') and generation.usage_metadata:
                    usage = generation.usage_metadata
                    print(f"🔢 토큰 사용량: {usage}")
                    self.token_usage = usage
                    
                # 구버전 호환성을 위한 llm_output 확인
                elif hasattr(response, 'llm_output') and response.llm_output:
                    usage = response.llm_output.get('token_usage', {})
                    if usage:
                        print(f"🔢 토큰 사용량: {usage}")
                        self.token_usage = usage
                        
                # 응답 길이 체크
                if hasattr(generation, 'text'):
                    response_text = generation.text
                    print(f"📊 응답 길이: {len(response_text)} 문자")
        
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """LLM 호출에서 오류가 발생할 때 호출"""
        print(f"❌ LLM 호출 오류: {str(error)}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """현재까지의 통계 정보를 반환"""
        return {
            "total_calls": self.call_count,
            "last_token_usage": self.token_usage
        }
    
