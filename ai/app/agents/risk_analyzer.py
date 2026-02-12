import os
import json
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.schemas import RiskClassificationResponse

import logging
logger = logging.getLogger(__name__)

class RiskAnalyzerAgent:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0):
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.parser = JsonOutputParser()

    def analyze_risk(self, text: str) -> RiskClassificationResponse:
        prompt = ChatPromptTemplate.from_template(
            """You are a civic risk intensity analyzer. 
            Analyze the following complaint text and classify its risk intensity as "low", "medium", or "high".
            
            Criteria:
            - "high": Immediate danger to life or property, severe traffic disruption, massive utility failure (e.g., major water main burst, live wire down, bridge collapse, building fire).
            - "medium": Significant inconvenience or potential for damage if left unaddressed soon, health hazards (e.g., large potholes on main roads, non-functional street lights in dangerous areas, overflowing sewage, garbage not cleared for days).
            - "low": Minor issues, cosmetic damage, or slight inconvenience (e.g., small potholes on side streets, graffiti, minor litter, requested tree pruning).
            
            Text: {text}
            
            Output ONLY valid JSON correctly matching the following schema:
            {{
                "intensity": "low" | "medium" | "high",
                "confidence": float,
                "reason": "string (brief explanation for the intensity)"
            }}
            Confidence must be a float between 0 and 1.
            No explanation or extra text outside the JSON."""
        )
        
        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({
                "text": text
            })
            
            return RiskClassificationResponse(
                intensity=result.get("intensity", "medium"),
                confidence=float(result.get("confidence", 0.5)),
                reason=result.get("reason", "Standard classification applied."),
                model_name=self.llm.model_name
            )
        except Exception as e:
            import traceback
            logger.error(f"Risk analysis error: {traceback.format_exc()}")
            return RiskClassificationResponse(
                intensity="medium",
                confidence=0.0,
                reason=f"Error during analysis: {str(e)}",
                model_name=self.llm.model_name
            )

# Singleton instance
risk_analyzer_agent: Optional[RiskAnalyzerAgent] = None

def get_risk_analyzer_agent() -> RiskAnalyzerAgent:
    global risk_analyzer_agent
    if risk_analyzer_agent is None:
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        risk_analyzer_agent = RiskAnalyzerAgent(model_name=model_name)
    return risk_analyzer_agent
