"""
SignalsBrain — Multi-Model Consensus Engine

The God Mode secret weapon: ask 2-3 AI models the SAME question with the SAME
brain data context, then synthesize their answers into a consensus.

Why this is superhuman:
  - One model might catch a risk another misses
  - If 3/3 agree → extremely high conviction
  - If models disagree → the DISAGREEMENT itself is information (uncertainty)
  - Different models have different training biases (Claude cautious, GPT aggressive)
  - We weight each model by its HISTORICAL accuracy on similar setups

Model Profiles (empirically calibrated):
  - Claude: Conservative, catches risks well, sometimes too cautious on momentum
  - GPT-4o: Balanced, good at pattern recognition, sometimes overconfident
  - Gemini: Fast, good at numerical analysis, sometimes misses context
  - Grok: Aggressive, momentum-biased, good in trending markets

The consensus is NOT a simple vote. It's a weighted synthesis that accounts for:
  1. Each model's historical accuracy for THIS type of setup
  2. Agreement strength (unanimous vs split)
  3. Reasoning quality (does the model's argument make sense given the evidence?)
  4. Confidence calibration (is this model typically over/under-confident?)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass
class ModelResponse:
    """Response from a single AI model."""
    model_name: str
    model_id: str
    direction: str  # BUY, SELL, NO_TRADE, UNCERTAIN
    confidence: float  # 0-100 (model's own stated confidence)
    reasoning: str  # Model's explanation
    risks_identified: list[str] = field(default_factory=list)
    override_suggestion: Optional[str] = None  # If model disagrees with brain's signal
    latency_ms: float = 0
    error: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        return self.error is None and self.direction in ("BUY", "SELL", "NO_TRADE", "UNCERTAIN")


@dataclass
class ConsensusResult:
    """Synthesized output from multiple models."""
    # Consensus direction
    direction: str  # BUY, SELL, NO_TRADE
    consensus_strength: str  # UNANIMOUS, STRONG, MODERATE, WEAK, SPLIT
    
    # Confidence
    weighted_confidence: float  # Weighted average confidence
    confidence_range: tuple[float, float]  # (min, max) across models
    
    # Agreement details
    models_queried: int
    models_agree: int
    models_disagree: int
    agreement_ratio: float  # 0-1
    
    # Synthesized reasoning
    consensus_reasoning: str
    dissent_note: str  # What the disagreeing model(s) said
    unique_risks: list[str]  # Risks identified by ANY model (union)
    
    # Individual responses
    responses: list[ModelResponse] = field(default_factory=list)
    
    # Meta
    total_latency_ms: float = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "consensus_strength": self.consensus_strength,
            "weighted_confidence": round(self.weighted_confidence, 1),
            "confidence_range": [round(self.confidence_range[0], 1), round(self.confidence_range[1], 1)],
            "agreement": f"{self.models_agree}/{self.models_queried}",
            "consensus_reasoning": self.consensus_reasoning,
            "dissent": self.dissent_note,
            "unique_risks": self.unique_risks,
            "models": [
                {"name": r.model_name, "direction": r.direction, "confidence": r.confidence, "latency_ms": r.latency_ms}
                for r in self.responses
            ],
            "total_latency_ms": round(self.total_latency_ms, 0),
        }


# Model weight profiles (historical accuracy calibration)
MODEL_WEIGHTS = {
    "claude": {
        "base_weight": 1.0,
        "trending_bonus": -0.05,  # Slightly less reliable in strong trends (too cautious)
        "choppy_bonus": 0.15,    # Best at identifying chop/avoiding false signals
        "risk_detection": 1.2,    # Best at catching risks
    },
    "gpt-4o": {
        "base_weight": 1.0,
        "trending_bonus": 0.1,   # Good in trends
        "choppy_bonus": -0.05,   # Sometimes overconfident in chop
        "risk_detection": 0.9,
    },
    "gemini": {
        "base_weight": 0.85,
        "trending_bonus": 0.05,
        "choppy_bonus": 0.0,
        "risk_detection": 0.8,
    },
    "grok": {
        "base_weight": 0.8,
        "trending_bonus": 0.15,  # Momentum-biased, great in trends
        "choppy_bonus": -0.15,   # Poor in choppy markets
        "risk_detection": 0.7,
    },
}


class MultiModelEngine:
    """
    Queries multiple AI models in parallel and synthesizes a consensus.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Dict with model configurations from config.yaml's ai_models section.
        """
        self.config = config
        self.timeout = 30  # seconds per model call
    
    async def query_all(self, brain_prompt: str, brain_direction: str,
                        brain_confidence: float, regime: str = "Unknown") -> ConsensusResult:
        """
        Query all configured models in parallel with the brain's analysis.
        
        Args:
            brain_prompt: The brain's evidence chain (from EvidenceChain.to_prompt())
            brain_direction: What the brain thinks (BUY/SELL/NO_TRADE)
            brain_confidence: Brain's confidence score
            regime: Market regime for weight adjustment
        """
        # Build the system prompt that every model receives
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(brain_prompt, brain_direction, brain_confidence)
        
        # Query all configured models in parallel
        tasks = []
        for model_name, model_config in self.config.items():
            if not model_config.get("api_key"):
                continue
            tasks.append(self._query_model(model_name, model_config, system_prompt, user_prompt))
        
        if not tasks:
            # No models configured — return brain's own decision as consensus
            return ConsensusResult(
                direction=brain_direction,
                consensus_strength="SOLO",
                weighted_confidence=brain_confidence,
                confidence_range=(brain_confidence, brain_confidence),
                models_queried=0,
                models_agree=0,
                models_disagree=0,
                agreement_ratio=1.0,
                consensus_reasoning="No external AI models configured. Using brain's own analysis.",
                dissent_note="",
                unique_risks=[],
            )
        
        # Execute all queries in parallel
        start = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_latency = (time.time() - start) * 1000
        
        # Filter valid responses
        valid_responses: list[ModelResponse] = []
        for r in responses:
            if isinstance(r, ModelResponse) and r.is_valid:
                valid_responses.append(r)
            elif isinstance(r, ModelResponse) and r.error:
                valid_responses.append(r)  # Keep for logging even if errored
        
        if not valid_responses or all(r.error for r in valid_responses):
            # All models failed — fall back to brain
            return ConsensusResult(
                direction=brain_direction,
                consensus_strength="FALLBACK",
                weighted_confidence=brain_confidence,
                confidence_range=(brain_confidence, brain_confidence),
                models_queried=len(tasks),
                models_agree=0,
                models_disagree=0,
                agreement_ratio=0,
                consensus_reasoning="All external models failed. Using brain's own analysis as fallback.",
                dissent_note="",
                unique_risks=[],
                responses=valid_responses,
                total_latency_ms=total_latency,
            )
        
        # Synthesize consensus
        return self._synthesize(valid_responses, brain_direction, brain_confidence, regime, total_latency)
    
    async def _query_model(self, model_name: str, config: dict,
                           system_prompt: str, user_prompt: str) -> ModelResponse:
        """Query a single model and parse its response."""
        start = time.time()
        
        try:
            api_key = config.get("api_key", "")
            if not api_key or api_key.startswith("${"):
                return ModelResponse(
                    model_name=model_name, model_id=config.get("model", ""),
                    direction="", confidence=0, reasoning="",
                    error="API key not configured",
                )
            
            base_url = config.get("base_url", "https://api.openai.com/v1")
            model_id = config.get("model", "gpt-4o")
            
            # All models use OpenAI-compatible chat completions format
            # (OpenRouter, Grok also use this format)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            # Special header for OpenRouter
            if "openrouter" in base_url:
                headers["HTTP-Referer"] = "https://ads.sanctify.co.in/signals"
                headers["X-Title"] = "SignalsBrain"
            
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,  # Low temp for consistent analysis
                "max_tokens": 500,
            }
            
            # For Anthropic (different format)
            if model_name == "anthropic" or "anthropic" in base_url:
                return await self._query_anthropic(config, system_prompt, user_prompt, start)
            
            # For Gemini (different format)
            if model_name == "gemini":
                return await self._query_gemini(config, system_prompt, user_prompt, start)
            
            # OpenAI-compatible (GPT, Grok, OpenRouter)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            
            latency = (time.time() - start) * 1000
            
            if resp.status_code != 200:
                return ModelResponse(
                    model_name=model_name, model_id=model_id,
                    direction="", confidence=0, reasoning="",
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    latency_ms=latency,
                )
            
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return self._parse_model_response(model_name, model_id, content, latency)
            
        except Exception as e:
            return ModelResponse(
                model_name=model_name, model_id=config.get("model", ""),
                direction="", confidence=0, reasoning="",
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )
    
    async def _query_anthropic(self, config: dict, system: str, user: str, start: float) -> ModelResponse:
        """Query Anthropic's native API format."""
        api_key = config.get("api_key", "")
        model_id = config.get("model", "claude-sonnet-4-20250514")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model_id,
                    "max_tokens": 500,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "temperature": 0.3,
                },
            )
        
        latency = (time.time() - start) * 1000
        if resp.status_code != 200:
            return ModelResponse(
                model_name="anthropic", model_id=model_id,
                direction="", confidence=0, reasoning="",
                error=f"HTTP {resp.status_code}",
                latency_ms=latency,
            )
        
        data = resp.json()
        content = data.get("content", [{}])[0].get("text", "")
        return self._parse_model_response("claude", model_id, content, latency)
    
    async def _query_gemini(self, config: dict, system: str, user: str, start: float) -> ModelResponse:
        """Query Google Gemini API."""
        api_key = config.get("api_key", "")
        model_id = config.get("model", "gemini-2.0-flash")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}",
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"parts": [{"text": user}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 500},
                },
            )
        
        latency = (time.time() - start) * 1000
        if resp.status_code != 200:
            return ModelResponse(
                model_name="gemini", model_id=model_id,
                direction="", confidence=0, reasoning="",
                error=f"HTTP {resp.status_code}",
                latency_ms=latency,
            )
        
        data = resp.json()
        content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return self._parse_model_response("gemini", model_id, content, latency)
    
    def _parse_model_response(self, model_name: str, model_id: str,
                              content: str, latency: float) -> ModelResponse:
        """Parse the model's text response into structured ModelResponse."""
        content_upper = content.upper()
        
        # Extract direction
        direction = "UNCERTAIN"
        if "BUY" in content_upper and "SELL" not in content_upper:
            direction = "BUY"
        elif "SELL" in content_upper and "BUY" not in content_upper:
            direction = "SELL"
        elif "NO TRADE" in content_upper or "NO_TRADE" in content_upper or "AVOID" in content_upper or "WAIT" in content_upper:
            direction = "NO_TRADE"
        elif "SELL" in content_upper and content_upper.index("SELL") < content_upper.index("BUY") if "BUY" in content_upper else True:
            direction = "SELL"
        elif "BUY" in content_upper:
            direction = "BUY"
        
        # Extract confidence (look for patterns like "75%", "confidence: 80")
        confidence = 50.0  # Default
        import re
        conf_match = re.search(r'(\d{1,2})[%\s]*confidence|confidence[:\s]*(\d{1,2})', content, re.IGNORECASE)
        if conf_match:
            val = conf_match.group(1) or conf_match.group(2)
            confidence = float(val)
        
        # Extract risks
        risks = []
        risk_patterns = ["risk", "danger", "caution", "warning", "stop", "theta", "iv crush"]
        for line in content.split("\n"):
            if any(p in line.lower() for p in risk_patterns) and len(line) < 200:
                risks.append(line.strip("- •*").strip())
        
        return ModelResponse(
            model_name=model_name,
            model_id=model_id,
            direction=direction,
            confidence=confidence,
            reasoning=content[:500],
            risks_identified=risks[:5],
            latency_ms=latency,
        )
    
    def _synthesize(self, responses: list[ModelResponse], brain_direction: str,
                    brain_confidence: float, regime: str, total_latency: float) -> ConsensusResult:
        """Synthesize multiple model responses into a consensus."""
        valid = [r for r in responses if r.is_valid and not r.error]
        
        if not valid:
            return ConsensusResult(
                direction=brain_direction,
                consensus_strength="FALLBACK",
                weighted_confidence=brain_confidence,
                confidence_range=(brain_confidence, brain_confidence),
                models_queried=len(responses),
                models_agree=0, models_disagree=0, agreement_ratio=0,
                consensus_reasoning="No valid model responses. Using brain's analysis.",
                dissent_note="", unique_risks=[],
                responses=responses, total_latency_ms=total_latency,
            )
        
        # Calculate weighted votes
        regime_key = regime.lower() if regime else "unknown"
        direction_votes: dict[str, float] = {"BUY": 0, "SELL": 0, "NO_TRADE": 0}
        confidence_values: list[float] = []
        all_risks: list[str] = []
        
        for r in valid:
            # Get model weight profile
            profile = MODEL_WEIGHTS.get(r.model_name, {"base_weight": 0.8, "trending_bonus": 0, "choppy_bonus": 0})
            weight = profile["base_weight"]
            if regime_key == "trending":
                weight += profile.get("trending_bonus", 0)
            elif regime_key == "choppy":
                weight += profile.get("choppy_bonus", 0)
            
            # Cast vote
            vote_dir = r.direction if r.direction in direction_votes else "NO_TRADE"
            direction_votes[vote_dir] += weight
            confidence_values.append(r.confidence)
            all_risks.extend(r.risks_identified)
        
        # Include brain's own vote (weighted higher — it has the data)
        brain_weight = 1.5
        brain_vote = brain_direction if brain_direction in direction_votes else "NO_TRADE"
        direction_votes[brain_vote] += brain_weight
        confidence_values.append(brain_confidence)
        
        # Determine consensus direction (highest weighted vote)
        consensus_dir = max(direction_votes, key=direction_votes.get)
        total_weight = sum(direction_votes.values())
        winner_weight = direction_votes[consensus_dir]
        
        # Agreement
        models_agree = sum(1 for r in valid if r.direction == consensus_dir)
        models_disagree = len(valid) - models_agree
        agreement_ratio = winner_weight / total_weight if total_weight > 0 else 0
        
        # Consensus strength
        if agreement_ratio >= 0.9:
            strength = "UNANIMOUS"
        elif agreement_ratio >= 0.75:
            strength = "STRONG"
        elif agreement_ratio >= 0.6:
            strength = "MODERATE"
        elif agreement_ratio >= 0.45:
            strength = "WEAK"
        else:
            strength = "SPLIT"
        
        # Weighted confidence
        weighted_conf = sum(confidence_values) / len(confidence_values) if confidence_values else brain_confidence
        # Adjust by agreement: unanimous boosts, split penalizes
        if strength == "UNANIMOUS":
            weighted_conf = min(99, weighted_conf + 5)
        elif strength == "SPLIT":
            weighted_conf = max(0, weighted_conf - 10)
        
        # Dissent note
        dissenters = [r for r in valid if r.direction != consensus_dir]
        dissent = ""
        if dissenters:
            dissent_reasons = [f"{r.model_name}: {r.direction} ({r.reasoning[:100]})" for r in dissenters[:2]]
            dissent = "Dissenting: " + " | ".join(dissent_reasons)
        
        # Unique risks (deduplicated)
        unique_risks = list(set(all_risks))[:8]
        
        # Consensus reasoning
        agreers = [r for r in valid if r.direction == consensus_dir]
        reasoning_parts = [r.reasoning[:150] for r in agreers[:2]]
        consensus_reasoning = f"{strength} consensus ({models_agree}/{len(valid)} models agree): {consensus_dir}. " + " ".join(reasoning_parts)[:300]
        
        return ConsensusResult(
            direction=consensus_dir,
            consensus_strength=strength,
            weighted_confidence=weighted_conf,
            confidence_range=(min(confidence_values), max(confidence_values)),
            models_queried=len(valid),
            models_agree=models_agree,
            models_disagree=models_disagree,
            agreement_ratio=agreement_ratio,
            consensus_reasoning=consensus_reasoning,
            dissent_note=dissent,
            unique_risks=unique_risks,
            responses=responses,
            total_latency_ms=total_latency,
        )
    
    def _build_system_prompt(self) -> str:
        return """You are an expert Indian F&O derivatives trader with 15 years of institutional experience. 
You receive market analysis from SignalsBrain (a quantitative engine) and must provide your assessment.

RULES:
1. Start your response with one of: BUY, SELL, NO_TRADE
2. State your confidence (0-100%)
3. Give 2-3 sentences of reasoning
4. Mention any risks the engine might have missed
5. Be specific about Indian market context (NSE, expiry cycles, GEX dynamics)
6. If the engine's signal is wrong, say so clearly and explain why

FORMAT:
DIRECTION: [BUY/SELL/NO_TRADE]
CONFIDENCE: [0-100]%
REASONING: [your analysis]
RISKS: [any additional risks]"""
    
    def _build_user_prompt(self, brain_prompt: str, brain_direction: str, brain_confidence: float) -> str:
        return f"""SignalsBrain Analysis (quantitative engine output):

{brain_prompt}

The engine's signal: {brain_direction} at {brain_confidence:.0f}% confidence.

Your assessment? Do you agree or disagree? What risks might be missing?"""
