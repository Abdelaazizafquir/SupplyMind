"""
LLM Coordinator Agent - LangGraph Implementation
==============================================
LangGraph-based state machine that orchestrates multi-agent decisions.

Structure:
START -> observe -> [Routing] -> advising/override -> explain -> END
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, TypedDict, Literal
from dataclasses import dataclass, field
from enum import Enum

try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    print("Warning: langgraph is not installed. Please run: pip install langgraph")
    # Stub classes for graceful failure if not installed
    class StateGraph:
        def __init__(self, *args, **kwargs): pass
        def add_node(self, *args, **kwargs): pass
        def add_edge(self, *args, **kwargs): pass
        def add_conditional_edges(self, *args, **kwargs): pass
        def compile(self): return self
    START = "start"
    END = "end"


class CoordinatorMode(str, Enum):
    ADVISORY = "advisory"
    OVERRIDE = "override"
    EXPLAIN = "explain"
    NONE = "none"


class GraphState(TypedDict):
    """The State dictionary managed by the LangGraph application."""
    step: int
    env_state: Dict[str, Any]
    disruption_info: Dict[str, Any]
    agent_ids: List[str]
    num_agents_config: Dict[str, int]
    
    proposed_actions: Dict[str, np.ndarray]
    
    # Coordinator Internal States
    mode: CoordinatorMode
    inventory_summary: str
    proposals_summary: str
    
    # Outputs
    adjusted_actions: Dict[str, np.ndarray]
    llm_response: Dict[str, Any]
    explanation: str


class PromptTemplates:
    """Structured prompt templates for the LLM Coordinator."""
    
    SYSTEM_PROMPT = """You are an expert Supply Chain Coordinator AI managing a multi-echelon 
inventory system with {num_agents} agents (retailers, distributors, suppliers).

Your role is to review agent order proposals and provide coordination decisions.
You must balance: minimizing costs, preventing stockouts, and reducing the bullwhip effect.

Current supply chain structure:
- {num_retailers} Retailers (serve customers directly)
- {num_distributors} Distributors (intermediary)
- {num_suppliers} Suppliers (raw material source)

Respond ONLY in the specified JSON format."""
    
    ADVISORY_PROMPT = """## Current State (Step {step})

### Inventory Levels
{inventory_summary}

### Demand Info
See agent observations

### Agent Proposals (MAPPO Actions)
{proposals}

### Disruption Status
{disruption_status}

## Task
Review the agent proposals and suggest adjustments. Consider:
1. Are any agents ordering too much (risk of excess holding cost)?
2. Are any agents ordering too little (risk of stockout)?
3. Is there coordination needed between echelons?
4. Are orders amplifying demand variability (bullwhip effect)?

Respond in JSON:
{{
    "adjustments": {{
        "agent_0": {{"factor": 1.0, "reason": "..."}},
        "agent_1": {{"factor": 1.0, "reason": "..."}}
    }},
    "coordination_note": "Brief overall assessment",
    "risk_level": "low|medium|high"
}}"""

    OVERRIDE_PROMPT = """## 🚨 DISRUPTION ALERT (Step {step})

### Disruption Details
Type: {disruption_type}
Severity: {severity}
Affected Nodes: {affected_nodes}

### Current Inventory
{inventory_summary}

### Agent Proposals (may be suboptimal under disruption)
{proposals}

## Task
The standard MAPPO policy was trained under normal conditions and may not handle 
this disruption optimally. Override the agent actions if needed.

Respond in JSON:
{{
    "override_actions": {{
        "agent_0": {{"order_quantity_normalized": 0.5, "reason": "..."}}
    }},
    "emergency_strategy": "Brief description of emergency response",
    "expected_recovery": "When the system should stabilize"
}}"""

    EXPLAIN_PROMPT = """## Decision Explanation Request (Step {step})

### Actions Taken
{actions_taken}

### Results / Predictions
Decision Mode used: {mode}

## Task
Explain the decisions made in plain language for a supply chain manager.
Focus on WHY these specific order quantities were chosen based on the recent analysis ({llm_response}).

Respond in JSON:
{{
    "explanation": "2-3 sentence explanation",
    "key_factors": ["factor1", "factor2", "factor3"],
    "recommendation": "What to watch for next"
}}"""


class LLMCoordinatorGraph:
    """
    LLM-powered coordinator orchestrated by LangGraph.
    """
    
    def __init__(self, config: dict = None):
        cfg = config or {}
        
        self.provider = cfg.get("llm_provider", "mock")
        self.model = cfg.get("llm_model", "llama3:8b")
        self.temperature = cfg.get("temperature", 0.3)
        self.max_tokens = cfg.get("max_tokens", 512)
        
        self.advisory_enabled = cfg.get("advisory_enabled", True)
        self.override_enabled = cfg.get("override_enabled", True)
        self.override_threshold = cfg.get("override_threshold", 0.8)
        self.explanation_enabled = cfg.get("explanation_enabled", True)
        
        self.explanation_log: List[str] = []
        self.templates = PromptTemplates()
        
        # Initialize LLM client
        self.client = self._init_client()
        
        # Build LangGraph StateMachine
        self.app = self._build_graph()

    def _init_client(self):
        """Initialize the LLM client based on provider."""
        if self.provider == "ollama":
            try:
                import requests
                resp = requests.get("http://localhost:11434/api/tags", timeout=5)
                if resp.status_code == 200:
                    print(f"  ✅ Ollama connected. Model: {self.model}")
                    return "ollama"
            except Exception:
                print("  ⚠️ Ollama not available. Falling back to mock.")
                self.provider = "mock"
                return "mock"
                
        elif self.provider == "openai":
            try:
                import openai
                client = openai.OpenAI()
                print(f"  ✅ OpenAI client initialized. Model: {self.model}")
                return client
            except Exception as e:
                print(f"  ⚠️ OpenAI not available: {e}. Falling back to mock.")
                self.provider = "mock"
                return "mock"

        elif self.provider == "gemini":
            try:
                import os
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not set")
                genai.configure(api_key=api_key)
                # Default to Gemma 3 12B on Google AI Studio if no model set
                if self.model in ("llama3:8b", "gemini-1.5-flash", ""):
                    self.model = "gemma-3-12b-it"
                client = genai.GenerativeModel(
                    model_name=self.model,
                    generation_config=genai.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                print(f"  ✅ Google AI Studio client ready. Model: {self.model}")
                return client
            except Exception as e:
                print(f"  ⚠️ Gemini/Gemma not available: {e}. Falling back to mock.")
                self.provider = "mock"
                return "mock"

        elif self.provider == "groq":
            try:
                import os
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not set")
                if self.model in ("llama3:8b", ""):
                    self.model = "gemma2-9b-it"
                client = Groq(api_key=api_key)
                print(f"  ✅ Groq client ready. Model: {self.model}")
                return client
            except Exception as e:
                print(f"  ⚠️ Groq not available: {e}. Falling back to mock.")
                self.provider = "mock"
                return "mock"

        return "mock"

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the API and return generation text."""
        if self.provider == "mock":
            return self._mock_response(user_prompt)

        elif self.provider == "gemini":
            # Google AI Studio: combine system + user (Gemma models don't support system role)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.client.generate_content(full_prompt)
            return response.text

        elif self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        
        elif self.provider == "ollama":
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": False,
                    "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
                },
                timeout=60,
            )
            return response.json().get("response", "")
            
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

    def _mock_response(self, prompt: str) -> str:
        if "DISRUPTION ALERT" in prompt:
            return json.dumps({
                "override_actions": {
                    "agent_0": {"order_quantity_normalized": 0.7, "reason": "Increase safety stock"},
                },
                "emergency_strategy": "Increase orders at non-affected nodes to compensate",
                "expected_recovery": "System should stabilize within 5-10 periods",
            })
        elif "Explanation Request" in prompt:
            return json.dumps({
                "explanation": "Orders were balanced to minimize total cost while maintaining service levels.",
                "key_factors": ["demand_forecast", "current_inventory"],
                "recommendation": "Monitor retailer 0 for potential stockout",
            })
        else:
            return json.dumps({
                "adjustments": {},
                "coordination_note": "Current proposals are well-balanced.",
                "risk_level": "low",
            })

    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON robustly from LLM string output."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            return {"raw_response": text}

    # ==========================================
    # LANGGRAPH NODES
    # ==========================================
    def node_observe(self, state: GraphState) -> GraphState:
        """Ingest the state and properly format observation strings."""
        disruption_info = state["disruption_info"]
        
        # Determine mode
        num_disruptions = disruption_info.get("disruptions_active", 0)
        severity = disruption_info.get("disruption_severity", 0.0)
        
        if self.override_enabled and num_disruptions > 0 and severity >= self.override_threshold:
            mode = CoordinatorMode.OVERRIDE
        elif self.advisory_enabled:
            mode = CoordinatorMode.ADVISORY
        elif self.explanation_enabled:
            mode = CoordinatorMode.EXPLAIN
        else:
            mode = CoordinatorMode.NONE
            
        # Format Inventory Summary
        lines = []
        for i, node in enumerate(state["env_state"].get("nodes", [])):
            pipeline = sum(q for _, q in node.get("pipeline", []))
            lines.append(f"- Agent {i} ({node.get('type', 'unknown')}): Inventory={node.get('inventory', 0):.0f}, Pipeline={pipeline:.0f}")
        inv_sum = "\n".join(lines) if lines else "No inventory data"
        
        # Format Proposals Summary
        lines = []
        for agent_id, action in state["proposed_actions"].items():
            order = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            lines.append(f"- {agent_id}: order_normalized={order:.4f}")
        prop_sum = "\n".join(lines)
        
        return {
            "mode": mode,
            "inventory_summary": inv_sum,
            "proposals_summary": prop_sum,
            "adjusted_actions": dict(state["proposed_actions"]) # default passthrough
        }

    def node_advisory(self, state: GraphState) -> GraphState:
        """Call LLM to soft-adjust proposals based on advisory conditions."""
        sys_prompt = self.templates.SYSTEM_PROMPT.format(
            num_agents=len(state["agent_ids"]), **state["num_agents_config"]
        )
        
        user_prompt = self.templates.ADVISORY_PROMPT.format(
            step=state["step"],
            inventory_summary=state["inventory_summary"],
            proposals=state["proposals_summary"],
            disruption_status="Active" if state["disruption_info"].get("disruptions_active") else "None"
        )
        
        response_text = self._call_llm(sys_prompt, user_prompt)
        response_json = self._parse_json_response(response_text)
        
        adjusted = dict(state["proposed_actions"])
        if "adjustments" in response_json:
            for agent_id, adj in response_json["adjustments"].items():
                if agent_id in adjusted:
                    factor = adj.get("factor", 1.0)
                    adjusted[agent_id] = np.clip(adjusted[agent_id] * factor, 0.0, 1.0)
                    
        return {"adjusted_actions": adjusted, "llm_response": response_json}

    def node_override(self, state: GraphState) -> GraphState:
        """Call LLM to hard-override proposed actions during disruptions."""
        sys_prompt = self.templates.SYSTEM_PROMPT.format(
            num_agents=len(state["agent_ids"]), **state["num_agents_config"]
        )
        
        user_prompt = self.templates.OVERRIDE_PROMPT.format(
            step=state["step"],
            disruption_type=state["disruption_info"].get("disruption_types", ["unknown"]),
            severity=state["disruption_info"].get("disruption_severity", 0),
            affected_nodes=state["disruption_info"].get("failed_nodes", []),
            inventory_summary=state["inventory_summary"],
            proposals=state["proposals_summary"]
        )
        
        response_text = self._call_llm(sys_prompt, user_prompt)
        response_json = self._parse_json_response(response_text)
        
        adjusted = dict(state["proposed_actions"])
        if "override_actions" in response_json:
            for agent_id, override in response_json["override_actions"].items():
                if agent_id in adjusted:
                    new_qty = override.get("order_quantity_normalized", 0.5)
                    adjusted[agent_id] = np.array([new_qty], dtype=np.float32)
                    
        return {"adjusted_actions": adjusted, "llm_response": response_json}

    def node_explain(self, state: GraphState) -> GraphState:
        """Generate a natural language explanation for the final action chosen."""
        if not self.explanation_enabled:
            return {"explanation": "Explanation disabled."}
            
        sys_prompt = self.templates.SYSTEM_PROMPT.format(
            num_agents=len(state["agent_ids"]), **state["num_agents_config"]
        )
        
        actions_taken_str = "\n".join([f"- {a}: {v[0]:.4f}" for a, v in state["adjusted_actions"].items()])
        
        user_prompt = self.templates.EXPLAIN_PROMPT.format(
            step=state["step"],
            actions_taken=actions_taken_str,
            mode=state["mode"].value,
            llm_response=json.dumps(state.get("llm_response", {}))
        )
        
        response_text = self._call_llm(sys_prompt, user_prompt)
        response_json = self._parse_json_response(response_text)
        expl = response_json.get("explanation", response_text)
        
        self.explanation_log.append(f"[Step {state['step']}] {expl}")
        return {"explanation": expl}

    # ==========================================
    # GRAPH COMPILATION
    # ==========================================
    def _build_graph(self):
        """Constructs the cyclical LangGraph and compiles it."""
        workflow = StateGraph(GraphState)
        
        workflow.add_node("observe", self.node_observe)
        workflow.add_node("advising", self.node_advisory)
        workflow.add_node("override", self.node_override)
        workflow.add_node("explain", self.node_explain)
        
        workflow.add_edge(START, "observe")
        
        def route_after_observe(state: GraphState) -> str:
            if state["mode"] == CoordinatorMode.OVERRIDE:
                return "override"
            elif state["mode"] == CoordinatorMode.ADVISORY:
                return "advising"
            elif state["mode"] == CoordinatorMode.EXPLAIN:
                return "explain"
            return END
            
        workflow.add_conditional_edges(
            "observe", 
            route_after_observe,
            {"override": "override", "advising": "advising", "explain": "explain", END: END}
        )
        
        workflow.add_edge("advising", "explain")
        workflow.add_edge("override", "explain")
        workflow.add_edge("explain", END)
        
        return workflow.compile()

    # ==========================================
    # ENTRY POINT
    # ==========================================
    def coordinate(
        self,
        step: int,
        observations: Dict[str, np.ndarray],
        proposed_actions: Dict[str, np.ndarray],
        env_state: Dict,
        disruption_info: Dict = None,
        agent_ids: List[str] = None,
        num_agents_config: Dict = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Execute the LangGraph coordination graph.
        
        Returns:
            (adjusted_actions, coordinator_info_dict)
        """
        state_input = {
            "step": step,
            "env_state": env_state,
            "disruption_info": disruption_info or {},
            "proposed_actions": proposed_actions,
            "agent_ids": agent_ids or list(proposed_actions.keys()),
            "num_agents_config": num_agents_config or {"num_retailers": 3, "num_distributors": 2, "num_suppliers": 1},
            "llm_response": {},
            "adjusted_actions": proposed_actions,
        }
        
        # Run LangGraph compilation (synchronous)
        try:
            final_state = self.app.invoke(state_input)
        except Exception as e:
            # Fallback to passthrough if graph invocation fails (e.g. missing package)
            print(f"[Coordinator] Warning - Graph invocation failed: {e}")
            final_state = state_input
            final_state["mode"] = CoordinatorMode.NONE
            
        coord_info = {
            "mode": final_state.get("mode", CoordinatorMode.NONE).value if isinstance(final_state.get("mode"), CoordinatorMode) else final_state.get("mode", "none"),
            "response": final_state.get("llm_response", {}),
            "actions_modified": final_state.get("mode") in [CoordinatorMode.OVERRIDE, CoordinatorMode.ADVISORY],
            "explanation": final_state.get("explanation", "")
        }
            
        return final_state.get("adjusted_actions", proposed_actions), coord_info

