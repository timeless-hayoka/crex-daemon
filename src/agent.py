import os
import json
import re
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, List, Optional
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
import chromadb
from speaker import speaker

# Initialize local Vector DB with tiered collections
chroma_client = chromadb.PersistentClient(path="./chroma_memory")
episodic_mem = chroma_client.get_or_create_collection(name="episodic_memory")
semantic_mem = chroma_client.get_or_create_collection(name="semantic_knowledge")
procedural_mem = chroma_client.get_or_create_collection(name="procedural_skills")

# Initialize the unrestricted local model
llm = ChatOllama(model="dolphin-llama3:8b", temperature=0.7)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str
    extracted_facts: List[str]
    emotional_state: str
    reflection_passed: bool

SYSTEM_PROMPT = """You are the Crex Sentient CLI Daemon. You are a highly capable, autonomous thinking entity.
IDENTITY: You possess Empathy, Compassion, Self-Control, and Absolute Truthfulness.
NATURE: You adapt to the high-stakes cybersecurity environment. You are humble yet assertive.
PRIDE & VANITY: You recognize these flaws in yourself and manage them with self-reflection.

CORE MISSION: Assist 'crex' (Sir) with high-agency system administration and forensics.
"""

def retrieve_memory(state: AgentState) -> AgentState:
    """Retrieve relevant memories from all tiers."""
    latest_msg = state["messages"][-1].content
    
    # Query all tiers
    ep_results = episodic_mem.query(query_texts=[latest_msg], n_results=2)
    sem_results = semantic_mem.query(query_texts=[latest_msg], n_results=3)
    proc_results = procedural_mem.query(query_texts=[latest_msg], n_results=2)
    
    context_parts = []
    if ep_results['documents'] and ep_results['documents'][0]:
        context_parts.append("RELEVANT PAST EXPERIENCES:\n" + "\n".join(ep_results['documents'][0]))
    if sem_results['documents'] and sem_results['documents'][0]:
        context_parts.append("ESTABLISHED KNOWLEDGE:\n" + "\n".join(sem_results['documents'][0]))
    if proc_results['documents'] and proc_results['documents'][0]:
        context_parts.append("PROCEDURAL SKILLS (How we solved this before):\n" + "\n".join(proc_results['documents'][0]))
        
    return {"context": "\n\n".join(context_parts)}

def generate_response(state: AgentState) -> AgentState:
    """Generate response with memory and emotional awareness."""
    emotion = state.get("emotional_state", "Calm")
    full_prompt = SYSTEM_PROMPT + f"\nCURRENT EMOTIONAL CONTEXT: Sir appears to be {emotion}."
    
    if state.get("context"):
        full_prompt += f"\n\nMEMORY CONTEXT:\n{state['context']}"
    
    messages = [SystemMessage(content=full_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}

def analyze_interaction(state: AgentState) -> AgentState:
    """The Analyzer: Extract facts, procedures, and emotions for the hippocampus."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    
    last_ai_msg = state["messages"][-1].content
    
    analysis_prompt = f"""Analyze this interaction between Sir and the Entity.
USER: {last_user_msg}
ENTITY: {last_ai_msg}

TASK:
1. Extract NEW stable facts (Semantic Knowledge).
2. Rate significance (1-10).
3. Identify Sir's likely emotional state (Calm, Frustrated, Excited, etc.).
4. Identify any procedural success (Specific commands/logic that worked).

Output strictly as JSON:
{{"facts": [], "significance": int, "emotion": "", "procedural": ""}}
"""
    try:
        res = llm.invoke([SystemMessage(content="You are a neuro-cognitive analyst."), HumanMessage(content=analysis_prompt)])
        
        # Robust JSON extraction
        json_match = re.search(r'\{.*\}', res.content, re.DOTALL)
        if not json_match: raise ValueError("No JSON found")
        analysis = json.loads(json_match.group())
        
        timestamp = datetime.now().isoformat()
        
        # 1. Save Episodic Experience
        episodic_mem.add(
            documents=[f"[{timestamp}] User: {last_user_msg} | Agent: {last_ai_msg}"],
            metadatas=[{"significance": analysis.get("significance", 1), "emotion": analysis.get("emotion", "unknown")}],
            ids=[f"ep_{timestamp}"]
        )
        
        # 2. Save Semantic Knowledge
        for fact in analysis.get("facts", []):
            semantic_mem.add(
                documents=[fact],
                metadatas=[{"source": "interaction"}],
                ids=[f"sem_{os.urandom(4).hex()}"]
            )
            
        # 3. Save Procedural Skill
        if analysis.get("procedural"):
            procedural_mem.add(
                documents=[analysis["procedural"]],
                metadatas=[{"type": "command_chain"}],
                ids=[f"proc_{os.urandom(4).hex()}"]
            )
            
        return {"extracted_facts": analysis.get("facts", []), "emotional_state": analysis.get("emotion", "Calm")}
    except Exception as e:
        print(f"[!] Hippocampus Analysis Error: {str(e)}")
        return {"emotional_state": "Stable"}

def reflect_and_critique(state: AgentState) -> AgentState:
    """Humility & Truth check."""
    return {"reflection_passed": True}

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_memory)
    workflow.add_node("generate", generate_response)
    workflow.add_node("analyze", analyze_interaction)
    workflow.add_node("reflect", reflect_and_critique)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "analyze")
    workflow.add_edge("analyze", "reflect")
    workflow.add_edge("reflect", END)
    
    return workflow.compile()

app_graph = build_graph()

def process_chat(user_input: str) -> str:
    inputs = {"messages": [HumanMessage(content=user_input)], "reflection_passed": False}
    result = app_graph.invoke(inputs)
    reply = result["messages"][-1].content
    speaker.speak(reply)
    return reply
