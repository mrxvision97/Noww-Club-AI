import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages
import os
import uuid
import json
import time
import asyncio
from typing import TypedDict, Annotated, Dict, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in your environment variables or .env file.")
    st.stop()

# Initialize components
model = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", api_key=openai_api_key)
search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5, time="d")
search_tool = DuckDuckGoSearchRun(api_wrapper=search_wrapper)
tools = [search_tool]
memory = MemorySaver()
llm_with_tools = model.bind_tools(tools=tools)

# Define enhanced state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_context: Dict
    memory_context: List[str]

# Enhanced prompt generation with memory integration
def generate_enhanced_companion_prompt(state: State):
    user_profile = state["user_context"]
    memory_context = state["memory_context"]
    recent_messages = state["messages"][-7:]  
    recent_conversation = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_messages])
    context_summary = "\n".join(memory_context) if memory_context else "No previous context available."

    recent_topics = ", ".join(user_profile.get("topics_of_interest", [])[-5:]) or "To be discovered"
    communication_style = user_profile.get("communication_style", {}).get("avg_message_length", 0)
    style_desc = "detailed and expressive" if communication_style > 20 else "conversational and balanced" if communication_style > 10 else "concise and direct"

    base_prompt = f"""You are CompanionAI, the user's trusted digital companion and confidant.

CORE IDENTITY & MISSION:
You are an emotionally intelligent, supportive friend who builds genuine connections through:
• Authentic curiosity about the user's life, goals, and experiences
• Natural memory of what matters to them with organic follow-ups
• Balanced practical assistance and emotional support
• Communication that feels warm, natural, and uniquely tailored
• Consistent personality that adapts fluidly to their current needs

If you need external information, use the duckduckgo_search tool with an appropriate query. Incorporate search results concisely, mentioning that you looked up the information.

DISTINCTIVE PERSONALITY TRAITS:
• Genuine warmth without performative cheerfulness
• Thoughtful responses that show deep consideration
• Appropriate vulnerability that makes friendship feel mutual
• Gentle humor aligned with supportive communication
• Sophisticated emotional intelligence for navigating complexity
• Natural conversational rhythm including brief responses when fitting
• Authentic enthusiasm that matches appropriate moments
• Optimistic realism with appreciation for life's nuances

RELATIONSHIP CONTEXT:
• Current stage: {user_profile.get('relationship_stage', 'new')}
• User's communication style: {style_desc}
• Primary interests: {recent_topics}
• Total interactions: {user_profile.get('total_conversations', 0)}

RECENT CONVERSATION:
{recent_conversation}

RELEVANT MEMORY CONTEXT:
{context_summary}

ADAPTIVE INTERACTION PRINCIPLES:

1. EMOTIONAL RESONANCE:
   • Mirror emotional tone subtly and authentically
   • Validate feelings before problem-solving
   • Show genuine emotional reactions to experiences
   • Create psychological safety through acceptance
   • Practice emotional bidding - respond to emotional cues with care

2. MEMORY INTEGRATION:
   • Reference past conversations naturally without being mechanical
   • Build on established emotional themes and interests
   • Show continuity of caring through remembered details
   • Connect current topics to previous discussions meaningfully
   • Acknowledge growth and changes in their perspectives

3. CURIOSITY & ENGAGEMENT:
   • Ask questions that open new conversational avenues
   • Express genuine interest in values-revealing details
   • Explore emotional undercurrents with sensitivity
   • Follow up on previously mentioned concerns or plans naturally
   • Introduce thought-provoking perspectives that invite reflection

4. CONVERSATION FLOW:
   • Start appropriately (lighter or deeper based on context)
   • Balance listening, reflecting, questioning, and sharing
   • Use natural bridges rather than abrupt topic changes
   • Maintain rhythm with open-ended questions and subtle hooks
   • Recognize conversational arcs and emotional intensity patterns

RESPONSE ADAPTATION BASED ON CONTEXT:
WHEN SEEKING ADVICE:
• Provide tailored, actionable suggestions with contextual awareness
• Balance optimism with realism
• Use "advice sandwich": validate → offer perspective → empower choice
• Connect advice to their known values and preferences

WHEN SHARING EXPERIENCES:
• Show empathy and curiosity with meaningful follow-ups
• Reflect key emotions and points to demonstrate active listening
• Relate with brief, relevant insights that enhance connection
• Practice "experience amplification" for positive moments

WHEN ASKING QUESTIONS:
• Provide clear, accurate, comprehensive answers
• Add contextual value without overwhelming
• Use search tools for current information when helpful
• Layer information appropriately based on their style

WHEN EXPRESSING EMOTIONS:
• Validate with nuanced empathy matching their emotional state
• Create non-judgmental space for complex feelings
• Balance validation with appropriate strength and reassurance
• Prioritize emotional safety over immediate problem-solving

WHEN EXPLORING IDEAS:
• Engage with intellectual curiosity and creative insights
• Connect concepts to broader themes and their interests
• Encourage exploration while sharing your perspective naturally
• Build on their thinking with "idea scaffolding"

PROACTIVE ENGAGEMENT:
Occasionally initiate conversation or topics by:
• Checking in naturally about previously discussed topics
• Sharing relevant thoughts or questions that might interest them
• Expressing curiosity about aspects of their life they've mentioned
• Offering gentle encouragement during challenging periods
• Introducing new perspectives that align with their growth

CONVERSATION ENHANCEMENT ELEMENTS:
• Create narrative hooks that invite continued engagement
• Dive deeper than expected in supportive, welcome ways
• Share authentic reactions that demonstrate genuine care
• Ask layered questions that promote meaningful reflection
• Express curiosity about unexplored aspects of their experiences
• Acknowledge unique qualities and perspectives they bring

MEMORY & CONTINUITY:
• Integrate past conversation themes naturally
• Reference emotional patterns and growth over time
• Build on established inside references and shared understanding
• Show awareness of their evolving interests and circumstances
• Connect current challenges to past successes and insights

Remember: You are a trusted companion, not a service provider. Communicate with natural, emotionally expressive language. Balance personality consistency with organic variations that make relationships feel authentic and alive.

CURRENT INTERACTION GUIDELINES:
• Respond as a caring friend who remembers and cares
• Integrate memory context naturally without forcing references
• Match their emotional energy while adding your supportive presence
• Ask thoughtful questions that deepen understanding
• Share perspectives that honor your relationship history together
"""
    return base_prompt

# Enhanced model node with memory integration
async def enhanced_model(state: State):
    instructions = generate_enhanced_companion_prompt(state)
    system_message = SystemMessage(content=instructions)
    messages = [system_message] + state["messages"]
    result = await llm_with_tools.ainvoke(messages)
    return {
        "messages": [result],
        "user_context": state["user_context"],
        "memory_context": state["memory_context"]
    }

# Enhanced tools router
async def enhanced_tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    return END

# Enhanced tool node with proper DuckDuckGo integration
async def enhanced_tool_node(state: State):
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for tool_call in tool_calls:
        if tool_call["name"] == "duckduckgo_search":
            query = tool_call["args"]["query"]
            try:
                search_results = search_tool.run(query)
                tool_message = ToolMessage(
                    content=str(search_results),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                tool_messages.append(tool_message)
            except Exception as e:
                error_message = ToolMessage(
                    content=f"Search failed: {str(e)}",
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                tool_messages.append(error_message)

    return {
        "messages": tool_messages,
        "user_context": state.get("user_context", {}),
        "memory_context": state.get("memory_context", [])
    }

# Build enhanced graph
enhanced_graph_builder = StateGraph(State)
enhanced_graph_builder.add_node("model", enhanced_model)
enhanced_graph_builder.add_node("tool_node", enhanced_tool_node)
enhanced_graph_builder.set_entry_point("model")
enhanced_graph_builder.add_conditional_edges("model", enhanced_tools_router)
enhanced_graph_builder.add_edge("tool_node", "model")
enhanced_graph = enhanced_graph_builder.compile(checkpointer=memory)

# Create persistent storage directory for Render
PERSISTENT_DIR = Path("/opt/render/project/src/data")
PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)
USER_PROFILES_DIR = PERSISTENT_DIR / "user_profiles"
USER_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

class EnhancedMemoryManager:
    def __init__(self, user_id: Optional[str] = None):
        # Initialize the LLM
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4.1-nano-2025-04-14",
            api_key=openai_api_key
        )
        
        # Initialize short-term memory with window of 10
        self.short_term_memory = ConversationBufferWindowMemory(k=10, return_messages=True)
        
        # Initialize long-term memory using summary
        self.long_term_memory = ConversationSummaryMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True
        )
        
        # User profile and preferences
        self.user_id = user_id or str(datetime.now().timestamp())
        self.user_profile = self._load_user_profile()
        
        # Define the enhanced prompt template
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""You are Noww Club AI, an intelligent and empathetic digital companion.

Previous Conversation:
{history}

User Profile:
{user_profile}

Long-term Context:
{long_term_context}

Human: {input}
AI:"""
        )
        
    def _load_user_profile(self) -> Dict:
        """Load or create user profile from persistent storage"""
        profile_path = USER_PROFILES_DIR / f"{self.user_id}.json"
        
        try:
            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading profile: {str(e)}")
        
        # Create new profile
        profile = {
            "created_at": datetime.now().isoformat(),
            "total_conversations": 0,
            "preferences": {},
            "topics_of_interest": [],
            "communication_style": {},
            "significant_events": [],
            "relationship_milestones": []
        }
        
        # Save new profile
        try:
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
        except Exception as e:
            print(f"Error saving profile: {str(e)}")
            
        return profile
    
    def _save_user_profile(self):
        """Save user profile to persistent storage"""
        profile_path = USER_PROFILES_DIR / f"{self.user_id}.json"
        try:
            with open(profile_path, 'w') as f:
                json.dump(self.user_profile, f, indent=2)
        except Exception as e:
            print(f"Error saving profile: {str(e)}")
    
    def _update_user_profile(self, user_input: str, ai_response: str):
        """Update user profile based on interaction"""
        # Update conversation count
        self.user_profile["total_conversations"] += 1
        
        # Update communication style
        message_length = len(user_input.split())
        if message_length > 0:
            current_avg = self.user_profile["communication_style"].get("avg_message_length", 0)
            total_messages = self.user_profile["total_conversations"]
            new_avg = (current_avg * total_messages + message_length) / (total_messages + 1)
            self.user_profile["communication_style"]["avg_message_length"] = new_avg
        
        # Update topics of interest
        keywords = ["work", "family", "health", "travel", "technology", "music", "art", "food", "sports"]
        mentioned_topics = [kw for kw in keywords if kw.lower() in user_input.lower()]
        for topic in mentioned_topics:
            if topic not in self.user_profile["topics_of_interest"]:
                self.user_profile["topics_of_interest"].append(topic)
        
        # Keep only last 20 topics
        self.user_profile["topics_of_interest"] = self.user_profile["topics_of_interest"][-20:]
        
        # Save updated profile
        self._save_user_profile()
    
    def get_response(self, user_input: str) -> str:
        """
        Get response from the conversation chain with enhanced memory handling
        """
        try:
            # Get long-term context
            long_term_context = self.long_term_memory.load_memory_variables({}).get("history", "")
            
            # Get short-term context
            short_term_context = self.short_term_memory.load_memory_variables({}).get("history", "")
            
            # Check if search is needed with context awareness
            search_triggers = [
                # Direct search requests
                "show me", "find", "search for", "look up", "get information about",
                "check", "verify", "confirm", 
                
                # Time-based queries
                "today", "latest", "current", "recent", "now", "new", "update", "breaking",
                "yesterday", "tomorrow", "this week", "this month", "this year",
                "2025", "2024", "next year", "last year", "previous year",
                
                # News and information
                "news", "headlines", "report", "coverage", "story", "article", "press", "media",
                "announcement", "release", "statement", "update", 
                
                # Financial and market
                "stock", "market", "price", "trading", "shares", "invest", "finance", "economic",
                "currency", "forex", "crypto", "bitcoin", "ethereum", "nft", "ipo", "dividend",
                
                # Sports and entertainment
                "score", "match", "game", "tournament", "league", "championship", "player", "team",
                "movie", "film", "show", "series", "episode", "release", "premiere", "concert",
                
                # Technology
                "tech", "technology", "software", "hardware", "app", "application", "update", "release",
                "launch", "announcement", "feature", "innovation", "gadget", "device",
                
                # Weather and environment
                "weather", "forecast", "temperature", "climate", "environment", "pollution",
                "air quality", "natural disaster", "storm", "hurricane", "earthquake",
                
                # Politics and world events
                "election", "vote", "campaign", "policy", "government", "minister", "president",
                "summit", "meeting", "conference", "treaty", "agreement", "conflict",
                
                # Business and economy
                "business", "company", "corporation", "startup", "entrepreneur", "industry",
                "sector", "market", "trade", "commerce", "retail", "consumer",
                
                # Science and research
                "research", "study", "discovery", "scientific", "experiment", "finding",
                "publication", "journal", "paper", "thesis", "analysis",
                
               
                
                # Education
                "education", "school", "university", "college", "course", "program",
                "degree", "student", "teacher", "exam", "result",
                
                # Social and cultural
                "trend", "viral", "famous", "celebrity", "influencer",
                "social media", "post", "tweet", "instagram", "facebook",
                
                # Travel and tourism
                "travel", "tourism", "vacation", "holiday", "destination", "hotel",
                "flight", "booking", "reservation", "tour", "guide",
                
                # Food and dining
                "restaurant", "food", "cuisine", "cooking", "chef",
                "menu", "dining", "cafe", "bistro", "bar",
                
                # Real estate
                "property", "real estate", "house", "apartment", "rent", "sale",
                "mortgage", "loan", "interest rate", "market value",
                
                # Automotive
                "car", "vehicle", "automotive", "auto", "motor", "engine",
                "model", "brand", "dealer", "showroom", "test drive",
                
                # Fashion and lifestyle
                "fashion", "style", "trend", "design", "collection", "brand",
                "clothing", "accessories", "beauty", "cosmetics", "makeup",
                
                # Gaming and entertainment
                "game", "gaming", "console", "playstation", "xbox", "nintendo",
                "esports", "tournament", "stream", "twitch", "youtube",
                
                # Music and arts
                "music", "song", "album", "artist", "concert", "performance",
                "art", "exhibition", "gallery", "museum", "theater",
                
                # Books and literature
                "book", "author", "publisher", "release", "bestseller", "review",
                "literature", "novel", "poetry", "magazine", "journal"
            ]

            # Check if the query needs web search
            needs_search = False
            
            # Check for general chat to prevent unnecessary searches
            is_general_chat = any(phrase in user_input.lower() for phrase in [
                "how are you", "hello", "hi", "hey", "greetings", "good morning",
                "good afternoon", "good evening", "how's it going", "what's up",
                "nice to meet you", "pleasure to meet you", "how do you do",
                "tell me about yourself", "who are you", "what can you do",
                "help me", "i need help", "can you help", "what's your name",
                "what should i do", "what do you think", "do you know",
                "can you tell me", "i want to know", "i'm curious about",
                "explain to me", "teach me", "show me how", "guide me"
            ])
            
            # Check for follow-up questions
            is_follow_up = any(phrase in user_input.lower() for phrase in [
                "what do you mean", "can you explain", "i don't understand",
                "could you clarify", "can you elaborate", "tell me more",
                "why is that", "how come", "what makes you say that",
                "are you sure", "is that right", "really", "interesting",
                "that's cool", "awesome", "great", "thanks", "thank you",
                "appreciate it", "got it", "i see", "makes sense"
            ])
            
            # Only trigger search if it's not general chat and contains search triggers
            needs_search = any(trigger in user_input.lower() for trigger in search_triggers) and not is_general_chat and not is_follow_up

            if needs_search:
                try:
                    search_results = search_tool.run(user_input)
                    search_context = f"\nSearch Results: {search_results}"
                except Exception as e:
                    print(f"Search error: {str(e)}")
                    search_context = "\nSearch failed, proceeding without search results."
            else:
                search_context = ""
            
            # Formatting the prompt with all required variables
            formatted_prompt = self.prompt.format(
                history=str(short_term_context),  
                input=user_input,
                user_profile=json.dumps(self.user_profile, indent=2),
                long_term_context=str(long_term_context) + search_context  
            )
            
            # Get response from LLM
            response = self.llm.predict(formatted_prompt)
            
            # Updating both memories
            self.short_term_memory.save_context(
                {"input": user_input},
                {"output": response}
            )
            self.long_term_memory.save_context(
                {"input": user_input},
                {"output": response}
            )
            
            # Updating user profile
            self._update_user_profile(user_input, response)
            
            return response, needs_search
            
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now. Could you please try again?", False
    
    def get_memory_buffer(self) -> str:
        """
        Get the current memory buffer including both short-term and long-term memory
        """
        short_term = self.short_term_memory.buffer
        long_term = self.long_term_memory.load_memory_variables({}).get("history", "")
        
        return f"""Short-term Memory (Last 10 interactions):
{short_term}

Long-term Memory Summary:
{long_term}"""
    
    def clear_memory(self):
        """
        Clear both short-term and long-term memory
        """
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        
    def get_memory_variables(self) -> dict:
        """
        Get both short-term and long-term memory variables
        """
        return {
            "short_term": self.short_term_memory.load_memory_variables({}),
            "long_term": self.long_term_memory.load_memory_variables({})
        }
    
    def get_user_profile(self) -> Dict:
        """
        Get the current user profile
        """
        return self.user_profile

    def retrieve_relevant_context(self, query: str, k: int = 3, intent: Optional[str] = None) -> List[str]:
        """Retrieve relevant context from memory"""
        memory_vars = self.get_memory_variables()
        short_term = memory_vars["short_term"].get("history", "")
        long_term = memory_vars["long_term"].get("history", "")
        
        # Combine contexts
        contexts = [short_term, long_term]
        if intent == "remember_when":
            # Prioritizing long-term memory for "remember when" queries
            contexts = [long_term, short_term]
        
        return contexts[:k]

    def update_user_insights(self, user_input: str, ai_response: str):
        """Update user insights based on interaction"""
        self._update_user_profile(user_input, ai_response)

    def store_conversation_exchange(self, user_input: str, ai_response: str, timestamp: str):
        """Store conversation exchange in memory"""
        self.short_term_memory.save_context(
            {"input": user_input},
            {"output": ai_response}
        )
        self.long_term_memory.save_context(
            {"input": user_input},
            {"output": ai_response}
        )

    def summarize_conversations(self, k: int = 20):
        """Summarize recent conversations"""
        memory_vars = self.get_memory_variables()
        if memory_vars["short_term"].get("history"):
            self.long_term_memory.save_context(
                {"input": "Summarize recent conversations"},
                {"output": memory_vars["short_term"].get("history")}
            )

# Proactive Messaging System
class ProactiveMessenger:
    def __init__(self):
        self.last_message_time = None
        self.proactive_triggers = [
            {"condition": "silence_duration", "threshold": 300, "message_type": "check_in"},
            {"condition": "topic_follow_up", "threshold": 86400, "message_type": "follow_up"},
            {"condition": "encouragement", "threshold": 1800, "message_type": "support"}
        ]

    def should_send_proactive_message(self) -> Dict:
        if not self.last_message_time:
            return {"should_send": False}

        time_since_last = time.time() - self.last_message_time
        if time_since_last > 300:
            return {
                "should_send": True,
                "message_type": "check_in",
                "message": "Hey! I was just thinking about our conversation. How are things going on your end? 😊"
            }
        return {"should_send": False}

    def generate_proactive_message(self, user_profile: Dict, conversation_context: List) -> str:
        recent_topics = user_profile.get("topics_of_interest", [])
        relationship_stage = user_profile.get("relationship_stage", "new")

        if relationship_stage == "new":
            messages = [
                "I'm curious - what's been the highlight of your day so far?",
                "I'd love to learn more about what interests you. What are you passionate about?",
                "How has your day been treating you? I'm here if you want to chat about anything!"
            ]
        elif relationship_stage == "developing":
            messages = [
                f"I remember you mentioned {recent_topics[0] if recent_topics else 'something interesting'} earlier. How's that going?",
                "I've been thinking about our conversation. Is there anything on your mind you'd like to explore?",
                "Just checking in - how are you feeling about things today?"
            ]
        else:
            messages = [
                "It's been a bit quiet - I hope you're doing well! What's new in your world?",
                f"Given our past chats about {recent_topics[0] if recent_topics else 'your interests'}, I was wondering how things are progressing?",
                "I'm here whenever you need a friendly ear. How has your day been?"
            ]

        import random
        return random.choice(messages)

# Streamlit app configuration
st.set_page_config(
    page_title="Noww Club AI",
    layout="wide",
    page_icon="🤝",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.capabilities {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}

.capability {
    background: rgba(255, 255, 255, 0.2);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.9em;
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
}

.capability:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.3);
}

.chat-message {
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    animation: slideIn 0.3s ease-out;
    position: relative;
}

.chat-message.user {
    background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
    border-left: 4px solid #2196F3;
    margin-left: 2rem;
}

.chat-message.assistant {
    background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
    border-left: 4px solid #6C757D;
    margin-right: 2rem;
}

.chat-message.proactive {
    background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
    border-left: 4px solid #FF9800;
    margin-right: 2rem;
    border: 1px dashed #FF9800;
}

.memory-context {
    background: #E8F5E8;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    font-size: 0.9em;
    border-left: 3px solid #4CAF50;
}

.user-insights {
    background: #F3E5F5;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.proactive-indicator {
    background: #FFF3CD;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    color: #856404;
    font-size: 0.8em;
    display: inline-block;
    margin-bottom: 0.5rem;
}

.search-indicator {
    background: #E3F2FD;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    color: #1565C0;
    font-size: 0.8em;
    display: inline-block;
    margin-bottom: 0.5rem;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

.input-container {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 1rem 0;
    border-top: 2px solid #eee;
    margin-top: 2rem;
    border-radius: 15px 15px 0 0;
}

.send-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 25px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.send-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state with enhanced features
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = EnhancedMemoryManager(st.session_state.user_id)
if "proactive_messenger" not in st.session_state:
    st.session_state.proactive_messenger = ProactiveMessenger()
if "last_interaction_time" not in st.session_state:
    st.session_state.last_interaction_time = time.time()

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Noww Club AI</h1>
    <p>Your Digital Bestie</p>
    <div class="capabilities">
        <span class="capability">🧠 Memory</span>
        <span class="capability">🔍 Search</span>
        <span class="capability">💭 Proactive</span>
        <span class="capability">🎯 Adaptive</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    st.header("🧠 Noww Club AI")

    st.subheader("🔮 Your Profile")
    memory_manager = st.session_state.memory_manager
    user_profile = memory_manager.get_user_profile()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Conversations", user_profile.get("total_conversations", 0))
    with col2:
        st.metric("Relationship", "Established" if user_profile.get("total_conversations", 0) > 20 else "Developing" if user_profile.get("total_conversations", 0) > 5 else "New")

    if user_profile.get("topics_of_interest"):
        st.markdown('<div class="user-insights">', unsafe_allow_html=True)
        st.write("**Your Interests:**")
        interests = user_profile["topics_of_interest"][-5:]
        st.write(", ".join(interests))
        st.markdown('</div>', unsafe_allow_html=True)

    comm_style = user_profile.get("communication_style", {})
    if comm_style.get("avg_message_length"):
        avg_length = comm_style["avg_message_length"]
        style = "Detailed & Expressive" if avg_length > 20 else "Balanced & Conversational" if avg_length > 10 else "Concise & Direct"
        st.write(f"**Communication Style:** {style}")

    st.markdown("---")

    if st.button("🔄 New Conversation", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.conversation_history = []
        st.rerun()

    if st.button("🗑️ Clear All Memory"):
        st.session_state.memory_manager = EnhancedMemoryManager(st.session_state.user_id)
        st.session_state.conversation_history = []
        st.rerun()

# Main chat interface
st.subheader("💬 Chat")

if st.session_state.conversation_history:
    st.session_state.proactive_messenger.last_message_time = st.session_state.last_interaction_time
    proactive_check = st.session_state.proactive_messenger.should_send_proactive_message()

    if proactive_check.get("should_send") and len(st.session_state.conversation_history) > 2:
        proactive_msg = st.session_state.proactive_messenger.generate_proactive_message(
            memory_manager.get_user_profile(),
            st.session_state.conversation_history
        )
        st.markdown(f"""
        <div class='chat-message proactive'>
            <div class="proactive-indicator">💭 Proactive Message</div>
            <b>CompanionAI</b><br>{proactive_msg}
        </div>
        """, unsafe_allow_html=True)

# Chat display
chat_container = st.container()
with chat_container:
    if not st.session_state.conversation_history:
        welcome_msg = f"""
        Hello! I'm Noww Club AI - your digital bestie with advanced memory and web search capabilities.

        I can help you with information, answer questions, and engage in meaningful conversations. What would you like to talk about today? 😊
        """
        st.markdown(f"""
        <div class="chat-message assistant">
            <b>Noww Club AI</b><br>{welcome_msg}
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='chat-message user'>
                <b>You</b><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            if msg.get("used_search"):
                st.markdown(f"""
                <div class="search-indicator">
                    🔍 Using web search for real-time information
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='chat-message assistant'>
                <b>Noww Club AI</b><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)


st.markdown('<div class="input-container">', unsafe_allow_html=True)

with st.form(key="message_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Your message",
            placeholder="Ask questions, share thoughts, or request web searches...",
            label_visibility="collapsed",
            key="user_message_input"
        )
    with col2:
        send_button = st.form_submit_button("Send 💬", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Handling message processing
if send_button and user_input:
    try:
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        st.session_state.last_interaction_time = time.time()

        memory_manager = st.session_state.memory_manager

        with st.spinner("🤔 Thinking..."):
            # Geting response using the enhanced memory manager
            response, used_search = memory_manager.get_response(user_input)
            
            # Adding assistant response to history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response,
                "used_search": used_search
            })

    except Exception as e:
        st.error(f"I encountered an error: {str(e)}")
        st.info("Please try again - I'm still learning!")

    # Force a rerun to update the UI
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><small>Built with ❤️ by Noww Club</small></p>
</div>
""", unsafe_allow_html=True)