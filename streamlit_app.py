import streamlit as st
import os
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

# Note: In a production environment, use Streamlit's native secrets management.
# This part is for local testing with a .env file.
if not st.secrets:
    load_dotenv()
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
else:
    # On Streamlit Cloud, access secrets directly
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

# Pydantic model for the request body (moved here from backend)
class PromptRequest(BaseModel):
    prompt: str

# --- DeepSeek API Configuration ---
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# The prompt engineering knowledge base (moved here from backend)
KNOWLEDGE_BASE = """
Effective Prompt Engineering: The 4C's Framework
To transform a vague prompt into a well-defined one, incorporate these four essential elements:
Context: Specify who you are, your role, and the situation. Include your background, expertise level, and the specific circumstances requiring AI assistance. This helps the AI understand your perspective and tailor responses appropriately.
Constraints: Define clear boundaries including word count, format requirements, tone (academic, casual, professional), deadlines, and any limitations on tools or sources. Specify what to avoid and include safety parameters when relevant.
Content Goal: Articulate exactly what output you need - whether it's a summary, analysis, plan, code, or creative content. Be specific about the type of deliverable, its purpose, and intended audience. Replace broad requests with precise objectives.
Check: Include verification criteria by asking the AI to confirm sources are real (not hallucinated), validate structure meets requirements, or explain its reasoning. Add acceptance tests you'll use to evaluate the output's quality and accuracy.
Implementation: Structure your prompt as: "[ROLE/CONTEXT] + [SPECIFIC GOAL] + [FORMAT/LENGTH/TONE] + [KEY DETAILS] + [VERIFICATION REQUEST]"
This framework transforms generic requests like "tell me about X" into targeted instructions that produce reliable, useful, and actionable AI outputs while minimizing errors and misunderstandings.
"""

# The core logic, now a regular function
def refine_prompt_core(user_prompt: str):
    if not DEEPSEEK_API_KEY:
        st.error("DeepSeek API key not found. Please add it to your Streamlit secrets.")
        return None

    full_prompt_for_deepseek = f"""
    You are a world-class prompt engineer. Your task is to take a basic user prompt and, using the "4C's Framework" provided below,
    transform it into a highly detailed, professional, and effective prompt for a large language model.

    **Instructions:**
    1.  Read and apply the "4C's Framework" provided in the <knowledge_base> section.
    2.  Based on the user's <vague_prompt>, you must *invent* and *make up* specific, realistic details for the missing 4C's elements (Context, Constraints, Content Goal, Check). This is crucial. For example, if the user says "write a poem," you should invent a persona like "a hopeful poet" and constraints like "a sonnet with 14 lines."
    3.  Combine these invented details with the user's prompt to create a final, well-structured prompt.
    4.  The final output must be only the complete, refined prompt, formatted neatly and without any additional conversation or explanation.

    <knowledge_base>
    {KNOWLEDGE_BASE}
    </knowledge_base>

    <vague_prompt>
    {user_prompt}
    </vague_prompt>
    """

    messages = [
        {"role": "user", "content": full_prompt_for_deepseek}
    ]

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = httpx.post(
            DEEPSEEK_API_URL, 
            json=payload, 
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        
        api_response = response.json()
        refined_prompt = api_response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        return refined_prompt

    except httpx.HTTPStatusError as e:
        st.error(f"DeepSeek API error: {e.response.text}")
        return None
    except httpx.RequestError as e:
        st.error(f"An error occurred while calling the DeepSeek API: {e}")
        return None

# --- UI Components ---
st.title("💡 AI Prompt Engineer")
st.markdown("Enter a simple, vague prompt below, and our AI will transform it into a professional, well-rounded prompt.")

user_prompt = st.text_area(
    "Your Basic Prompt", 
    height=150, 
    placeholder="e.g., write a report about renewable energy"
)

if st.button("Engineer Prompt", help="Click to get a professional prompt."):
    if user_prompt:
        with st.spinner("Engineering your prompt..."):
            refined_prompt = refine_prompt_core(user_prompt)

            if refined_prompt:
                st.success("Prompt Refined Successfully!")
                st.subheader("Final Engineered Prompt:")
                st.code(refined_prompt, language="text")
            else:
                st.error("Error: Could not retrieve a refined prompt. Please check your API key and try again.")
    else:
        st.warning("Please enter a prompt to engineer.")

