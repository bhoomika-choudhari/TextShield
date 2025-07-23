# agent_runner.py
import streamlit as st
def run_agentic_task(prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
# Optional if you use st.secrets in frontend

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        safety_settings={
            "HARASSMENT": "BLOCK_NONE",
            "HATE": "BLOCK_NONE",
            "SEXUAL": "BLOCK_NONE",
            "DANGEROUS": "BLOCK_NONE"
        }
    )

    response = model.generate_content(prompt)
    return response.text.strip()





