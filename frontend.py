import streamlit as st

st.set_page_config(page_title="GPT Shield", layout='wide')
import time
from agent_runner import run_agentic_task
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords
import string
import json
import os

nltk.download('punkt')
nltk.download('stopwords')

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return tokenizer, model

tokenizer, model = load_model()

def calculate_perplexity(text):
    encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    input_ids = encoded_input[0]
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0))
        logits = outputs.logits
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    return torch.exp(loss).item()

def calculate_burstiness(text):
    tokens = nltk.word_tokenize(text.lower())
    word_freq = FreqDist(tokens)
    repeated_count = sum(count > 1 for count in word_freq.values())
    return repeated_count / len(word_freq)

def plot_top_repeated_words(text):
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    words, counts = zip(*top_words) if top_words else ([], [])
    fig = px.bar(x=words, y=counts, labels={'x': 'Words', 'y': 'Counts'}, title="Top 10 Most Repeated Words")
    st.plotly_chart(fig, use_container_width=True)
import streamlit as st
# ... all other imports like torch, plotly, speech_recognition, agent_runner, etc.

# Your login_page(), signup_page() etc. should be defined before or after this

def welcome_screen():
    st.markdown("""
        <style>
        body {
            background-image: url('https://images.unsplash.com/photo-1634942537031-7963b6a3a25f?auto=format&fit=crop&w=1950&q=80');
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            color: white;
        }

        .fade-in {
            animation: fadeIn 2s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .typewriter h1 {
            overflow: hidden;
            border-right: .15em solid white;
            white-space: nowrap;
            margin: 0 auto;
            animation:
                typing 3s steps(30, end),
                blink-caret .75s step-end infinite;
        }

        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }

        @keyframes blink-caret {
            from, to { border-color: transparent }
            50% { border-color: white; }
        }

        .centered {
            text-align: center;
            margin-top: 50px;
        }

        .robot-img {
            width: 160px;
        }

        .logo-gif {
            width: 80px;
            margin-top: 10px;
        }

        .blur-bg {
            background: rgba(0, 0, 0, 0.65);
            padding: 40px;
            border-radius: 20px;
            max-width: 600px;
            margin: 50px auto;
        }
        </style>

        <div class="blur-bg fade-in">
            <div class="centered">
                <img class="robot-img" src="https://cdn-icons-png.flaticon.com/512/4712/4712102.png" alt="Robot">
                <img class="logo-gif" src="https://media.giphy.com/media/MF1Vpej2R7PBL4ElwE/giphy.gif" alt="GPT Logo">
                <div class="typewriter">
                    <h1>Welcome to GPT Shield</h1>
                </div>
                <p style='font-size:18px;'>Your AI-Powered Writing & Detection Companion</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    login_page()

def load_users():
    if not os.path.exists("users.json"):
        with open("users.json", "w") as f:
            json.dump({}, f)
    with open("users.json", "r") as f:
        return json.load(f)

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

def signup(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = {"password": password, "history": []}
    save_users(users)
    return True

def login(username, password):
    users = load_users()
    return username in users and users[username]["password"] == password


# Set initial session state for login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "users" not in st.session_state:
    st.session_state.users = {"admin": "1234"}  # default user

def login_page():
    st.markdown("## 🔐 GPT Shield Login")
    auth_mode = st.radio("Login or Signup", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Submit"):
        if auth_mode == "Login":
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.success("Logged in successfully.")
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")
        else:  # Signup
            if username in st.session_state.users:
                st.warning("Username already exists.")
            else:
                st.session_state.users[username] = password
                st.success("Signup successful. Please login now.")

if not st.session_state.authenticated:
    welcome_screen()
    st.stop()

# 🛑 Login screen (after welcome)
if not st.session_state.authenticated:
    login_page()
    st.stop()


# Sidebar
agent_mode = st.sidebar.toggle("🧠 Enable Agentic AI", value=True)
st.sidebar.title("🧰 GPT Shield Toolkit")
selected_tool = st.sidebar.radio(
    "Choose a feature:",
    ["AI Content Detection", "Document Analyzer", "AI Rewriter", "Readability & Grammar", "AI Chat Assistant"],
    key="main_tool_selector"
)

if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    
    # Show toast and auto-refresh with animation
    st.markdown("""
        <script>
        // Toast message
        alert("✅ You have been logged out successfully!");
        
        // Smooth redirect after 1 second
        setTimeout(function() {
            window.location.reload();
        }, 1000);
        </script>
    """, unsafe_allow_html=True)
    
    st.stop()



#Theme mode
theme_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
if theme_mode:
    st.markdown("""
        <style>
        body { background-color: #111; color: white; }
        .stButton>button { background-color: #444; color: white; }
        </style>
    """, unsafe_allow_html=True)

# AI Content Detection (placeholder)
if selected_tool == "AI Content Detection":
    st.header("🛡️ GPT Shield: AI Content Detector")
    with st.expander("📘 What is Perplexity?"):
        st.markdown("""
        Perplexity is a measure of how predictable a text is for a language model.
        - **Low Perplexity**: Model finds the text predictable → could be AI-generated.
        - **High Perplexity**: Text is more complex or less predictable → more likely human-written.
        """)


    text_area = st.text_area("✍️ Enter your text")

    if text_area:
        if st.button("Analyze"):

            if len(text_area.split()) < 20:
                st.warning("⚠️ Please enter at least 20 words for accurate analysis.")
            else:
                # Auto scroll to bottom
                st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.info("📄 Your Input Text")
                    st.success(text_area)

                with col2:
                    st.info("📊 Calculated Scores")
                    perplexity = calculate_perplexity(text_area)
                    burstiness_score = calculate_burstiness(text_area)

                    st.success(f"Perplexity Score: {perplexity:.2f}")
                    st.success(f"Burstiness Score: {burstiness_score:.2f}")

                    if perplexity > 30000 and burstiness_score < 0.2:
                        st.error("🔍 AI Generated Content Detected")
                    elif 10000 < perplexity <= 30000 or 0.2 <= burstiness_score < 0.4:
                        st.warning("🤖 Possibly AI-Assisted Content")
                    else:
                        st.success("✅ Likely Human-Written Content")

                    # 📥 Download Result Button
                    result = f"""
    GPT Shield - AI Plagiarism Detection Report

    Input:
    {text_area.strip()}

    Perplexity Score: {perplexity:.2f}
    Burstiness Score: {burstiness_score:.2f}

    Result: {"AI Generated" if perplexity > 30000 and burstiness_score < 0.2 else "Possibly AI-Assisted" if 10000 < perplexity <= 30000 or 0.2 <= burstiness_score < 0.4 else "Likely Human-Written"}
    """.strip()

                    st.download_button("📥 Download Report", result, file_name="gpt_shield_report.txt")

                with col3:
                    st.info("📈 Basic Insights")
                    plot_top_repeated_words(text_area)
    # st.info("⚠️ This tool is not affected by Agent Mode.")
    # st.write("Please scroll up to view the full original AI Detection logic.")
    # st.warning("This section is unchanged — review manually if needed.")

# Document Analyzer (placeholder)
elif selected_tool == "Document Analyzer":
    st.header("📄 Document Analyzer")
    st.info("⚠️ This tool is not affected by Agent Mode.")
    st.write("Upload a `.pdf`, `.docx`, or `.txt` file to analyze for AI-generated content.")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

    extracted_text = ""

    if uploaded_file:
        file_type = uploaded_file.type

        if file_type == "text/plain":
            extracted_text = uploaded_file.read().decode("utf-8")

        elif file_type == "application/pdf":
            try:
                import fitz  # PyMuPDF
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                    for page in doc:
                        extracted_text += page.get_text()
            except Exception as e:
                st.error(f"Error reading PDF: {e}")

        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                from docx import Document
                doc = Document(uploaded_file)
                extracted_text = "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                st.error(f"Error reading DOCX: {e}")

    if extracted_text:
        st.subheader("📃 Extracted Text")
        st.text_area("Text from uploaded file:", value=extracted_text, height=300)

        if st.button("Analyze Document"):
            if len(extracted_text.split()) < 20:
                st.warning("⚠️ The document is too short for reliable analysis.")
            else:
                perplexity = calculate_perplexity(extracted_text)
                burstiness_score = calculate_burstiness(extracted_text)

                st.success(f"Perplexity Score: {perplexity:.2f}")
                st.success(f"Burstiness Score: {burstiness_score:.2f}")

                if perplexity > 30000 and burstiness_score < 0.2:
                    st.error("🔍 AI Generated Content Detected")
                elif 10000 < perplexity <= 30000 or 0.2 <= burstiness_score < 0.4:
                    st.warning("🤖 Possibly AI-Assisted Content")
                else:
                    st.success("✅ Likely Human-Written Content")

                result = f"""
GPT Shield - Document Report

Perplexity Score: {perplexity:.2f}
Burstiness Score: {burstiness_score:.2f}

Result: {"AI Generated" if perplexity > 30000 and burstiness_score < 0.2 else "Possibly AI-Assisted" if 10000 < perplexity <= 30000 or 0.2 <= burstiness_score < 0.4 else "Likely Human-Written"}
""".strip()

                st.download_button("📥 Download Report", result, file_name="document_report.txt")

    st.warning("This section is unchanged — review manually if needed.")

# AI Rewriter
elif selected_tool == "AI Rewriter":
    st.header("✍️ AI Rewriter / Paraphraser")
    input_text = st.text_area("Enter the text you want to rephrase:")

    if st.button("🔄 Rephrase Text"):
        if input_text.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            with st.spinner("Rewriting..."):
                try:
                    if agent_mode:
                        from agent_runner import run_agentic_task
                        prompt = f"Rewrite this text in a formal tone:\n\n{input_text}"
                        rewritten_text = run_agentic_task(prompt)
                    else:
                        import google.generativeai as genai
                        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        response = model.generate_content(f"Paraphrase this text in natural English:\n\n{input_text}")
                        rewritten_text = response.text.strip()

                    st.success("✅ Rephrased Successfully!")
                    st.text_area("Rewritten Text:", value=rewritten_text, height=200)
                    st.download_button("📥 Download Rewritten Text", rewritten_text, file_name="rewritten_text.txt")

                except Exception as e:
                    st.error(f"⚠️ Failed to rewrite: {e}")

# Readability & Grammar
elif selected_tool == "Readability & Grammar":
    st.header("🧠 Readability & Grammar Analyzer")
    language = st.selectbox("🌐 Choose Language", ["English", "Hindi"])
    input_text = st.text_area("Enter your text for analysis:")

    if st.button("🔍 Analyze Readability & Grammar"):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            try:
                import textstat
                import plotly.graph_objs as go
                from gtts import gTTS
                from io import BytesIO
                import base64

                # Readability
                flesch = textstat.flesch_reading_ease(input_text)
                grade_level = textstat.text_standard(input_text)

                st.subheader("📊 Readability Scores")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=flesch,
                    title={'text': "Flesch Reading Ease"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "blue"},
                           'steps': [
                               {'range': [0, 30], 'color': "red"},
                               {'range': [30, 60], 'color': "orange"},
                               {'range': [60, 100], 'color': "green"}]}))
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"📘 Suggested Grade Level: {grade_level}")

                # Grammar Correction
                if agent_mode:
                    from agent_runner import run_agentic_task
                    prompt = f"Fix grammar and explain corrections:\n\n{input_text}"
                    corrected = run_agentic_task(prompt)
                    st.text_area("Corrected Text (Agent):", value=corrected, height=200)
                else:
                    import language_tool_python
                    tool = language_tool_python.LanguageToolPublicAPI('en-US')
                    matches = tool.check(input_text)
                    corrected = language_tool_python.utils.correct(input_text, matches)

                    #####
                    st.subheader("🔧 Grammar Suggestions")
                    st.info(f"Number of issues detected: {len(matches)}")
                    for match in matches[:5]:
                        st.write(f"• {match.message} (suggestion: `{', '.join(match.replacements)}`)")
                    st.markdown("**✏️ Corrected Version:**")
                    st.text_area("Corrected Text:", value=corrected, height=200)
                    ######

                st.markdown("**🔊 Listen to Corrected Text:**")
                tts = gTTS(corrected, lang="en" if language == "English" else "hi")
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                b64_audio = base64.b64encode(audio_bytes.read()).decode()
                st.audio(f"data:audio/mp3;base64,{b64_audio}", format="audio/mp3")

            

            except Exception as e:
                st.error(f"⚠️ Grammar analysis failed: {e}")

# AI Chat Assistant
elif selected_tool == "AI Chat Assistant":
    st.header("💬 AI Chat Assistant")
    st.write("Ask anything and get smart responses powered by GPT!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                if agent_mode:
                    from agent_runner import run_agentic_task
                    full_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history]) + "\nAssistant:"
                    assistant_reply = run_agentic_task(full_prompt)
                else:
                    import google.generativeai as genai
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    full_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history]) + "\nAssistant:"
                    response = model.generate_content(full_prompt)
                    assistant_reply = response.text.strip()

                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

            except Exception as e:
                st.error(f"❌ Failed to get response: {e}")
    

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                try:
                    from gtts import gTTS
                    from io import BytesIO
                    import base64

                    tts = gTTS(text=msg["content"], lang='en')
                    audio_bytes = BytesIO()
                    tts.write_to_fp(audio_bytes)
                    audio_bytes.seek(0)
                    b64_audio = base64.b64encode(audio_bytes.read()).decode()
                    st.audio(f"data:audio/mp3;base64,{b64_audio}", format="audio/mp3")
                except Exception as e:
                    st.warning(f"🔊 TTS failed: {e}")
