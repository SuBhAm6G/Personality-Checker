import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from utils import get_questions, calculate_scores, predict_archetype, TRAITS

# --- Page Config ---
st.set_page_config(
    page_title="Solaris | Personality Detector",
    page_icon="☀️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS - Retro Fancy ---
def inject_retro_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Righteous&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

        /* Global */
        * {
            font-family: 'Righteous', cursive;
        }
        
        /* Background - Groovy Gradient */
        .stApp {
            background: linear-gradient(135deg, #ff006e 0%, #fb5607 25%, #ffbe0b 50%, #8338ec 75%, #3a86ff 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Retro Card with Glow */
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
            background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
            padding: 50px 40px;
            border-radius: 0px;
            border: 4px solid #ff006e;
            box-shadow: 
                0 0 20px rgba(255, 0, 110, 0.6),
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.6);
            font-size: 3rem;
            margin-bottom: 1rem;
            margin-bottom: 1.5rem;
            text-shadow: 2px 2px 0px #ff006e;
            letter-spacing: 1px;
        }

        h3 {
            color: #fb5607;
            font-weight: 700;
            font-size: 1.3rem;
            margin-bottom: 1rem;
            text-shadow: 1px 1px 0px #3a86ff;
        }

        p {
            color: #1a1a1a;
            font-size: 1.1rem;
            line-height: 1.8;
            font-family: 'Fredoka One', cursive;
        }

        /* Buttons - Neon Style */
        .stButton > button {
            background: linear-gradient(135deg, #ff006e, #fb5607);
            color: #fff;
            border: 3px solid #ffbe0b;
            padding: 16px 40px;
            border-radius: 0px;
            font-size: 1.2rem;
            font-weight: 900;
            transition: all 0.3s;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 
                0 0 20px rgba(255, 0, 110, 0.8),
                0 8px 16px rgba(0, 0, 0, 0.3);
            font-family: 'Righteous', cursive;
            cursor: pointer;
            position: relative;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #fb5607, #ffbe0b);
            transform: translate(-3px, -3px);
            box-shadow: 
                0 0 30px rgba(255, 190, 11, 0.9),
                12px 12px 0px rgba(51, 134, 252, 0.6);
        }

        .stButton > button:active {
            transform: translate(0, 0);
        }

        /* Slider */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #ff006e, #3a86ff);
        }

        .stSlider div[role="slider"] {
            background-color: #ffbe0b !important;
            border: 3px solid #ff006e !important;
            box-shadow: 0 0 15px rgba(255, 0, 110, 0.8);
        }

        /* Progress Bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #ff006e, #fb5607, #ffbe0b);
            box-shadow: 0 0 20px rgba(255, 0, 110, 0.8);
        }

            border: 3px solid #ff006e !important;
            border-radius: 0px !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }

        ul[role="listbox"] li {
            color: #1a1a1a !important;
            padding: 12px 16px;
            font-family: 'Fredoka One', cursive;
            border-bottom: 1px solid rgba(255, 0, 110, 0.3);
        }

        ul[role="listbox"] li:hover {
            background: linear-gradient(135deg, #ffbe0b, #fb5607) !important;
            color: white !important;
        }

        /* Number Input Stepper */
        .stNumberInput input {
            font-family: 'VT323', monospace !important;
        }

        /* Label styling */
        label {
            font-family: 'Righteous', cursive !important;
        }

    </style>
    """, unsafe_allow_html=True)

inject_retro_theme()

# --- Session State ---
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'current_q_index' not in st.session_state:
    st.session_state.current_q_index = 0
if 'user_text' not in st.session_state:
    st.session_state.user_text = ""
if 'demographics' not in st.session_state:
    st.session_state.demographics = {}
if 'question_list' not in st.session_state:
    st.session_state.question_list = []

# --- Navigation ---
def go_to_demographics():
    st.session_state.page = 'demographics'
    st.rerun()

def start_quiz(age, gender, occupation):
    st.session_state.demographics = {'age': age, 'gender': gender, 'occupation': occupation}
    st.session_state.question_list = get_questions(age, occupation)
    st.session_state.page = 'quiz'
    st.rerun()

def next_question():
    st.session_state.current_q_index += 1
    st.rerun()

def finish_quiz():
    st.session_state.page = 'analysis'
    st.rerun()

def restart():
    st.session_state.page = 'landing'
    st.session_state.answers = {}
    st.session_state.current_q_index = 0
    st.session_state.user_text = ""
    st.session_state.demographics = {}
    st.session_state.question_list = []
    st.rerun()

# --- Pages ---

def landing_page():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("☀️ Solaris")
    st.markdown("### 🌈 Personality Detector")
    st.write("✨ Step into the groovy light of self-discovery ✨")
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if st.button("🎯 Begin Journey"):
        go_to_demographics()
    st.markdown('</div>', unsafe_allow_html=True)

def demographics_page():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("👤 Who Are You?")
    st.write("Customize your cosmic experience.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary", "Prefer not to say"])
    occupation = st.selectbox("Current Occupation", ["Student", "Working Professional", "Self-Employed", "Retired", "Other"])
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if st.button("⚡ Continue"):
        start_quiz(age, gender, occupation)
    st.markdown('</div>', unsafe_allow_html=True)

def quiz_page():
    q_list = st.session_state.question_list
    total_q = len(q_list)
    current_idx = st.session_state.current_q_index
    
    # Progress
    progress = (current_idx / (total_q + 1))
    st.progress(progress)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    if current_idx < total_q:
        q = q_list[current_idx]
        st.markdown(f"### 🔮 Question {current_idx + 1}/{total_q}")
        st.markdown(f"## {q['text']}")
        
        # Slider
        val = st.slider("Rate your vibe", 1, 5, 3, key=f"q_{q['id']}", label_visibility="collapsed")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        if st.button("→ Next"):
            st.session_state.answers[q['id']] = val
            next_question()
            
    else:
        # Final Step
        st.markdown("### 💫 Final Reflection")
        st.markdown("## Describe yourself...")
        
        user_input = st.text_area("I am...", height=150, placeholder="Share your essence here...")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        if st.button("🌟 Reveal Persona"):
            st.session_state.user_text = user_input
            finish_quiz()

    st.markdown('</div>', unsafe_allow_html=True)

def results_page():
    scores = calculate_scores(st.session_state.answers, st.session_state.question_list, st.session_state.user_text)
    archetype, distance = predict_archetype(scores)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title(f"🎆 {archetype}")
    st.write("✨ Your solar signature has been revealed ✨")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Radar Chart - Retro Colors
    categories = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='You',
        line_color='#ff006e',
        fillcolor='rgba(255, 0, 110, 0.4)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                linecolor='rgba(255, 190, 11, 0.6)',
                gridcolor='rgba(255, 0, 110, 0.3)',
                tickcolor='#ff006e'
            ),
            bgcolor='rgba(25, 84, 123, 0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ff006e', size=16, family='Righteous'),
        margin=dict(l=80, r=80, t=80, b=80),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if st.button("🔄 Another Journey"):
        restart()
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main Routing ---
if st.session_state.page == 'landing':
    landing_page()
elif st.session_state.page == 'demographics':
    demographics_page()
elif st.session_state.page == 'quiz':
    quiz_page()
elif st.session_state.page == 'analysis':
    results_page()