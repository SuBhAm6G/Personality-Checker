import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.neighbors import NearestNeighbors

# --- Constants & Configuration ---

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# Expanded Questions with Tags
# Tags: 'all', 'student', 'worker', 'retired', 'young', 'mature'
QUESTIONS_POOL = [
    {"id": 1, "text": "I have a vivid imagination.", "trait": "Openness", "tags": ["all"]},
    {"id": 2, "text": "I get chores done right away.", "trait": "Conscientiousness", "tags": ["all"]},
    {"id": 3, "text": "I am the life of the party.", "trait": "Extraversion", "tags": ["all"]},
    {"id": 4, "text": "I sympathize with others' feelings.", "trait": "Agreeableness", "tags": ["all"]},
    {"id": 5, "text": "I get stressed out easily.", "trait": "Neuroticism", "tags": ["all"]},
    
    # Student specific
    {"id": 11, "text": "I often explore new subjects outside my curriculum.", "trait": "Openness", "tags": ["student"]},
    {"id": 12, "text": "I submit my assignments well before the deadline.", "trait": "Conscientiousness", "tags": ["student"]},
    {"id": 13, "text": "I prefer studying in groups rather than alone.", "trait": "Extraversion", "tags": ["student"]},
    
    # Worker specific
    {"id": 21, "text": "I constantly look for better ways to do my job.", "trait": "Openness", "tags": ["worker"]},
    {"id": 22, "text": "I keep my workspace organized and tidy.", "trait": "Conscientiousness", "tags": ["worker"]},
    {"id": 23, "text": "I take charge in team meetings.", "trait": "Extraversion", "tags": ["worker"]},
    {"id": 24, "text": "I try to mediate conflicts between colleagues.", "trait": "Agreeableness", "tags": ["worker"]},
    
    # Age based (Young < 30)
    {"id": 31, "text": "I enjoy trying out new trends and fads.", "trait": "Openness", "tags": ["young"]},
    {"id": 32, "text": "I worry about my future career path.", "trait": "Neuroticism", "tags": ["young"]},
    
    # Age based (Mature >= 30)
    {"id": 41, "text": "I prefer established routines over surprises.", "trait": "Conscientiousness", "tags": ["mature"]},
    {"id": 42, "text": "I feel secure in my life choices.", "trait": "Neuroticism", "tags": ["mature"]}, # Reverse coded logic handled in scoring? For simplicity, we'll assume high score = high neuroticism, so this might need context. Let's keep it simple: "I worry about stability"
    {"id": 43, "text": "I worry about financial stability.", "trait": "Neuroticism", "tags": ["mature"]},
]

# Archetypes (Same as before)
ARCHETYPES = {
    "The Visionary": [90, 70, 60, 60, 40],
    "The Executor": [40, 90, 70, 50, 30],
    "The Socialite": [60, 50, 90, 80, 20],
    "The Peacemaker": [50, 60, 40, 90, 30],
    "The Sentinel": [30, 80, 40, 50, 80],
    "The Balanced Realist": [50, 50, 50, 50, 50]
}

def get_questions(age, occupation):
    """
    Returns a list of questions filtered by user demographics.
    """
    age = int(age)
    selected_questions = []
    
    for q in QUESTIONS_POOL:
        tags = q["tags"]
        
        # Include 'all'
        if "all" in tags:
            selected_questions.append(q)
            continue
            
        # Occupation filter
        if occupation.lower() == "student" and "student" in tags:
            selected_questions.append(q)
        elif occupation.lower() != "student" and "worker" in tags: # Broad bucket for worker
            selected_questions.append(q)
            
        # Age filter
        if age < 30 and "young" in tags:
            selected_questions.append(q)
        elif age >= 30 and "mature" in tags:
            selected_questions.append(q)
            
    return selected_questions

def analyze_text_input(text):
    if not text or len(text.strip()) < 5:
        return np.zeros(5)
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    adjustments = np.array([subjectivity * 10, 0, sentiment * 5, sentiment * 10, -sentiment * 10])
    return adjustments

def calculate_scores(answers, questions_list, text_input=None):
    """
    Calculates scores based on the specific list of questions asked.
    """
    raw_scores = {t: 0 for t in TRAITS}
    counts = {t: 0 for t in TRAITS}
    
    # Map ID to trait from the questions_list
    q_map = {q["id"]: q["trait"] for q in questions_list}
    
    for qid, score in answers.items():
        if qid in q_map:
            trait = q_map[qid]
            raw_scores[trait] += score
            counts[trait] += 1
            
    final_scores = []
    for t in TRAITS:
        if counts[t] > 0:
            avg = raw_scores[t] / counts[t]
            norm_score = (avg - 1) / 4 * 100
        else:
            norm_score = 50
        final_scores.append(norm_score)
        
    final_scores = np.array(final_scores)
    
    if text_input:
        adj = analyze_text_input(text_input)
        final_scores = final_scores + adj
        
    final_scores = np.clip(final_scores, 0, 100)
    return dict(zip(TRAITS, final_scores))

def predict_archetype(scores_dict):
    user_vector = np.array([scores_dict[t] for t in TRAITS]).reshape(1, -1)
    data = np.array(list(ARCHETYPES.values()))
    labels = list(ARCHETYPES.keys())
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(data)
    distances, indices = nbrs.kneighbors(user_vector)
    return labels[indices[0][0]], distances[0][0]
