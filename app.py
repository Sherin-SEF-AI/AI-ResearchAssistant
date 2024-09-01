import streamlit as st
import requests
import json
from collections import Counter
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import bcrypt
from textblob import TextBlob
import seaborn as sns
from pyvis.network import Network
import base64
from io import BytesIO
import yake
import folium
from streamlit_folium import folium_static
import random

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Gemini API configuration
GEMINI_API_KEY = "Enter Your Gemini API herer"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# Database setup
conn = sqlite3.connect('research_assistant.db')
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS queries
             (id INTEGER PRIMARY KEY, user_id INTEGER, query TEXT, result TEXT,
              FOREIGN KEY (user_id) REFERENCES users(id))''')
c.execute('''CREATE TABLE IF NOT EXISTS chat_history
             (id INTEGER PRIMARY KEY, user_id INTEGER, message TEXT, response TEXT,
              FOREIGN KEY (user_id) REFERENCES users(id))''')
conn.commit()

# Streamlit page configuration
st.set_page_config(page_title="AI Research Assistant", layout="wide")

# Custom CSS for modern UI with dark theme tabs
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .Widget>label {
        color: #31333F;
        font-weight: 600;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stSelectbox>div>div>select {
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E1E1E;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Authentication functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def create_user(username, password):
    hashed_password = hash_password(password)
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()

def get_user(username):
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    return c.fetchone()

def save_query(user_id, query, result):
    c.execute("INSERT INTO queries (user_id, query, result) VALUES (?, ?, ?)", (user_id, query, result))
    conn.commit()

def get_user_queries(user_id):
    c.execute("SELECT query, result FROM queries WHERE user_id=?", (user_id,))
    return c.fetchall()

def save_chat_message(user_id, message, response):
    c.execute("INSERT INTO chat_history (user_id, message, response) VALUES (?, ?, ?)", (user_id, message, response))
    conn.commit()

def get_chat_history(user_id):
    c.execute("SELECT message, response FROM chat_history WHERE user_id=?", (user_id,))
    return c.fetchall()

@st.cache_data
def query_gemini(prompt):
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    params = {"key": GEMINI_API_KEY}
    
    response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data)
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def summarize_text(text):
    prompt = f"Summarize the following text, highlighting key points and arguments:\n\n{text}"
    return query_gemini(prompt)

def analyze_citations(text):
    prompt = f"Analyze the citations in the following text and identify influential sources:\n\n{text}"
    return query_gemini(prompt)

def identify_topics(text):
    try:
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform([text])
        
        if doc_term_matrix.shape[1] < 5:
            return "The text is too short or doesn't contain enough unique words for topic modeling. Please provide a longer text."
        
        lda = LatentDirichletAllocation(n_components=min(5, doc_term_matrix.shape[1]), random_state=42)
        lda.fit(doc_term_matrix)
        
        topics = []
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        
        return "\n".join(topics)
    except ValueError as e:
        return f"Error in topic modeling: {str(e)}. Please provide a longer or more diverse text."

def create_knowledge_graph(text):
    doc = nlp(text)
    G = nx.Graph()
    
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ('nsubj', 'dobj', 'pobj'):
                G.add_edge(token.head.text, token.text)
    
    return G

def visualize_knowledge_graph(G):
    if len(G.nodes()) == 0:
        return None
    
    net = Network(notebook=True, width="100%", height="600px", bgcolor="#ffffff", font_color="black")
    for node in G.nodes():
        net.add_node(node, label=node, title=node)
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    
    net.toggle_physics(True)
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

def generate_word_cloud(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word.lower() for word in word_tokens if word.isalnum() and word.lower() not in stop_words]
    
    if not filtered_text:
        return None
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_text))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Score"},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, 0.5], 'color': "gray"},
                {'range': [0.5, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_score
            }
        }
    ))
    
    return sentiment, sentiment_score, sentiment_subjectivity, fig

def generate_quiz(text):
    prompt = f"Generate a quiz with 5 multiple-choice questions based on the following text:\n\n{text}"
    return query_gemini(prompt)

def named_entity_recognition(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def visualize_named_entities(entities):
    if not entities:
        return None
    
    entity_counts = Counter(entity[1] for entity in entities)
    fig = px.bar(x=list(entity_counts.keys()), y=list(entity_counts.values()),
                 labels={'x': 'Entity Type', 'y': 'Count'},
                 title='Named Entity Recognition Results')
    return fig

def chat_with_ai(message, context=""):
    prompt = f"Context: {context}\nHuman: {message}\nAI:"
    response = query_gemini(prompt)
    return response

def extract_keywords(text):
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=10, features=None)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def generate_research_questions(text):
    prompt = f"Generate 5 research questions based on the following text:\n\n{text}"
    return query_gemini(prompt)

def suggest_research_methodologies(text):
    prompt = f"Suggest appropriate research methodologies for studying the topics in the following text:\n\n{text}"
    return query_gemini(prompt)

def literature_review_suggestions(text):
    prompt = f"Provide suggestions for conducting a literature review on the topics mentioned in the following text:\n\n{text}"
    return query_gemini(prompt)

import re
import random

# A list of some common countries (you can expand this list)
COUNTRIES = [
    "United States", "China", "India", "Brazil", "Russia", "Japan", "Germany", 
    "United Kingdom", "France", "Italy", "Canada", "Australia", "Spain", "Mexico", 
    "Indonesia", "Netherlands", "Saudi Arabia", "Turkey", "Switzerland", "Sweden"
]

def create_country_map(text):
    countries = set()
    for country in COUNTRIES:
        if re.search(r'\b' + re.escape(country) + r'\b', text, re.IGNORECASE):
            countries.add(country)
    
    if not countries:
        return None
    
    m = folium.Map(zoom_start=2)
    for country in countries:
        folium.Marker(
            location=random_location(),
            popup=country,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    return m

def random_location():
    return [random.uniform(-90, 90), random.uniform(-180, 180)]

def main():
    st.title("AI-Powered Research Assistant")

    # User Authentication
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    if st.session_state.user_id is None:
        st.sidebar.title("Login / Register")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Login"):
            user = get_user(username)
            if user and check_password(password, user[2]):
                st.session_state.user_id = user[0]
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")
        if col2.button("Register"):
            if username and password:
                try:
                    create_user(username, password)
                    st.success("User registered successfully!")
                except sqlite3.IntegrityError:
                    st.error("Username already exists")
            else:
                st.error("Please enter both username and password")
    else:
        st.sidebar.title("Logged In")
        if st.sidebar.button("Logout"):
            st.session_state.user_id = None
            st.rerun()

    if st.session_state.user_id:
        # Main Application
        tabs = st.tabs(["Research Tools", "AI Chat Assistant", "Advanced Analysis", "Research History"])
        
        with tabs[0]:
            st.header("Research Tools")
            tool = st.selectbox("Choose a tool", 
                ["Semantic Search", "Text Summarization", "Citation Analysis", "Topic Modeling", 
                 "Knowledge Graph", "Word Cloud", "Sentiment Analysis", "Quiz Generator",
                 "Named Entity Recognition", "Keyword Extraction", "Research Questions Generator"])

            if tool == "Semantic Search":
                query = st.text_input("Enter your research question:")
                if query:
                    with st.spinner("Searching..."):
                        result = query_gemini(f"Provide a detailed answer to the following research question: {query}")
                    st.write(result)
                    save_query(st.session_state.user_id, query, result)

            elif tool == "Text Summarization":
                text = st.text_area("Enter the text to summarize:")
                if st.button("Summarize"):
                    with st.spinner("Summarizing..."):
                        summary = summarize_text(text)
                    st.write(summary)

            elif tool == "Citation Analysis":
                text = st.text_area("Enter text with citations:")
                if st.button("Analyze Citations"):
                    with st.spinner("Analyzing citations..."):
                        analysis = analyze_citations(text)
                    st.write(analysis)

            elif tool == "Topic Modeling":
                text = st.text_area("Enter text for topic modeling:")
                if st.button("Identify Topics"):
                    with st.spinner("Identifying topics..."):
                        topics = identify_topics(text)
                    st.write(topics)

            elif tool == "Knowledge Graph":
                text = st.text_area("Enter text to create a knowledge graph:")
                if st.button("Generate Knowledge Graph"):
                    with st.spinner("Generating knowledge graph..."):
                        G = create_knowledge_graph(text)
                        if G:
                            html = visualize_knowledge_graph(G)
                            if html:
                                st.components.v1.html(html, height=600)
                            else:
                                st.write("Not enough data to create a knowledge graph.")
                        else:
                            st.write("Failed to create knowledge graph. Please try with a different text.")

            elif tool == "Word Cloud":
                text = st.text_area("Enter text to generate a word cloud:")
                if st.button("Generate Word Cloud"):
                    with st.spinner("Generating word cloud..."):
                        fig = generate_word_cloud(text)
                        if fig:
                            st.pyplot(fig)
                        else:
                            st.write("Not enough data to create a word cloud.")

            elif tool == "Sentiment Analysis":
                text = st.text_area("Enter text for sentiment analysis:")
                if st.button("Analyze Sentiment"):
                    with st.spinner("Analyzing sentiment..."):
                        sentiment, score, subjectivity, fig = sentiment_analysis(text)
                        st.write(f"Sentiment: {sentiment}")
                        st.write(f"Subjectivity: {subjectivity:.2f}")
                        st.plotly_chart(fig)

            elif tool == "Quiz Generator":
                text = st.text_area("Enter text to generate a quiz:")
                if st.button("Generate Quiz"):
                    with st.spinner("Generating quiz..."):
                        quiz = generate_quiz(text)
                    st.write(quiz)

            elif tool == "Named Entity Recognition":
                text = st.text_area("Enter text for named entity recognition:")
                if st.button("Recognize Entities"):
                    with st.spinner("Recognizing entities..."):
                        entities = named_entity_recognition(text)
                        if entities:
                            st.write("Recognized Entities:")
                            entity_df = pd.DataFrame(entities, columns=["Entity", "Type"])
                            st.dataframe(entity_df)
                            
                            fig = visualize_named_entities(entities)
                            if fig:
                                st.plotly_chart(fig)
                        else:
                            st.write("No entities recognized in the given text.")

            elif tool == "Keyword Extraction":
                text = st.text_area("Enter text for keyword extraction:")
                if st.button("Extract Keywords"):
                    with st.spinner("Extracting keywords..."):
                        keywords = extract_keywords(text)
                        st.write("Extracted Keywords:")
                        st.write(", ".join(keywords))

            elif tool == "Research Questions Generator":
                text = st.text_area("Enter text to generate research questions:")
                if st.button("Generate Research Questions"):
                    with st.spinner("Generating research questions..."):
                        questions = generate_research_questions(text)
                    st.write(questions)

        with tabs[1]:
            st.header("AI Chat Assistant")
            st.write("Chat with the AI assistant about your research or any questions you have.")
            
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            for message, response in st.session_state.chat_history:
                st.text_area("You:", value=message, height=100, disabled=True)
                st.text_area("AI:", value=response, height=100, disabled=True)

            user_input = st.text_input("Your message:")
            if st.button("Send"):
                if user_input:
                    with st.spinner("AI is thinking..."):
                        context = "\n".join([f"Human: {m}\nAI: {r}" for m, r in st.session_state.chat_history[-5:]])
                        ai_response = chat_with_ai(user_input, context)
                    st.session_state.chat_history.append((user_input, ai_response))
                    save_chat_message(st.session_state.user_id, user_input, ai_response)
                    st.rerun()

        with tabs[2]:
            st.header("Advanced Analysis")
            advanced_tool = st.selectbox("Choose an advanced tool", 
                ["Research Methodology Suggestions", "Literature Review Helper", "Geographic Analysis"])

            if advanced_tool == "Research Methodology Suggestions":
                text = st.text_area("Enter your research topic or abstract:")
                if st.button("Suggest Methodologies"):
                    with st.spinner("Suggesting methodologies..."):
                        suggestions = suggest_research_methodologies(text)
                    st.write(suggestions)

            elif advanced_tool == "Literature Review Helper":
                text = st.text_area("Enter your research topic or key concepts:")
                if st.button("Get Literature Review Suggestions"):
                    with st.spinner("Generating suggestions..."):
                        suggestions = literature_review_suggestions(text)
                    st.write(suggestions)

            elif advanced_tool == "Geographic Analysis":
                text = st.text_area("Enter text containing country names:")
                if st.button("Generate Country Map"):
                    with st.spinner("Creating map..."):
                        m = create_country_map(text)
                        if m:
                            folium_static(m)
                        else:
                            st.write("No countries detected in the text.")

        with tabs[3]:
            st.header("Research History")
            history_type = st.radio("Select history type:", ["Queries", "Chat"])
            
            if history_type == "Queries":
                queries = get_user_queries(st.session_state.user_id)
                if queries:
                    for i, (query, result) in enumerate(queries, 1):
                        with st.expander(f"Query {i}: {query[:50]}..."):
                            st.write("Query:")
                            st.write(query)
                            st.write("Result:")
                            st.write(result)
                else:
                    st.write("No research history found. Start by using the Research Tools!")
            
            elif history_type == "Chat":
                chat_history = get_chat_history(st.session_state.user_id)
                if chat_history:
                    for i, (message, response) in enumerate(chat_history, 1):
                        with st.expander(f"Chat {i}: {message[:50]}..."):
                            st.write("You:")
                            st.write(message)
                            st.write("AI:")
                            st.write(response)
                else:
                    st.write("No chat history found. Start chatting with the AI assistant!")

    # Add this at the end of the main function
    st.sidebar.markdown("---")
    st.sidebar.info("AI Research Assistant v3.0")
    st.sidebar.text("© 2024 DeepMost Innovations")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Close the database connection when the script ends
        conn.close()
