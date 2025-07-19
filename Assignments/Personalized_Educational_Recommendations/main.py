import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew
import google.generativeai as genai
import requests
import re
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Gemini LLM for CrewAI
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

# Pydantic Models
class LearningMaterial(BaseModel):
    title: str
    link: str
    type: str  # 'video', 'article', or 'exercise'

class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    answer: str

class ProjectIdea(BaseModel):
    title: str
    description: str
    level: str

# Helper Functions
def search_learning_materials(topic: str) -> dict:
    """Search for learning materials on a given topic."""
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY}
        
        # Search for videos
        video_query = f"{topic} tutorial video"
        video_results = requests.post(url, json={"q": video_query}, headers=headers).json()
        
        # Search for articles
        article_query = f"{topic} guide article"
        article_results = requests.post(url, json={"q": article_query}, headers=headers).json()
        
        # Search for exercises
        exercise_query = f"{topic} practice exercises"
        exercise_results = requests.post(url, json={"q": exercise_query}, headers=headers).json()
        
        videos = []
        articles = []
        exercises = []
        
        # Extract videos
        for v in video_results.get("organic", [])[:3]:
            videos.append(LearningMaterial(title=v['title'], link=v['link'], type='video'))
        
        # Extract articles
        for a in article_results.get("organic", [])[:3]:
            articles.append(LearningMaterial(title=a['title'], link=a['link'], type='article'))
            
        # Extract exercises
        for e in exercise_results.get("organic", [])[:3]:
            exercises.append(LearningMaterial(title=e['title'], link=e['link'], type='exercise'))
        
        return {
            "topic": topic,
            "videos": videos,
            "articles": articles,
            "exercises": exercises
        }
    except Exception as e:
        return {
            "topic": topic,
            "videos": [LearningMaterial(title=f"Error searching videos: {str(e)}", link="", type="video")],
            "articles": [LearningMaterial(title=f"Error searching articles: {str(e)}", link="", type="article")],
            "exercises": [LearningMaterial(title=f"Error searching exercises: {str(e)}", link="", type="exercise")]
        }

def generate_quiz_questions(topic: str) -> list[QuizQuestion]:
    """Generate quiz questions on a given topic."""
    try:
        prompt = f"""Create 3 multiple-choice questions about {topic}. Format each question as follows:

Question: [Your question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Answer: [Correct option letter]

Make sure the questions are clear and educational."""
        
        response = model.generate_content(prompt).text
        questions = []
        
        # Parse the response
        question_blocks = response.split("Question:")
        for block in question_blocks[1:]:  # Skip first empty element
            lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
            if len(lines) >= 6:
                question = lines[0]
                options = []
                answer_line = ""
                
                for line in lines[1:]:
                    if line.startswith(('A)', 'B)', 'C)', 'D)')):
                        options.append(line[3:].strip())
                    elif line.startswith("Answer:"):
                        answer_line = line.split(":")[-1].strip()
                
                if len(options) == 4 and answer_line:
                    # Convert answer letter to actual answer text
                    answer_index = ord(answer_line.upper()) - ord('A')
                    if 0 <= answer_index < 4:
                        questions.append(QuizQuestion(
                            question=question,
                            options=options,
                            answer=options[answer_index]
                        ))
        
        return questions[:3]
    except Exception as e:
        return [QuizQuestion(question=f"Error generating quiz: {str(e)}", options=["Error"]*4, answer="Error")]

def suggest_projects(topic: str, level: str) -> list[ProjectIdea]:
    """Generate project ideas based on topic and expertise level."""
    try:
        prompt = f"""Suggest 3 practical project ideas for someone at a {level} level learning about {topic}.
For each project, provide:
- A clear title
- A detailed description explaining what the project involves
- Why it's suitable for {level} level

Format each project as:
Project: [Title]
Description: [Detailed description]
"""
        
        response = model.generate_content(prompt).text
        projects = []
        
        # Parse the response
        project_blocks = response.split("Project:")
        for block in project_blocks[1:]:  # Skip first empty element
            lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
            
            title = lines[0] if lines else "Untitled Project"
            description = ""
            
            for line in lines[1:]:
                if line.startswith("Description:"):
                    description = line.split(":", 1)[1].strip()
                    break
            
            if description:
                projects.append(ProjectIdea(
                    title=title,
                    description=description,
                    level=level
                ))
        
        return projects[:3]
    except Exception as e:
        return [ProjectIdea(title=f"Error generating projects: {str(e)}", description="Unable to generate project suggestions", level=level)]

# Agents without tools - they will use the functions directly
learning_agent = Agent(
    role="Learning Material Curator",
    goal="Find the best learning resources for a given topic using web search",
    backstory="""You are an expert researcher with years of experience in educational content curation. 
    You excel at finding diverse learning materials including videos, articles, and practical exercises.
    You have access to web search capabilities to find current and relevant learning materials.""",
    llm=gemini_llm,
    verbose=True
)

quiz_agent = Agent(
    role="Quiz Master",
    goal="Create effective assessment quizzes for learning topics",
    backstory="""You are specialized in educational assessment and test creation. 
    You create engaging multiple-choice questions that test understanding and promote learning.
    You can generate high-quality quiz questions on any topic.""",
    llm=gemini_llm,
    verbose=True
)

project_agent = Agent(
    role="Project Mentor",
    goal="Suggest practical projects matching skill levels",
    backstory="""You are experienced in curriculum development and project-based learning. 
    You design hands-on projects that reinforce learning and build practical skills.
    You can suggest projects appropriate for different skill levels.""",
    llm=gemini_llm,
    verbose=True
)

# Tasks with detailed descriptions
def create_learning_task(topic: str):
    return Task(
        description=f"""Search for comprehensive learning materials about '{topic}'. 
        Find videos, articles, and exercises that would help someone learn this topic effectively.
        
        Use web search to find:
        1. Educational videos and tutorials
        2. Articles and guides
        3. Practice exercises and examples
        
        Return the results in a structured format with titles and links.""",
        agent=learning_agent,
        expected_output=f"""A comprehensive list of learning materials for {topic} including:
        - Videos: List of educational videos with titles and links
        - Articles: List of articles and guides with titles and links  
        - Exercises: List of practice exercises with titles and links"""
    )

def create_quiz_task(topic: str):
    return Task(
        description=f"""Create a quiz about '{topic}' with 3 multiple-choice questions. 
        Make sure the questions are educational and test important concepts.
        
        Each question should have:
        - A clear question
        - 4 multiple choice options (A, B, C, D)
        - The correct answer indicated
        
        Focus on testing understanding rather than memorization.""",
        agent=quiz_agent,
        expected_output=f"""A set of 3 quality multiple-choice questions about {topic}, each with:
        - Question text
        - 4 answer options
        - Correct answer identified"""
    )

def create_project_task(topic: str, level: str):
    return Task(
        description=f"""Suggest 3 practical project ideas about '{topic}' suitable for {level} level learners. 
        Each project should have a clear title and detailed description.
        
        Consider the {level} skill level when designing projects:
        - Beginner: Simple, guided projects with clear steps
        - Intermediate: Projects requiring some independent thinking
        - Advanced: Complex projects requiring expertise and creativity
        
        Each project should be practical and help reinforce learning.""",
        agent=project_agent,
        expected_output=f"""3 practical project ideas for {level} level learners about {topic}, each with:
        - Project title
        - Detailed description
        - Why it's suitable for {level} level"""
    )

# Execution function
def generate_learning_path_with_crew(topic: str, level: str):
    """Generate a complete learning path for the given topic and level using CrewAI sequential orchestration."""
    try:
        # Define tasks
        learning_task = Task(
            description=f"""Search for comprehensive learning materials about '{topic}'. Find videos, articles, and exercises that would help someone learn this topic effectively. Use web search to find: 1. Educational videos and tutorials 2. Articles and guides 3. Practice exercises and examples. Return the results in a structured format with titles and links.""",
            agent=learning_agent,
            expected_output=f"""A comprehensive list of learning materials for {topic} including: - Videos: List of educational videos with titles and links - Articles: List of articles and guides with titles and links  - Exercises: List of practice exercises with titles and links"""
        )
        quiz_task = Task(
            description=f"""Create a quiz about '{topic}' with 3 multiple-choice questions. Make sure the questions are educational and test important concepts. Each question should have: - A clear question - 4 multiple choice options (A, B, C, D) - The correct answer indicated. Focus on testing understanding rather than memorization.""",
            agent=quiz_agent,
            expected_output=f"""A set of 3 quality multiple-choice questions about {topic}, each with: - Question text - 4 answer options - Correct answer identified"""
        )
        project_task = Task(
            description=f"""Suggest 3 practical project ideas about '{topic}' suitable for {level} level learners. Each project should have a clear title and detailed description. Consider the {level} skill level when designing projects: - Beginner: Simple, guided projects with clear steps - Intermediate: Projects requiring some independent thinking - Advanced: Complex projects requiring expertise and creativity. Each project should be practical and help reinforce learning.""",
            agent=project_agent,
            expected_output=f"""3 practical project ideas for {level} level learners about {topic}, each with: - Project title - Detailed description - Why it's suitable for {level} level"""
        )

        # Create Crew with sequential process
        crew = Crew(
            agents=[learning_agent, quiz_agent, project_agent],
            tasks=[learning_task, quiz_task, project_task],
            process='sequential',
            verbose=True
        )

        # Run Crew
        results = crew.kickoff()

        # Parse results (since we use our own helper functions for structure)
        learning_materials = search_learning_materials(topic)
        quiz_questions = generate_quiz_questions(topic)
        project_ideas = suggest_projects(topic, level)

        return {
            "learning_materials": learning_materials,
            "quiz_questions": quiz_questions,
            "project_ideas": project_ideas,
            "raw_result": results
        }
    except Exception as e:
        st.error(f"❌ Error generating content: {str(e)}")
        return {
            "learning_materials": {},
            "quiz_questions": [],
            "project_ideas": []
        }

# Streamlit UI
def main():
    st.set_page_config(page_title="Personalized Learning Assistant", page_icon="🎓", layout="wide")

    # --- Custom CSS for modern look ---
    st.markdown(
        """
        <style>
        .main {background-color: #f8f9fa;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
        .stTabs [data-baseweb="tab"] {font-size: 1.1rem; font-weight: 600; color: #2d3748;}
        .stTabs [aria-selected="true"] {color: #2563eb; border-bottom: 2px solid #2563eb;}
        .stButton>button {background: linear-gradient(90deg, #2563eb 0%, #38bdf8 100%); color: white; font-weight: 600; border-radius: 8px; padding: 0.5rem 1.5rem;}
        .stTextInput>div>div>input, .stSelectbox>div>div>div>input {border-radius: 8px; border: 1px solid #cbd5e1;}
        .stMarkdown h3 {margin-top: 1.5rem;}
        .stMarkdown h2 {margin-top: 2rem;}
        .stMarkdown ul {margin-bottom: 1.5rem;}
        .stAlert {border-radius: 8px;}
        .footer {margin-top: 2rem; color: #888; font-size: 0.95rem; text-align: center;}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='font-size:2.5rem; margin-bottom:0.2em;'>🎓 Personalized Learning Assistant</h1>
        <p style='font-size:1.2rem; color:#555;'>Generate comprehensive learning materials, quizzes, and project ideas for any topic!</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar for input ---
    with st.sidebar:
        st.markdown("<h2 style='margin-bottom:0.5em;'>🛠️ Customize Your Path</h2>", unsafe_allow_html=True)
        topic = st.text_input("📚 Enter your learning topic:", placeholder="e.g., Machine Learning, Python, Data Science")
        level = st.selectbox("📊 Select your skill level:", ["Beginner", "Intermediate", "Advanced"])
        st.markdown("---")
        generate = st.button("🚀 Generate Learning Path", type="primary")
        st.markdown("<div style='font-size:0.95rem; color:#888;'>Powered by <b>Google Gemini AI</b> & <b>Serper API</b></div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- Main content ---
    if not os.getenv("GEMINI_API_KEY"):
        st.error("⚠️ Please set your GEMINI_API_KEY in the environment variables.")
        st.stop()
    if not os.getenv("SERPER_API_KEY"):
        st.error("⚠️ Please set your SERPER_API_KEY in the environment variables.")
        st.stop()

    if generate:
        if not topic.strip():
            st.error("Please enter a topic to learn about.")
            return
        with st.spinner("🔍 Creating your personalized learning path..."):
            result = generate_learning_path_with_crew(topic, level)
            if result:
                st.success("✅ Learning path generated successfully!")
                st.markdown("---")
                tab1, tab2, tab3 = st.tabs([
                    "📚 <b>Learning Materials</b>",
                    "📝 <b>Quiz</b>",
                    "🚀 <b>Project Ideas</b>"
                ])
                with tab1:
                    st.markdown("<h3>🎥 Videos</h3>", unsafe_allow_html=True)
                    learning_materials = result.get("learning_materials", {})
                    if learning_materials.get("videos"):
                        for video in learning_materials["videos"]:
                            st.markdown(f"<div style='margin-bottom:0.5em;'><b>{video.title}</b> <span style='color:#2563eb;'>(Video)</span><br><a href='{video.link}' target='_blank'>{video.link}</a></div>", unsafe_allow_html=True)
                    st.markdown("<h3>📄 Articles</h3>", unsafe_allow_html=True)
                    if learning_materials.get("articles"):
                        for article in learning_materials["articles"]:
                            st.markdown(f"<div style='margin-bottom:0.5em;'><b>{article.title}</b> <span style='color:#2563eb;'>(Article)</span><br><a href='{article.link}' target='_blank'>{article.link}</a></div>", unsafe_allow_html=True)
                    st.markdown("<h3>💪 Exercises</h3>", unsafe_allow_html=True)
                    if learning_materials.get("exercises"):
                        for exercise in learning_materials["exercises"]:
                            st.markdown(f"<div style='margin-bottom:0.5em;'><b>{exercise.title}</b> <span style='color:#2563eb;'>(Exercise)</span><br><a href='{exercise.link}' target='_blank'>{exercise.link}</a></div>", unsafe_allow_html=True)
                with tab2:
                    st.markdown("<h3>📝 Quiz Questions</h3>", unsafe_allow_html=True)
                    quiz_questions = result.get("quiz_questions", [])
                    if quiz_questions:
                        for i, q in enumerate(quiz_questions, 1):
                            st.markdown(f"<b>Question {i}:</b> {q.question}")
                            for j, option in enumerate(q.options, 1):
                                st.markdown(f"<span style='margin-left:1.5em;'>{chr(64+j)}) {option}</span>", unsafe_allow_html=True)
                            st.markdown(f"<span style='color:#22c55e;'><b>✅ Correct Answer:</b> {q.answer}</span>", unsafe_allow_html=True)
                            st.markdown("<hr style='margin:0.7em 0;'>", unsafe_allow_html=True)
                    else:
                        st.info("No quiz questions generated.")
                with tab3:
                    st.markdown("<h3>🚀 Project Ideas</h3>", unsafe_allow_html=True)
                    project_ideas = result.get("project_ideas", [])
                    if project_ideas:
                        for i, project in enumerate(project_ideas, 1):
                            st.markdown(f"<div style='margin-bottom:0.7em;'><b>Project {i}: {project.title}</b><br>"
                                        f"<span style='color:#555;'><b>Description:</b> {project.description}</span><br>"
                                        f"<span style='color:#2563eb;'><b>Level:</b> {project.level}</span></div>", unsafe_allow_html=True)
                            st.markdown("<hr style='margin:0.7em 0;'>", unsafe_allow_html=True)
                    else:
                        st.info("No project ideas generated.")
                if result.get("raw_result"):
                    with st.expander("🔍 View Raw AI Output"):
                        st.text(str(result["raw_result"]))

    st.markdown("---")
    st.markdown(
        """
        <div class='footer'>
            🤖 Powered by <a href='https://ai.google.dev/' target='_blank'>Google Gemini AI</a> | 🔍 Web Search via <a href='https://serper.dev/' target='_blank'>Serper API</a><br>
            💡 This tool generates learning materials, quizzes, and project ideas for any topic.<br>
            <span style='font-size:0.9em;'>Created for the Agentic AI Workshop</span>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()