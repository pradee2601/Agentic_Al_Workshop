import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import ast
from crewai_tools import CodeInterpreterTool

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Custom Python Analyzer (No ONNX)
def analyze_python_code(code: str) -> str:
    """Static analysis without executing code."""
    try:
        # 1. Check syntax via AST
        tree = ast.parse(code)
        
        # 2. Basic checks
        issues = []
        
        # Check for print statements (not recommended in production)
        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print' 
               for node in ast.walk(tree)):
            issues.append("⚠️ Found `print()` - Use logging in production.")

        # Check for broad exceptions
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append("⚠️ Found bare `except:` - Specify exception types.")

        # 3. Return results
        if issues:
            return "Found issues:\n" + "\n".join(issues)
        return "✅ No syntax errors found. Code looks good!"
    
    except SyntaxError as e:
        return f"❌ Syntax Error: {e.msg} (Line {e.lineno})"

# Initialize LLM (Groq or Gemini)
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY") , temperature=0.1)  # or ChatGoogleGenerativeAI(model="gemini-pro")
llm = LLM(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini/gemini-2.5-flash"  # Must include provider prefix
)
# ===== Agents =====
code_analyzer = Agent(
    role="Python Static Analyzer",
    goal="Find issues in Python code WITHOUT executing it",
    backstory="Expert in static code analysis using AST parsing.",
    llm=llm,
    verbose=True,
    tools=[CodeInterpreterTool()]
)

code_corrector = Agent(
    role="Python Code Fixer",
    goal="Fix issues while keeping original functionality",
    backstory="Specializes in clean, PEP 8 compliant fixes.",
    llm=llm,
    verbose=True
)

manager = Agent(
    role="Code Review Manager",
    goal="Ensure smooth analysis & correction",
    backstory="Coordinates the review process.",
    llm=llm,
    verbose=True
)

# ===== Streamlit UI =====
st.set_page_config(page_title="Python Code Reviewer", page_icon="🧑‍💻", layout="centered")

# Minimal, clean header
st.markdown("""
# 🔍 Python Code Reviewer (No ONNX)
##### Paste your Python code below and get instant static analysis and auto-fixes, powered by AI.
---
""")

with st.container():
    code_input = st.text_area("Paste Python code:", height=220, key="code_input")
    analyze_btn = st.button("Analyze & Fix", use_container_width=True)

if analyze_btn:
    if not code_input.strip():
        st.warning("Please enter Python code.")
    else:
        with st.spinner("Analyzing..."):
            # Task 1: Static Analysis
            analysis_task = Task(
                description=f"Analyze this code:\n```python\n{code_input}\n```",
                agent=code_analyzer,
                expected_output="List of static analysis issues."
            )

            # Task 2: Fix Code
            correction_task = Task(
                description="Fix all issues found.",
                agent=code_corrector,
                expected_output="Corrected Python code with explanations.",
                context=[analysis_task]
            )

            # Run CrewAI
            crew = Crew(
                agents=[code_analyzer, code_corrector, manager],
                tasks=[analysis_task, correction_task],
                verbose=True,
                process=Process.sequential,
                planning=True
            )
            
            results = crew.kickoff()
            # If results is a list or tuple, unpack; else, treat as string
            if isinstance(results, (list, tuple)) and len(results) == 2:
                analysis_result, correction_result = results
            elif isinstance(results, dict):
                analysis_result = results.get('analysis', '')
                correction_result = results.get('correction', '')
            else:
                analysis_result = "(Could not extract analysis result)"
                correction_result = results

            # Display Results
            with st.expander("🧐 Analysis Result", expanded=True):
                st.write(analysis_result)
            with st.expander("🔧 Fixed Code", expanded=True):
                st.code(correction_result, language="python")

st.markdown("---")
st.markdown("<div style='text-align:center; color: #888; font-size: 0.95rem;'>Made with ❤️ using Streamlit & CrewAI</div>", unsafe_allow_html=True)