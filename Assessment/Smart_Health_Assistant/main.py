import streamlit as st
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import google.generativeai as genai
import os
from dotenv import load_dotenv

# === Streamlit UI ===
st.set_page_config(page_title="Smart Health Assistant", layout="wide")

# Modern header with icon and subtitle
st.markdown("""
    <div style='display: flex; flex-direction: column; align-items: center; margin-bottom: 2rem;'>
        <h1 style='font-size: 2.2rem; margin-bottom: 0.2rem;'>🤖 Smart Health Assistant</h1>
        <span style='font-size: 1.1rem; opacity: 0.85;'>Your personalized AI-powered health, diet, and fitness planner</span>
    </div>
""", unsafe_allow_html=True)

load_dotenv()

default_api_key = os.getenv("GEMINI_API_KEY", "")
if not default_api_key:
    st.error("API key not found in environment. Please set GEMINI_API_KEY in your .env file.")

# Modern instructions card


# Centered card for the form

st.markdown("""
<div style='display: flex; justify-content: center; margin-bottom: 1.5rem;'>
    <div style='border-radius: 18px; box-shadow: 0 2px 12px rgba(44, 62, 80, 0.07); padding: 2.5rem 2.5rem 1.5rem 2.5rem; min-width: 350px; max-width: 480px; width: 100%; background: transparent;'>
""", unsafe_allow_html=True)

# === Session State ===
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "final_plan" not in st.session_state:
    st.session_state.final_plan = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Utility: Gemini Config Wrapper ===
def get_gemini_config(api_key: str, model: str = "gemini-1.5-flash"):
    return [{
        "model": model,
        "api_key": api_key,
        "api_type": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta"
    }]

# === BMI Tool ===
def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)

# === Health Form ===
with st.form("health_form"):
    st.markdown("<h3 style='margin-bottom: 1.2rem;'>📝 Enter Your Health Details</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1, format="%.1f")
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        dietary_preference = st.selectbox("Dietary Preference", ["Veg", "Non-Veg", "Vegan"])
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Generate Health Plan")

st.markdown("</div></div>", unsafe_allow_html=True)  # Close card and flex

# === Agent Initialization ===
def init_agents(api_key):
    genai.configure(api_key=api_key)
    config_list = get_gemini_config(api_key)

    bmi_agent = AssistantAgent(
        name="BMI_Agent",
        llm_config={"config_list": config_list, "cache_seed": None},
        system_message="""You are a BMI specialist. Analyze BMI results and:
        1. Calculate BMI from weight (kg) and height (cm)
        2. Categorize (underweight, normal, overweight, obese)
        3. Provide health recommendations
        Always include the exact BMI value in your response."""
    )

    diet_agent = AssistantAgent(
        name="Diet_Planner",
        llm_config={"config_list": config_list, "cache_seed": None},
        system_message=f"""You are a nutritionist. Create meal plans based on:
        1. BMI analysis from BMI_Agent
        2. Dietary preference ({dietary_preference})
        Include breakfast, lunch, dinner, and snacks with portions."""
    )

    workout_agent = AssistantAgent(
        name="Workout_Scheduler",
        llm_config={"config_list": config_list, "cache_seed": None},
        system_message=f"""You are a fitness trainer. Create weekly workout plans based on:
        1. Age ({age}) and gender ({gender})
        2. BMI recommendations
        3. Meal plan from Diet_Planner
        Include cardio, strength training with duration and intensity."""
    )

    user_proxy = UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        llm_config={"config_list": config_list, "cache_seed": None},
        system_message="Collects and shares user data with other agents."
    )

    user_proxy.register_function(function_map={"calculate_bmi": calculate_bmi})

    return user_proxy, bmi_agent, diet_agent, workout_agent, config_list

# === Submit Handler ===
if submit_btn and default_api_key:
    try:
        user_proxy, bmi_agent, diet_agent, workout_agent, config_list = init_agents(default_api_key)

        groupchat = GroupChat(
            agents=[user_proxy, bmi_agent, diet_agent, workout_agent],
            messages=[],
            max_round=6,
            speaker_selection_method="round_robin"
        )

        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config={"config_list": config_list, "cache_seed": None}
        )

        initial_message = f"""
        User Health Profile:
        - Basic Information:
          • Weight: {weight} kg
          • Height: {height} cm
          • Age: {age}
          • Gender: {gender}
        - Preferences:
          • Dietary Preference: {dietary_preference}

        Please proceed with the health assessment in this sequence:
        1. Calculate BMI using the 'calculate_bmi' function with weight={weight} and height={height}
        2. Analyze BMI and provide recommendations
        3. Create a meal plan based on BMI analysis and dietary preference
        4. Develop a workout schedule based on age, gender, and meal plan
        """

        with st.spinner("Generating your personalized health plan..."):
            user_proxy.initiate_chat(
                manager,
                message=initial_message,
                clear_history=True
            )

            st.session_state.conversation = []
            for msg in groupchat.messages:
                if msg['role'] != 'system' and msg['content'].strip():
                    st.session_state.conversation.append((msg['name'], msg['content']))
                    if msg['name'] == "Workout_Scheduler":
                        st.session_state.final_plan = msg['content']

        st.success("Health plan generated successfully! ✅")

    except Exception as e:
        st.markdown(f"<div style='color:#b94a48; background:#f8d7da; border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.5rem;'><b>Error occurred:</b> {str(e)}</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#555; background:#e2e3e5; border-radius:8px; padding:0.8rem 1rem;'><b>Please ensure:</b> 1) Valid API key in .env 2) Stable internet connection 3) Correct input values</div>", unsafe_allow_html=True)

# === Results Display ===
if st.session_state.conversation:
    st.markdown("---")
    st.markdown("### Health Plan Generation Process")

    for agent, message in st.session_state.conversation:
        with st.expander(f"{agent} says:"):
            st.markdown(message)

    st.markdown("---")
    st.markdown("## 🌟 Your Complete Health Plan")

    if st.session_state.final_plan:
        st.markdown(st.session_state.final_plan)
        st.download_button(
            label="⬇️ Download Health Plan",
            data=st.session_state.final_plan,
            file_name="personalized_health_plan.txt",
            mime="text/plain"
        )
    else:
        st.warning("Workout schedule not generated. Please try again.")

elif not submit_btn:
    st.markdown("---")
    st.info("""
    **Instructions:**
    1. Fill in your health details
    2. Click **Generate Health Plan**
    3. View your personalized recommendations
    """)