from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Crop Chat AI - Agentic Farming Assistant", page_icon="🌾", layout="wide")
st.title("🌱 Smart Crop Advisor - Agentic AI for Farmers")
st.markdown("Ask questions about your crop health, irrigation, soil, or fertilizer use and let AI agents guide you.")

# Sidebar Inputs
with st.sidebar:
    st.header("🛠️ Configuration")

    crop_name = st.text_input("Crop Name", placeholder="e.g., Wheat, Rice, Tomato")
    location = st.text_input("Location", placeholder="e.g., Bihar, Punjab")
    symptoms = st.text_area("Describe Crop Issue", placeholder="e.g., Yellow leaves with brown spots...")
    soil_info = st.text_area("Soil Details (Optional)", placeholder="e.g., Sandy loam, pH 6.5...")

    st.markdown("### 🔧 LLM Settings")
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)

    st.markdown("### 🌐 Search Tool Settings")
    n = st.slider("Number of Search Results", min_value=1, max_value=20, value=10, step=1)

    submit_btn = st.button("🧠 Get Smart Advice", type="primary")

# Show Explanation
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This AI-powered agentic system simulates a conversation with expert crop advisors.
    - **Crop Doctor**: Diagnoses plant problems and gives treatment.
    - **Weather Advisor**: Recommends irrigation based on weather.
    - **Soil Expert**: Gives suggestions based on soil condition.
    All agents collaborate using the CrewAI framework to give accurate guidance.
    """)

# Core Logic to Generate Smart Advice
def get_crop_advice(crop_name, location, symptoms, soil_info):
    llm = LLM(model="gpt-3.5-turbo", temperature=temperature)
    search_tool = SerperDevTool(n=n)

    # Agents
    crop_doctor = Agent(
        role="Crop Doctor",
        goal="Diagnose crop problems and provide treatment suggestions",
        backstory="You are an agronomist specializing in identifying and solving crop diseases and pest attacks.",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
        llm=llm
    )

    weather_advisor = Agent(
        role="Weather & Irrigation Expert",
        goal="Suggest irrigation based on real-time weather data",
        backstory="You are an agri-climate advisor helping farmers optimize water usage using weather insights.",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
        llm=llm
    )

    soil_expert = Agent(
        role="Soil Expert",
        goal="Analyze soil information and give fertilizer/soil health suggestions",
        backstory="You specialize in soil fertility and know how to improve yield through the right nutrients.",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
        llm=llm
    )

    # Tasks
    diagnosis_task = Task(
        description=f"Diagnose crop health based on the following symptoms: {symptoms} for crop: {crop_name}",
        expected_output="A diagnosis of the crop issue and treatment steps.",
        agent=crop_doctor
    )

    weather_task = Task(
        description=f"Provide weather forecast for {location} and suggest irrigation strategy for {crop_name}.",
        expected_output="Accurate irrigation advice based on forecasted weather.",
        agent=weather_advisor
    )

    soil_task = Task(
        description=f"Analyze the following soil data: {soil_info} and suggest appropriate fertilizers and soil care methods.",
        expected_output="Fertilizer suggestions and soil improvement plan.",
        agent=soil_expert
    )

    # Assemble Crew
    crew = Crew(
        agents=[crop_doctor, weather_advisor, soil_expert],
        tasks=[diagnosis_task, weather_task, soil_task],
        verbose=True
    )

    return crew.kickoff(inputs={"crop": crop_name, "location": location, "symptoms": symptoms, "soil": soil_info})

# Trigger on Button Click
if submit_btn:
    if not crop_name or not symptoms or not location:
        st.warning("Please fill out Crop Name, Location, and Symptoms to continue.")
    else:
        with st.spinner("🧠 Thinking... the agents are working together..."):
            try:
                result = get_crop_advice(crop_name, location, symptoms, soil_info)
                st.markdown("## 📝 Smart Crop Advice")
                st.markdown(result)

                st.download_button(
                    label="📥 Download Advice",
                    data=result.raw,
                    file_name=f"{crop_name.lower()}_crop_advice.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️  by [Ravi Kumar](https://imravi.me) | Powered by CrewAI + Streamlit")
