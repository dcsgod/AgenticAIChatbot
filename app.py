from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

# Topic for the AI Crew
topic = "Crop Health and Agricultural Advice"

# Initialize the LLM (You can change model as needed)
llm = LLM(model="gpt-4")

# Tool for external search
search_tool = SerperDevTool(n=10)

# Agent 1 - Crop Doctor
crop_doctor = Agent(
    role="Crop Doctor",
    goal="Analyze crop symptoms and provide possible causes and solutions",
    backstory="You are an experienced Agronomist who has spent years diagnosing crop issues such as diseases, pests, and nutrient deficiencies. You help farmers take corrective actions quickly.",
    allow_delegation=False,
    verbose=True,
    tools=[search_tool],
    llm=llm
)

# Agent 2 - Weather & Water Advisor
weather_expert = Agent(
    role="Weather & Irrigation Expert",
    goal="Provide local weather forecasts and optimal irrigation advice",
    backstory="You are a climate scientist specializing in agriculture. You understand how weather impacts different crops and help farmers adjust their watering strategies.",
    allow_delegation=False,
    verbose=True,
    tools=[search_tool],
    llm=llm
)

# Agent 3 - Soil Specialist
soil_specialist = Agent(
    role="Soil & Fertilizer Advisor",
    goal="Recommend soil treatments and fertilizer plans",
    backstory="You are a soil scientist who helps farmers choose the right fertilizers and improve soil health for better yields.",
    allow_delegation=False,
    verbose=True,
    tools=[search_tool],
    llm=llm
)

# Task 1 - Diagnose Crop Issue
crop_diagnosis_task = Task(
    description="Diagnose the crop problem based on symptoms shared by the user and recommend appropriate treatment.",
    expected_output="A detailed explanation of the crop issue and step-by-step solution.",
    agent=crop_doctor
)

# Task 2 - Give Weather & Watering Advice
weather_task = Task(
    description="Analyze the user's location and recommend proper watering based on weather trends.",
    expected_output="Weather forecast and smart irrigation advice tailored to the crop.",
    agent=weather_expert
)

# Task 3 - Analyze Soil and Suggest Fertilizer
soil_task = Task(
    description="Analyze user-provided soil information and recommend fertilizer and soil improvements.",
    expected_output="Best fertilizer choices and tips for enhancing soil quality.",
    agent=soil_specialist
)

# Create the Crew
crew = Crew(
    agents=[crop_doctor, weather_expert, soil_specialist],
    tasks=[crop_diagnosis_task, weather_task, soil_task],
    verbose=True
)

# Run the Crew
result = crew.kickoff(inputs={"topic": topic})

print(result)
