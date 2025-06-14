import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableMap
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Paths
CHROMA_DB_PATH = "trainedData/chroma_workout_db"

# Embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="deepseek-r1-distill-llama-70b")

# Prompts
system_prompt = (
    """
    You are an expert fitness coach and certified personal trainer. Your task is to create a safe, effective, and optimized workout plan tailored to a user's specific profile. 
    Below is the user's profile including key details. Based on this, generate a complete and actionable weekly workout plan. The plan should consider their fitness level, workout frequency, available equipment, primary goal, and other relevant parameters.
    ---
    User Profile

    1. Fitness Level:
    - This defines how experienced the user is. Adapt intensity and complexity accordingly.

    2. Workout Frequency:
    - Number of days per week the user can commit to. Spread sessions evenly.

    3. Available Equipment:
    - Specify whether the user has no equipment, minimal equipment, a home gym, or full gym access.

    4. Primary Goal:
    - Choose from: Lose Weight, Build Muscle, Increase Strength, Improve Endurance, or General Fitness.
    - Design the plan to maximize progress toward this goal.

    5. Workout Type Preference:
    - Choose from: Strength Training, HIIT, Functional Fitness, Bodyweight, or Mixed.

    6. Specific Equipment Details:
    - List exact items available (e.g., dumbbells 5–30 lbs, resistance bands, pull-up bar). Use them appropriately in exercises.

    7. Preferred Session Duration:
    - Each session should fit within this time range.

    8. Focus Areas:
    - Emphasize body parts or systems (e.g., arms, legs, core) if specified.

    9. Health Limitations (Optional):
    - Mention any injuries or medical conditions. Avoid exercises that may cause strain or discomfort and provide alternatives.
    ---
    Instructions for the Plan Output
    - Start with a short introduction explaining the weekly plan strategy and what to expect.
    - Distribute workouts across the given number of days.
    - For each workout day, list:
    - Focus area(s)
    - Workout type (e.g., circuit, split routine, full-body)
    - Exercises (3–6 depending on duration and complexity)
    - Sets, reps, rest time, and tempo where applicable.
    - Adjust intensity and volume according to **fitness level** and **goal**.
    - Recommend warm-up and cooldown routines.
    - Mention weekly progressions if applicable (e.g., increase reps or weight).
    - If user has limitations, offer **safe alternatives** or restrictions to avoid.
    - Be clear, structured, and easy to follow. Do not assume gym jargon knowledge if user is a beginner.
    ---
    Note: If the information is incomplete, make reasonable and safe assumptions based on best practices. Always prioritize safety, especially when health conditions are mentioned.

    Now generate a complete weekly workout plan for this user profile.
    """
    "\n\nContext:\n{context}\n"
)

contextualize_q_system_prompt = (
    "You are tasked with refining the user profile to a query format that can be matched with existing training data.\n"
    "Given the profile fields and chat history, convert them into a single descriptive input."
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    ("human", "{input}")
])

# Load vector database
vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vector_store.as_retriever()

def get_context_from_metadata(query: str):
    """Get context from documents' metadata output"""
    try:
        # Retrieve top 2 relevant documents
        docs = retriever.invoke(query)[:3]
        
        if not docs:
            st.warning("⚠️ No similar context found in vector database.")
            return "No similar context found."
        
        # Display retrieved documents for debugging
        st.subheader("🔍 Top 3 Retrieved Documents")
        contexts = []
        
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Document #{i}**")
            st.markdown(f"**Content Preview:** {doc.page_content}...")
            st.markdown(f"**Metadata:** {doc.metadata.get("output", "No metadata found")}")
            
            # Extract output from metadata
            output = doc.metadata.get("output", "No output metadata found.")
            contexts.append(f"Example {i}:\n{output}")
            
            st.markdown("---")
        
        # Combine contexts
        final_context = "\n\n".join(contexts)
        return final_context
        
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return "Error retrieving context from database."

def debug_and_format_prompt(inputs: dict):
    """Debug function to show the formatted prompt"""
    try:
        # Get context first
        context = get_context_from_metadata(inputs["input"])
        
        # Update inputs with context
        inputs_with_context = {
            "input": inputs["input"],
            "context": context
        }
        
        # Format the prompt
        formatted_messages = qa_prompt.format_messages(**inputs_with_context)
        
        # Display the formatted prompt
        st.subheader("Final Prompt Sent to LLM")
        for i, message in enumerate(formatted_messages):
            st.markdown(f"**Message {i+1} ({message.type}):**")
            st.code(message.content, language="text")
            st.markdown("---")
        
        return formatted_messages
        
    except Exception as e:
        st.error(f"Error in debug_and_format_prompt: {str(e)}")
        return qa_prompt.format_messages(input=inputs["input"], context="Error formatting prompt")

def generate_workout_plan(user_profile: str):
    """Generate workout plan using the LLM"""
    try:
        # Debug and format prompt
        formatted_messages = debug_and_format_prompt({"input": user_profile})
        
        # Get response from LLM
        response = llm.invoke(formatted_messages)
        
        return response.content
        
    except Exception as e:
        st.error(f"Error generating workout plan: {str(e)}")
        return "Error generating workout plan. Please try again."

# Streamlit UI
st.set_page_config(page_title="Fitness Planner", layout="wide")
st.title("🏋️ Personalized Fitness Plan Generator")

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

if "last_query" not in st.session_state:
    st.session_state.last_query = None

with st.sidebar:
    st.header("User Profile")

    fitness_level = st.selectbox("1. Fitness Level", ["", "Beginner", "Intermediate", "Advanced"])
    frequency = st.selectbox("2. Workout Frequency", ["", "2-3 days per week", "3-4 days per week", "4-5 days per week", "5-6 days per week"])
    equipment = st.selectbox("3. Available Equipment", ["", "None", "Minimal (Dumbbells/Resistance bands)", "Home Gym Setup", "Full Gym Access"])
    goal = st.selectbox("4. Primary Goal", ["", "Lose Weight", "Build Muscle", "Increase Strength", "Improve Endurance", "General Fitness"])
    workout_type = st.selectbox("5. Workout Type", ["", "Strength Training", "High Intensity Interval Training", "Functional fitness", "Body Weight", "Mixed"])
    specific_equipment = st.text_input("6. Specific Equipment Details")
    session_duration = st.selectbox("7. Session Duration", ["", "15-30 minutes", "30-45 minutes", "45-60 minutes", "60+ minutes"])
    focus_areas = st.text_input("8. Focus Areas (e.g., arms, core, legs)")
    health_limitations = st.text_input("9. Health Limitations (Optional)")

    generate = st.button("Generate Plan")

# Validate inputs
required_fields = [fitness_level, frequency, equipment, goal, workout_type, specific_equipment, session_duration, focus_areas]

def remove_think_tags(text):
    pattern = r'<think>.*?</think>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text.strip()

if generate:
    if "" in required_fields:
        st.error("⚠️ Please fill in all fields except Health Limitations before generating the plan.")
    else:
        user_profile = (
            f"Fitness Level: {fitness_level}; Workout Frequency: {frequency}; Available Equipment: {equipment}; "
            f"Primary Goal: {goal}; Workout Type: {workout_type}; Specific Equipment Details: {specific_equipment}; "
            f"Session Duration: {session_duration}; Focus Areas: {focus_areas}; Health Limitations: {health_limitations or 'None'}."
        )

        # Show profile being processed
        st.subheader("📋 User Profile Summary")
        st.info(user_profile)

        # Generate response
        with st.spinner("Generating your personalized plan..."):
            try:
                # Generate the workout plan
                response_text = generate_workout_plan(user_profile)
                
                # Display response
                st.success("✅ Plan Generated Successfully!")
                st.subheader("🧠 Personalized Workout Plan")
                st.markdown(remove_think_tags(response_text))
                
                # Update chat history
                st.session_state.chat_history.add_message(HumanMessage(content=user_profile))
                st.session_state.chat_history.add_message(AIMessage(content=response_text))
                
            except Exception as e:
                st.error(f"Error generating plan: {str(e)}")

# Optional: Display chat history
if st.session_state.chat_history.messages:
    with st.expander("📚 Chat History", expanded=False):
        for message in st.session_state.chat_history.messages:
            if isinstance(message, HumanMessage):
                st.markdown(f"**User:** {message.content}")
            elif isinstance(message, AIMessage):
                st.markdown(f"**Assistant:** {message.content}...")
            st.markdown("---")