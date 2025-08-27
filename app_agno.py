import streamlit as st
import fitz  # PyMuPDF for PDF
#import docx
from docx import Document
#To retry if the model is not working
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
#import openai
import google.generativeai as genai
import os
from agno.tools.duckduckgo import DuckDuckGoTools
#the finance tool to get stock price
from agno.tools.yfinance import YFinanceTools
#import google.generativeai as genai
from agno.agent import Agent
from agno.models.google import Gemini
from agno.agent import agent
from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key (for testing use dotenv or Streamlit secrets)
#openai.api_key = "YOUR_OPENAI_API_KEY"
# Set your google API key (for testing use dotenv or Streamlit secrets)
#genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#model = genai.GenerativeModel("gemini-1.5-flash")

#Web Agent to get the real time data from web
web_agent = Agent(
        name ="Web Agent",
        role = "Search the web for job opportunities",
        #model = OpenAIChat(id="gpt-4o"),
        model = Gemini(id="gemini-1.5-flash",api_key=os.getenv("GEMINI_API_KEY")),
        tools = [DuckDuckGoTools()],
        instructions ='Search for live job links matching user profile. Always include links.',
        show_tool_calls =True,
        markdown = True,
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=10),
    retry=retry_if_exception_type(Exception)
)
def run_web_agent_with_retry(prompt):
    return web_agent.run(prompt)


# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to send extracted resume to LLM
def analyze_resume_with_llm(resume_text, number, location, roles):
    try :
        number = int(number or "3")
        prompt = f"""
        You are a job recruitment assistant helping candidates to find the most suitable job alligning with their skillset.
        Analyze the following resume and extract:
        - Candidate's Name
        - Key Skills
        - Total Years of Experience
        - Suitable Job Titles
        - Preferred Industries

        Resume:
        {resume_text}
        and search in google and find top {number} of jobs from the {location} suitable for this profile from internet. 
        The candidate is mainly looking for the role of {roles} but you can also check for other suitable job titles you found from their resume.
        Give the job link list in a tabular manner.
        """
        #using web agent
        #response = web_agent.run(prompt)
        #to incorporate retry logic
        response = run_web_agent_with_retry(prompt)
        #print(response)
        #print(type(response))
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(e)
        return str(e)
    #uncomment if you are not using webagent
    #response = model.generate_content(prompt)
    #return response.text.strip()

    
    """response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()"""

# Title
st.title("AI Job Finder")

# Resume Upload
st.subheader("Upload Your Resume")
resume_file = st.file_uploader("Choose your resume (PDF or DOCX)", type=["pdf", "docx"])

# Location Preference
st.subheader("Job Preferences")
preferred_location = st.text_input("Preferred Job Location", placeholder="e.g., Dubai")

# Job Role or Keywords
preferred_roles = st.text_input("Preferred Roles or Keywords", placeholder="e.g., AI Developer, LLM, Prompt Engineer")

#Preferred number of job roles
preferred_number = st.text_input("Preferred Number of jobs", placeholder="e.g., 1,2,3")

# Submit Button
if st.button("Analyze Resume"):
    if resume_file:
        file_type = resume_file.name.split('.')[-1].lower()

        with st.spinner("Extracting text from resume..."):
            if file_type == "pdf":
                resume_text = extract_text_from_pdf(resume_file)
            elif file_type == "docx":
                resume_text = extract_text_from_docx(resume_file)
            else:
                st.error("Unsupported file type.")
                resume_text = ""

        if resume_text:
            with st.spinner("Analyzing resume with LLM..."):
                result = analyze_resume_with_llm(resume_text, preferred_number, preferred_location, preferred_roles)
                st.success("Resume Analysis Complete!")
                st.markdown("### 🔍 LLM Output:")
                st.markdown(result)
    else:
        st.warning("Please upload a resume file.")