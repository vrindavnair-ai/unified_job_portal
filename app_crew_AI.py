import streamlit as st
import fitz  # PyMuPDF for PDF
#import docx
from docx import Document
#To retry if the model is not working
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
#import openai
import google.generativeai as genai
import os
from crewai import Agent, Task, Crew
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.chat_models import ChatGooglePalm
from dotenv import load_dotenv
load_dotenv()
#from langchain_core.tools import Tool  # ✅ Import correct Tool class
#from langchain.tools import Tool
from crewai.tools import BaseTool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

class CrewAIToolWrapper(BaseTool):
     def __init__(self, lc_tool):
        super().__init__(name=lc_tool.name, description=lc_tool.description)
        self.lc_tool = lc_tool

     def _run(self, *args, **kwargs):
        # Call the LangChain tool's run method
        return self.lc_tool.run(*args, **kwargs)

lc_search_tool = DuckDuckGoSearchRun()
search_tool = CrewAIToolWrapper(lc_search_tool)

"""
# Wrap it correctly
search_tool = Tool.from_function(
    func=DuckDuckGoSearchRun().run,
    name="DuckDuckGo Search",
    description="Search the web for job listings"
)
"""


# Set your OpenAI API key (for testing use dotenv or Streamlit secrets)
#openai.api_key = "YOUR_OPENAI_API_KEY"
# Set your google API key (for testing use dotenv or Streamlit secrets)
#genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#model = genai.GenerativeModel("gemini-1.5-flash")
# Tool: DuckDuckGo Search
#search_tool = DuckDuckGoSearchRun()


#Web Agent to get the real time data from web
# Agent: Resume Analyst
resume_analyst = Agent(
    role="Resume Analyst",
    goal="Extract key details from a candidate's resume",
    backstory="Expert in understanding and interpreting resumes to identify key skills and suitable roles.",
    tools=[],
    verbose=True,
    llm=ChatGooglePalm(temperature=0.3),
)


# Agent: Job Finder
job_finder = Agent(
    role="Job Finder",
    goal="Search the web for suitable job opportunities based on resume analysis",
    backstory="Skilled at using web search tools to find real-time job listings.",
    tools=[search_tool],
    verbose=True,
    llm=ChatGooglePalm(temperature=0.3),
)



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
                # Sample extracted resume text (you can get this from your resume parser)
        resume_text = resume_text

        # Define Tasks
        task1 = Task(
            description=f"""Analyze this resume:
        {resume_text}

        Extract:
        - Candidate's Name
        - Key Skills
        - Years of Experience
        - Suitable Job Titles
        - Preferred Industries
        """,
            expected_output="A structured profile summary of the candidate.",
            agent=resume_analyst,
        )

        task2 = Task(
            description="""Based on the extracted profile, search for real-time job openings in Dubai relevant to the candidate.
        Provide job links and short descriptions in a table.""",
            expected_output="A markdown-formatted table with live job postings and links.",
            agent=job_finder,
        )

        # Define Crew
        crew = Crew(
            agents=[resume_analyst, job_finder],
            tasks=[task1, task2],
            verbose=True,
        )

        # Run
        result = crew.run()
        print(result)
        #using web agent
        #response = web_agent.run(prompt)
        #to incorporate retry logic
        return result
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