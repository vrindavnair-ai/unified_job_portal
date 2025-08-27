from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
import re
import streamlit as st
import fitz  # PyMuPDF for PDF
from docx import Document
#To retry if the model is not working
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
load_dotenv()  # loads .env file
#ensure the .env files in below format
#GOOGLE_API_KEY = "xxxxx"
#TAVILY_API_KEY = "yyy"

#We are usin google gemini model
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

#retry logic to retry if the model is not available for any reason
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=10),
    retry=retry_if_exception_type(Exception)
)
def run_web_agent_with_retry(input_message,number):
        #Web search tool to get the current details
        search = TavilySearch(max_results=number)
        tools = [search]
        #Calling the agent with model and tool
        agent_executor = create_react_agent(model, tools)
        response = agent_executor.invoke({"messages": [input_message]})
        messages = response['messages']

        urls = []
        #to get the urls from the result
        url_pattern = re.compile(r'https?://[^\s\)\]"]+')

        for msg in messages:
            content = msg.content
            found_urls = url_pattern.findall(content)
            urls.extend(found_urls)

        # Remove duplicates
        urls = list(set(urls))
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        #print(url_str)
        # Filter out unwanted URLs (e.g., GitHub)
        trusted_domains = [
        "indeed.com",
        "glassdoor.com",
        "naukri.com",
        "linkedin.com/jobs",
        "monster.com",
        "reed.co.uk",
        "jobrapido.com",
        "aijobs.net",
        "zerotaxjobs.com",
        # add others you trust
        ]
        #filtered_urls = [url for url in unique_urls if "github.com" not in url]
        filtered_urls = [url for url in unique_urls if any(domain in url for domain in trusted_domains)]
        filtered_urls = filtered_urls[:number]
        # Create markdown formatted clickable links
        markdown_links = "\n".join([f"- [{url}]({url})" for url in filtered_urls])
        # Add a header or message before the links
        output_md = f"Please find the URLs of the jobs you can apply:\n\n{markdown_links}"
        return output_md


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
        input_message = {
        "role": "user",
        "content": f"""
        You are a job recruitment assistant helping candidates find the most suitable jobs matching their skill set.

        Analyze the following resume and extract:
        - Candidate's Name
        - Key Skills
        - Total Years of Experience
        - Suitable Job Titles
        - Preferred Industries

        Resume:
        {resume_text}

        Then, search the internet for jobs that:
        - Match the candidate's skills and experience
        - Are located in {location if location else "Dubai"}
        - Were posted within the last 24 hours
        - Are mainly for the role of {roles}, but you can also include other highly relevant job titles found in the resume
        - Return **only direct job posting URLs** (no articles, blogs, GitHub pages, or unrelated links)

        Return:
        - A **list of job URLs only** (no text or summaries), one per line
        - Limit results to {number if number else 5} unique job postings
        """
        }
        
        #to incorporate retry logic
        response = run_web_agent_with_retry(input_message, number)
        return response
    except Exception as e:
        print(e)
        return str(e)

# Title
st.title("AI Job Finder")

# Resume Upload
st.subheader("Upload Your Resume")
resume_file = st.file_uploader("Choose your resume (PDF or DOCX)", type=["pdf", "docx"])

# Location Preference
st.subheader("Job Preferences")
preferred_location = st.text_input("Preferred Job Location", placeholder="e.g., Dubai")
if preferred_location == "":
    preferred_location = "Dubai"
#preferred_location = st.text_input("Preferred Job Location", value="Dubai", placeholder="e.g., Dubai")

# Job Role or Keywords
preferred_roles = st.text_input("Preferred Roles or Keywords", placeholder="e.g., AI Developer, LLM, Prompt Engineer")

#Preferred number of job roles
preferred_number = st.text_input("Preferred Number of jobs", placeholder="e.g., 1,2,3")
#preferred_number = st.number_input("Preferred Number of jobs", min_value=1, max_value=10, value=3)


# Submit Button
if st.button("Get the job links"):
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




