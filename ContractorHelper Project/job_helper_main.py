
#pip install -r requirements.txt

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from dotenv import load_dotenv

load_dotenv()

# Initialize Search Tool
search_tool = SerperDevTool()

# 1. Define Agents
cv_analyzer = Agent(
    role='CV Analyst',
    goal='Extract key skills and experience levels from the user CV text',
    backstory='Expert in HR and technical recruiting, able to distill a CV into a search query.',
    verbose=True,
    allow_delegation=False
)

job_researcher = Agent(
    role='Job Hunter',
    goal='Search for current open contracting positions on LinkedIn, Indeed, and specialized boards',
    backstory='A meticulous search expert who knows how to find the most recent job postings.',
    tools=[search_tool],
    verbose=True
)

ranker = Agent(
    role='Job Matcher',
    goal='Select the 10 most relevant jobs and provide their application links',
    backstory='A career coach who ensures only the highest quality matches are suggested.',
    verbose=True
)

# 2. Define Tasks
def create_tasks(cv_text):
    analyze_cv = Task(
        description=f"Analyze this CV: {cv_text}. Identify top 5 skills and preferred role types.",
        expected_output="A list of keywords and job titles optimized for search engines.",
        agent=cv_analyzer
    )

    search_jobs = Task(
        description="Search for at least 20 open contract positions based on the CV analysis. Focus on May 2026 postings.",
        expected_output="A raw list of potential job openings with titles and URLs.",
        agent=job_researcher,
        context=[analyze_cv]
    )

    format_output = Task(
        description="Filter the search results down to the 10 most suitable. Provide a clear Markdown list with: Title, Company, Why it matches, and [Link].",
        expected_output="A final report of 10 job links.",
        agent=ranker,
        context=[search_jobs]
    )
    
    return [analyze_cv, search_jobs, format_output]

# 3. Execution
my_cv = "Experienced Python Developer with 5 years in AI and Cloud Architecture..." # Replace with actual text

job_crew = Crew(
    agents=[cv_analyzer, job_researcher, ranker],
    tasks=create_tasks(my_cv),
    process=Process.sequential
)

result = job_crew.kickoff()
print(result)