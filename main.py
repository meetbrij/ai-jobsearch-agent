import streamlit as st
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Initialize Environment
load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 1. Page Configuration
st.set_page_config(
    page_title="AI Job Search Agent",
    page_icon="💼",
    layout="centered"
)

# 2. Production Structured Schemas
class JobPosting(BaseModel):
    """Schema for an individual structured job posting entry"""
    title: str = Field(description="The formal job title of the posting")
    company: str = Field(description="The name of the company offering the role")
    summary: str = Field(description="A concise 3-4 bullet-point summary summarizing core tech requirements, responsibilities, and benefits.")
    source_url: str = Field(description="The direct source, verified application, or listing URL explicitly present in the provided context text. DO NOT fabricate or hallucinate any URL.")

class AgentResponse(BaseModel):
    """Root schema ensuring the LLM maps responses strictly to a structural collection"""
    jobs: List[JobPosting] = Field(description="List of unique matching job postings discovered matching the requested count")

# 3. Initialize Production Models (Cached)
@st.cache_resource
def init_models():
    # Model 1: Search engine (free to browse and list raw text/links)
    search_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    search_llm_with_tools = search_llm.bind_tools([{"type": "web_search"}])
    
    # Model 2: Strict Extractor (destined only to map verified facts into the UI)
    extractor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_extractor = extractor_llm.with_structured_output(AgentResponse)
    
    return search_llm_with_tools, structured_extractor

search_agent, structured_extractor = init_models()

# 4. State Management
if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False
if "structured_result" not in st.session_state:
    st.session_state.structured_result = None

# Header Banner
st.title("💼 AI Job Search Agent")
st.write("Find verified job postings dynamically across the web using two-step grounded extraction.")
st.markdown("---")

# 5. Conditional UI View Rendering
if not st.session_state.search_clicked:
    # --- SEARCH VIEW ---
    st.subheader("Specify Search Criteria")
    
    # Input Fields
    job_title = st.text_input("Job Title / Target Role", placeholder="e.g., AI Engineer, Python Developer")
    location = st.text_input("Location / Country", placeholder="e.g., India, Bangalore, San Francisco")
    
    # Combined Layout for Dropdowns using Columns
    col1, col2 = st.columns(2)
    with col1:
        job_type = st.selectbox("Job Type", ["Any", "Remote", "Hybrid", "On-site"])
    with col2:
        job_count = st.selectbox("Number of Results", [5, 10, 15, 20])
    
    if st.button("Submit Search", type="primary"):
        if not job_title.strip() or not location.strip():
            st.warning("Please fill out both the Job Title and Location fields.")
        else:
            # Handle "Any" type phrasing dynamically
            type_phrasing = "any setup (remote, hybrid, or on-site)" if job_type == "Any" else f"{job_type.lower()}"
            
            # Step 1 Prompt: Focus purely on finding real postings and printing real links
            search_prompt = (
                f"Search the web for exactly {job_count} real, active {type_phrasing} job openings for an '{job_title}' "
                f"in or for candidates based in '{location}'. Provide a comprehensive list with the title, company name, "
                f"brief details, and the exact, real application or source URL link for each opening. "
                f"Make sure to explicitly output the actual source links you find in the search results."
            )
            
            # Execute Workflow inside visual loaders
            try:
                # --- STEP 1: GROUNDED WEB SEARCH ---
                with st.spinner("Step 1/2: Browsing live web indexes for real job openings..."):
                    search_response = search_agent.invoke([HumanMessage(content=search_prompt)])
                    raw_text_content = search_response.content

                # --- STEP 2: VERIFIED EXTRACTION ---
                with st.spinner("Step 2/2: Formatting data and extracting real source links..."):
                    extraction_prompt = (
                        f"Analyze the following job search data. Extract exactly the details requested into the "
                        f"structured schema format. Only include source URLs that are explicitly mentioned in the text. "
                        f"If a posting does not have a real URL provided in the text, use '#' as the source_url. "
                        f"Do not invent any URLs.\n\nJob Search Data:\n{raw_text_content}"
                    )
                    final_structured_data = structured_extractor.invoke([HumanMessage(content=extraction_prompt)])
                    
                # Store the strictly typed Pydantic object directly into the session state
                st.session_state.structured_result = final_structured_data
                st.session_state.search_clicked = True
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred while compiling your request: {e}")

else:
    # --- RESULTS VIEW ---
    if st.button("⬅ Back to Search", type="secondary"):
        st.session_state.search_clicked = False
        st.session_state.structured_result = None
        st.rerun()
        
    st.success("Search completed successfully!")
    st.subheader("🔍 Found Openings")
    
    final_data: AgentResponse = st.session_state.structured_result

    # Safe validation check if no items were passed or if an empty list was yielded
    if not final_data or not final_data.jobs:
        st.info("No matching job postings found. Try adjustments to your target location or phrasing criteria.")
    else:
        # Loop cleanly over typed model array entries without any string regex hacks
        for index, job in enumerate(final_data.jobs, 1):
            expander_title = f"{index}. 💼 {job.title} — {job.company}"
            
            with st.expander(expander_title, expanded=False):
                st.markdown("**Role Summary & Requirements:**")
                st.write(job.summary)
                st.markdown("---")
                if job.source_url and job.source_url != "#" and job.source_url.startswith("http"):
                    st.markdown(f"🔗 [Apply Here / View Source]({job.source_url})")
                else:
                    st.caption("⚠️ No direct, verifiable link returned in the primary search stream.")