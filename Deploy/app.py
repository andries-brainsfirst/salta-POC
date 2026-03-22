import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
import base64
from scipy.spatial import distance

# ==========================================
# 1. CONFIGURATION & BRANDING
# ==========================================
st.set_page_config(page_title="Study Choice Navigator | BrainsFirst", layout="wide", initial_sidebar_state="expanded")

def apply_custom_branding():
    custom_css = """
    <style>
        /* Import Poppins Font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        /* Enforce Poppins Font safely without breaking icons */
        html, body, p, h1, h2, h3, h4, h5, h6, li, a, label, span, div {
            font-family: 'Poppins', sans-serif !important;
        }

        /* PROTECT STREAMLIT ICONS: Stop Poppins from overriding the Material Icons font */
        [data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        }

        /* Hide default Streamlit bottom footer and right-side menu, but LEAVE header so sidebar toggle works */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Sidebar Background Color (Teal) */
        [data-testid="stSidebar"] {
            background-color: #3A506B !important;
        }

        /* Force Sidebar Text to be White for Contrast */
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] .stSelectbox label {
            color: #FFFFFF !important;
        }

        /* Keep dropdown menu text dark when expanded so it's readable */
        div[role="listbox"] * {
            color: #1A1A1A !important;
        }

        /* Hover effect for Expanders (Clickable Boxes) -> Cyan */
        [data-testid="stExpander"] details summary:hover {
            background-color: #41C5C5 !important;
            color: #1A1A1A !important;
            border-radius: 5px;
            transition: 0.3s ease-in-out;
        }

        /* Hover effect for Standard Buttons -> Cyan */
        .stButton button:hover {
            background-color: #41C5C5 !important;
            border-color: #41C5C5 !important;
            color: #1A1A1A !important;
            transition: 0.3s ease-in-out;
        }

        /* Make Expander headers bold */
        .streamlit-expanderHeader {
            font-weight: 600;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

NON_SKILL_COLS = ['Phase/Module', 'Specific Task', 'Inferred Course Competency', 'Expected Level']

@st.cache_data
def load_course_database():
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'courses')
    course_profiles = {}
    raw_courses = {} 
    
    ignore_files = ['competency_definitions_142.csv', 'Expanded_Dummy_Participants.csv']
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and not f.endswith('_agg.csv') and f not in ignore_files]
    
    all_skills = []

    for file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        course_name = file.replace('.csv', '')
        
        raw_courses[course_name] = df.copy()
        
        skills = [col for col in df.columns if col not in NON_SKILL_COLS]
        if not all_skills:
            all_skills = skills
            
        for skill in skills:
            df[skill] = pd.to_numeric(df[skill], errors='coerce').fillna(0)
            
        total_sums = df[skills].sum()
        max_sum = total_sums.max()
        
        if max_sum > 0:
            norm_100 = (total_sums / max_sum) * 100
        else:
            norm_100 = total_sums
            
        course_profiles[course_name] = norm_100.to_dict()
        
    return pd.DataFrame(course_profiles).T, all_skills, raw_courses

@st.cache_data
def load_skill_definitions():
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'courses')
    file_path = os.path.join(folder_path, 'competency_definitions_142.csv')
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return dict(zip(df['Skill_en'], df['Description_en']))
    return {}

@st.cache_data
def load_dummy_participants():
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'courses')
    file_path = os.path.join(folder_path, 'Expanded_Dummy_Participants.csv')
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.set_index(df.columns[0], inplace=True)
        return df
    return pd.DataFrame()

# ==========================================
# 2. NEUROLYMPICS API SIMULATION
# ==========================================
def fetch_neurolympics_data(skills_list, dummy_df, selected_candidate):
    if selected_candidate == "Random Profile (Demo)":
        np.random.seed(42) 
        dummy_scores = np.random.randint(0, 40, size=len(skills_list)) 
        spike_indices = np.random.choice(len(skills_list), 10, replace=False)
        dummy_scores[spike_indices] = np.random.randint(70, 100, size=10)
        return dict(zip(skills_list, dummy_scores))
    
    elif not dummy_df.empty and selected_candidate in dummy_df.index:
        user_row = dummy_df.loc[selected_candidate].fillna(0).to_dict()
        return {skill: float(user_row.get(skill, 0.0)) for skill in skills_list}
        
    return {skill: 0.0 for skill in skills_list}

# ==========================================
# 3. MATCHING ENGINES
# ==========================================
def calculate_matches(user_profile, course_df):
    user_vector = np.array([user_profile.get(skill, 0.0) for skill in course_df.columns])
    results = []
    max_possible_distance = np.sqrt(len(course_df.columns) * (100**2))
    
    for course_name, course_row in course_df.iterrows():
        course_vector = course_row.values
        dist = distance.euclidean(user_vector, course_vector)
        
        match_percentage = max(0, 100 * (1 - (dist / max_possible_distance)))
        adjusted_match = match_percentage ** 1.5 / 10 
        
        results.append({
            "Course": course_name,
            "Euclidean Distance": round(dist, 2),
            "Match Score (%)": round(min(100, adjusted_match), 1)
        })
        
    return pd.DataFrame(results).sort_values(by="Match Score (%)", ascending=False)

def get_top_tasks_for_user(user_profile, raw_course_df, skills, top_n=3):
    user_vec = np.array([user_profile.get(s, 0.0) for s in skills])
    task_matrix = raw_course_df[skills].fillna(0).values
    alignment_scores = task_matrix.dot(user_vec)
    
    scored_df = raw_course_df.copy()
    scored_df['Alignment_Score'] = alignment_scores
    top_tasks = scored_df.sort_values(by='Alignment_Score', ascending=False).head(top_n)
    
    return top_tasks

# ==========================================
# 4. STREAMLIT UI
# ==========================================
# ==========================================
# 5. SECURITY / LOGIN
# ==========================================
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Replace 'MakeEveryTalentCount' with whatever password you want to give your team
        if st.session_state["password"] == "MakeEveryTalentCount":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't keep the password in memory
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run: show the logo and the password input
        apply_custom_branding()
        st.markdown("<h2 style='text-align: center;'>BrainsFirst Internal Tool</h2>", unsafe_allow_html=True)
        st.text_input(
            "Please enter the access password", type="password", on_change=password_entered, key="password"
        )
        return False
    
    elif not st.session_state["password_correct"]:
        # Password incorrect: show input and an error message
        apply_custom_branding()
        st.markdown("<h2 style='text-align: center;'>BrainsFirst Internal Tool</h2>", unsafe_allow_html=True)
        st.text_input(
            "Please enter the access password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect. Please try again.")
        return False
    
    else:
        # Password correct.
        return True

def main():
    apply_custom_branding()
    
    # --- SAFE LOGO PLACEMENT ---
    folder_path = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(folder_path, 'Dark-Horizontal-Logo (1).svg')
    
    with st.sidebar:
        if os.path.exists(logo_path):
            # Encode the SVG to Base64 to completely insulate it from Streamlit's CSS bleed
            with open(logo_path, "rb") as f:
                base64_svg = base64.b64encode(f.read()).decode("utf-8")
            
            svg_uri = f"data:image/svg+xml;base64,{base64_svg}"
            
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; padding: 15px; border-radius: 8px; margin-bottom: 25px; text-align: center;">
                    <img src="{svg_uri}" style="max-width: 100%; height: auto;">
                </div>
                """, unsafe_allow_html=True
            )
        
        st.header("Candidate Setup")
        
        dummy_df = load_dummy_participants()
        if not dummy_df.empty:
            candidate_options = ["Random Profile (Demo)"] + dummy_df.index.tolist()
            selected_candidate = st.selectbox("Select a Candidate:", candidate_options)
        else:
            st.warning("Expanded_Dummy_Participants.csv not found.")
            selected_candidate = "Random Profile (Demo)"
            
        st.markdown("---")
        interest_filter = st.selectbox("Filter by Domain:", ["All", "IT & Tech", "Finance & Econ", "HR & Social"])

    # --- MAIN CONTENT ---
    st.title("Study Choice Navigator")
    st.markdown("Matching objective cognitive profiles against the rigorous demands of higher education.")
    
    with st.spinner("Loading Databases..."):
        course_df, all_skills, raw_courses_dict = load_course_database()
        skill_definitions = load_skill_definitions()
        
    if course_df.empty:
        st.error("No course CSVs found in the directory. Please add some to test.")
        return
    
    filtered_df = course_df.copy()
    if interest_filter == "IT & Tech":
        filtered_df = filtered_df[filtered_df.index.str.contains('IT|Cybersecurity|Software|Informatica|Data', case=False)]
    elif interest_filter == "Finance & Econ":
        filtered_df = filtered_df[filtered_df.index.str.contains('Finance|Economie|Kostencalculatie|Administratie|Accountancy', case=False)]
    elif interest_filter == "HR & Social":
        filtered_df = filtered_df[filtered_df.index.str.contains('Superdiversiteit|Criminologie|Arbeidsrecht', case=False)]

    user_profile = fetch_neurolympics_data(all_skills, dummy_df, selected_candidate)
    sorted_user_skills = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)
    
    # ==========================================
    # DISPLAY: USER PROFILE TABS
    # ==========================================
    st.markdown("---")
    st.subheader(f"Cognitive Profile: {selected_candidate}")
    
    tab1, tab2 = st.tabs(["Understand your brain", "Neurolympics Competency Profile"])
    
    with tab1:
        st.markdown("#### You have a natural talent for these competencies.")
        st.markdown("Click on any of your top skills below to read what it means and how it applies to your work style.")
        
        display_skills = sorted_user_skills[:10]
        col1, col2 = st.columns(2)
        
        for idx, (skill, score) in enumerate(display_skills):
            target_col = col1 if idx % 2 == 0 else col2
            with target_col:
                with st.expander(f"**{skill}** (Score: {score})"):
                    definition = skill_definitions.get(skill, "Definition not found in database.")
                    st.write(definition)

    with tab2:
        st.markdown("#### Your Complete Competency Profile")
        
        chart_data = []
        for skill, score in sorted_user_skills:
            desc = skill_definitions.get(skill, "No description available.")
            chart_data.append({"Skill": skill, "Score": score, "Description": desc})
            
        chart_df = pd.DataFrame(chart_data)
        dynamic_height = max(600, len(chart_df) * 22) 
        
        chart = alt.Chart(chart_df).mark_bar(color="#3A506B").encode(
            x=alt.X('Score:Q', title='Score (0-100)', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('Skill:N', sort='-x', title='', axis=alt.Axis(labelLimit=350, labelFontSize=12)),
            tooltip=['Skill', 'Score', 'Description']
        ).properties(
            height=dynamic_height
        )
        
        st.altair_chart(chart, use_container_width=True)

    # ==========================================
    # DISPLAY: COURSE MATCHES
    # ==========================================
    st.markdown("---")
    st.subheader("Your Best Course Matches")
    
    if filtered_df.empty:
        st.warning(f"No courses found in the '{interest_filter}' domain. Try selecting 'All'.")
    else:
        results_df = calculate_matches(user_profile, filtered_df)
        
        for index, row in results_df.iterrows():
            course_name = row['Course']
            match_score = row['Match Score (%)']
            
            with st.expander(f"**{course_name}** — {match_score}% Match"):
                st.markdown(f"*(Euclidean Distance: {row['Euclidean Distance']})*")
                st.markdown("#### Why you are a match:")
                st.write("Based on your cognitive strengths, you are naturally wired to excel at the following core tasks in this curriculum:")
                
                top_tasks = get_top_tasks_for_user(user_profile, raw_courses_dict[course_name], all_skills, top_n=3)
                
                for _, task_row in top_tasks.iterrows():
                    st.markdown(f"""
                    - **{task_row['Phase/Module']}**
                        - **Task:** {task_row['Specific Task']}
                        - **Competency Developed:** *{task_row['Inferred Course Competency']}*
                    """)

if __name__ == "__main__":
    if check_password():
        main()