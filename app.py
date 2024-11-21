import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import re

# Import the Dataset
skincare = pd.read_csv("export_skincare.csv", encoding='utf-8', index_col=None)

# Set up page configuration with customized layout
st.set_page_config(page_title="Skin Care Recommender System", page_icon=":rose:", layout="wide")

# Custom CSS for enhanced aesthetics and innovative color combinations
st.markdown("""
    <style>
        /* General body styling */
        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(to right, #fbc2eb, #a18cd1); /* Gradient background */
            color: #333;
        }
        .section-header {
            text-align: center;
            font-weight: bold;
            font-size: 32px;
            color: #ffffff;
            margin: 20px 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .video-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        video {
            max-width: 800px;
            width: 100%;
            height: auto;
            border-radius: 15px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }
        .info-box {
            text-align: center;
            margin-top: 20px;
            color: #ffffff;
        }
        .stButton button {
            background-color: #6c63ff;
            color: #ffffff;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
        }
        .stButton button:hover {
            background-color: #4e47d1;
        }
    </style>
""", unsafe_allow_html=True)
#-------
# Hero Section
st.markdown('<div class="hero"><h1 style="color:#6c63ff;">UltimAI</h1><p>Discover tailored skincare products.</p></div>', unsafe_allow_html=True)

# Navigation Tabs
selected_tab = option_menu(
    menu_title="",  
    options=["Home", "Recommendations", "Skin Care Essentials"],  
    icons=["house", "search", "book"],  
    menu_icon="cast",  
    default_index=0,  
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa"},
        "icon": {"color": "#6c63ff", "font-size": "18px"}, 
        "nav-link": {
            "font-size": "20px",
            "text-align": "center",
            "margin": "0px",
            "color": "#6c63ff",
            "--hover-color": "#e0e0e0",
        },
        "nav-link-selected": {"background-color": "#6c63ff", "color": "#ffffff"},
    },
)

# Home Tab Content
if selected_tab == "Home":
    st.markdown('<div class="section-header"></div>', unsafe_allow_html=True)
    #st.write("Explore skincare products personalized for your needs!")
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    st.video("skincare.mp4", start_time=1)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.info("Developed by Katyayani Jandhyala")
    st.markdown('</div>', unsafe_allow_html=True)

# Recommendations Tab Content
elif selected_tab == "Recommendations":
    st.markdown('<div class="section-header" style="color:#6c63ff;">Your Personalized Skin Care Recommendations</div>', unsafe_allow_html=True)
    ##st.write("### Share your skin details for customized product suggestions.")
    
    # Interactive Input with Three Columns
    col1, col2, col3 = st.columns(3)

    # Product Category selection
    category = col1.selectbox('Select Product Category:', options=skincare['product_category'].str.title().unique())
    category_pt = skincare[skincare['product_category'].str.title() == category]

    # Skin Type selection
    skin_type = col2.selectbox('Select Your Skin Type:', options=['Normal Skin', 'Dry Skin', 'Oily Skin', 'Combination Skin', 'Sensitive Skin'])
    category_st_pt = category_pt[category_pt['skintype'].str.contains(skin_type, case=False, na=False)]

    # Skin Concerns
    with st.expander("Select Your Skin Concerns"):
        prob = st.multiselect('Skin Problems:', options=[
            "Dull Skin", "Acne", "Scars", "Pigmentation", "Wrinkles", "Scaling", "Back Acne", "Bacne",
            "Lines", "Pores", "Eczema", "Acne Scars", "Large Pores", "Dark Spots",
            "Fine Lines And Wrinkles", "Blackheads", "Uneven Skin Tone", "Darkness",
            "Redness", "Sagging Skin", "White Heads", "Black Heads"
        ])

    # Notable Effects multiselect
    unique_values = set()
    category_st_pt['notable_effects'].dropna().str.split(',').apply(lambda items: unique_values.update([re.sub(r'[^a-zA-Z0-9\s]', '', item).strip() for item in items]))
    opsi_ne = list(unique_values)
    selected_options = st.multiselect('Select Desired Effects:', opsi_ne)
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

    # Product Card Layout
    #st.markdown("<div class='section-header'>Recommended Products for You:</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
    for product in sorted(category_ne_st_pt['product_name'].unique()):
        st.markdown(f"<div class='product-card'><b>{product}</b></div>", unsafe_allow_html=True)
#-------
    # Recommendation Engine Code
    skincare["notable_effects"] = skincare["notable_effects"].fillna("unknown")
    vectorizer = TfidfVectorizer(stop_words='english')
    effect_vectors = vectorizer.fit_transform(skincare['notable_effects'])
    cosine_sim = cosine_similarity(effect_vectors)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

    def skincare_recommendations(input_effects, skincare, effect_vectors, cosine_sim_df, top_n=5):
        input_vector = vectorizer.transform([input_effects])
        input_similarities = cosine_similarity(input_vector, effect_vectors).flatten()
        input_similar_indices = input_similarities.argsort()[-top_n:][::-1]
        input_recommendations = skincare.iloc[input_similar_indices][['product_name', 'notable_effects','benefits']]
        input_recommendations['similarity_score'] = input_similarities[input_similar_indices]
        input_recommendations = input_recommendations[input_recommendations['similarity_score'] > 0.1]
        
        product_index = input_similar_indices[0]
        product_name = skincare.iloc[product_index]['product_name']
        matrix_similarities = cosine_sim_df.loc[product_name].to_numpy()
        matrix_similar_indices = matrix_similarities.argsort()[-top_n:][::-1]
        matrix_recommendations = skincare.iloc[matrix_similar_indices][['product_name', 'notable_effects','benefits']]
        matrix_recommendations['similarity_score'] = matrix_similarities[matrix_similar_indices]
        matrix_recommendations = matrix_recommendations[matrix_recommendations['product_name'] != product_name]
        matrix_recommendations = matrix_recommendations[matrix_recommendations['similarity_score'] > 0.1]

        combined_recommendations = pd.concat([input_recommendations, matrix_recommendations]).drop_duplicates()
        combined_recommendations = combined_recommendations[['product_name','benefits']]
        combined_recommendations = combined_recommendations.rename(columns={'product_name': 'Product', 'benefits': 'Benefits'})
        ncombined_recommendations = combined_recommendations.reset_index(drop=True)
        return combined_recommendations[['Product','Benefits']].head(top_n)

    # Join selected concerns to use as input for recommendations
    input_effects = " ".join(prob)
    # Button to Generate Recommendations
    if st.button('Find More Product Recommendations!'):
        st.write("### Here are some product recommendations based on your preferences:")
        st.write(skincare_recommendations(input_effects, skincare, effect_vectors, cosine_sim_df))

# Skin Care Essentials Tab Content
elif selected_tab == "Skin Care Essentials":
    #st.title(" Skin Care Essentials - Tips & Tricks")style="color:#6c63ff;
    st.markdown('<h1 style="color:#6c63ff;">Skin Care Essentials - Tips & Tricks</h1>',unsafe_allow_html=True)
    st.write("---")
    st.write("### Enhance your skincare routine with these helpful tips:")
    st.image("imagepic.jpg", caption="Skin Care Essentials")

    # Use expanders for each skincare step
    with st.expander("1. Facial Wash"):
        st.write("""
        - Use a facial wash recommended for your skin type.
        - Wash no more than twice daily: once in the morning and once before bed.
        - Avoid harsh scrubbing and use your fingertips in gentle circular motions.
        """)

    with st.expander("2. Toner"):
        st.write("""
        - Choose a toner suited for your skin.
        - Apply using a cotton pad or hands for better absorption.
        - Use immediately after cleansing.
        """)

    with st.expander("3. Serum"):
        st.write("""
        - Select a serum tailored to your needs, e.g., acne scars or anti-aging.
        - Apply after cleansing for maximum absorption.
        - Use both morning and night.
        """)

    with st.expander("4. Moisturizer"):
        st.write("""
        - Use a moisturizer that matches your skin type.
        - Daytime moisturizers often include sunscreen, while nighttime moisturizers aid skin regeneration.
        """)

    with st.expander("5. Sunscreen"):
        st.write("""
        - Use sunscreen daily to protect against UVA, UVB, and blue light.
        - Reapply every 2-3 hours, especially if outdoors.
        """)

    with st.expander("6. Avoid Frequent Product Changes"):
        st.write("""
        - Constantly switching products can stress your skin. Stick with the same products to see their effects.
        """)

    with st.expander("7. Consistency is Key"):
        st.write("""
        - Consistency is crucial for long-term results in skincare.
        """)

    with st.expander("8. Your Face is Your Asset"):
        st.write("""
        - Treat your skin with care. Investing in skincare early benefits your future self.
        """)

    st.info("Note: For personalized advice, please consult a dermatologist.")
