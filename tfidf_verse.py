import numpy as np
import pandas as pd
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import streamlit as st

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('stopwords')

# Streamlit Page Configuration
st.set_page_config(page_title="Bible Verse Recommender", layout="wide")
##################################################################################################################
# Cache Data Loading
@st.cache_data
def load_data():
    data = pd.read_csv("t_bbe.csv").dropna()
    
    # Map book numbers to book names
    book_names = {
        1: 'Genesis', 2: 'Exodus', 3: 'Leviticus', 4: 'Numbers', 5: 'Deuteronomy',
        6: 'Joshua', 7: 'Judges', 8: 'Ruth', 9: '1 Samuel', 10: '2 Samuel',
        11: '1 Kings', 12: '2 Kings', 13: '1 Chronicles', 14: '2 Chronicles',
        15: 'Ezra', 16: 'Nehemiah', 17: 'Esther', 18: 'Job', 19: 'Psalms',
        20: 'Proverbs', 21: 'Ecclesiastes', 22: 'Song of Solomon', 23: 'Isaiah',
        24: 'Jeremiah', 25: 'Lamentations', 26: 'Ezekiel', 27: 'Daniel',
        28: 'Hosea', 29: 'Joel', 30: 'Amos', 31: 'Obadiah', 32: 'Jonah',
        33: 'Micah', 34: 'Nahum', 35: 'Habakkuk', 36: 'Zephaniah', 37: 'Haggai',
        38: 'Zechariah', 39: 'Malachi', 40: 'Matthew', 41: 'Mark', 42: 'Luke',
        43: 'John', 44: 'Acts', 45: 'Romans', 46: '1 Corinthians',
        47: '2 Corinthians', 48: 'Galatians', 49: 'Ephesians', 50: 'Philippians',
        51: 'Colossians', 52: '1 Thessalonians', 53: '2 Thessalonians',
        54: '1 Timothy', 55: '2 Timothy', 56: 'Titus', 57: 'Philemon',
        58: 'Hebrews', 59: 'James', 60: '1 Peter', 61: '2 Peter',
        62: '1 John', 63: '2 John', 64: '3 John', 65: 'Jude', 66: 'Revelation'
    }

    # Map book numbers to names
    data['Book Name'] = data['b'].map(book_names)

    # Process text: Remove stopwords
    stop_words = set(stopwords.words('english'))
    data['corpus'] = data['t'].astype(str).str.lower().apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )

    return data, book_names

# Load data
data, book_names = load_data()
##################################################################################################################
# Compute TF-IDF & Cosine Similarity
@st.cache_resource
def compute_similarity():
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(data['corpus'])
    return cosine_similarity(tf_idf_matrix)

similarity_matrix = compute_similarity()

# Reverse book name lookup
book_numbers = {v: k for k, v in book_names.items()}

# **Find Similar Verses Function**
def top_verse(input_book, input_chapter, input_verse, top_n=10):
    try:
        book_num = str(book_numbers.get(input_book, ""))
        locator = data.loc[
            (data['b'].astype(str) == book_num) &
            (data['c'].astype(str) == str(input_chapter)) &
            (data['v'].astype(str) == str(input_verse))
        ]
        if locator.empty:
            return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])
        idx = locator.index[0]
        similarity_scores = list(enumerate(similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in similarity_scores[1:top_n + 1]]
        sim_values = [i[1] for i in similarity_scores[1:top_n + 1]]
        recommended = data.iloc[sim_indices].copy()
        recommended['Similarity Score'] = sim_values
        recommended = recommended[['Book Name', 'c', 'v', 't', 'Similarity Score']]
        recommended.columns = ["Book", "Chapter", "Verse", "Text", "Similarity Score"]
        return recommended
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])

##################################################################################################################

st.title("📖 Bible Verse Search")
st.info("Enter a Book, Chapter and Verse ➡️ click 'Find Similar Verses' to find relevant Bible verses.")

with st.form("user_input"):
    col1, col2, col3 = st.columns(3)
    with col1:
        input_book = st.selectbox("Select Book", book_names.values())
    with col2:
        input_chapter = st.number_input("Chapter", min_value=1, max_value=150, value=1, step=1)
    with col3:
        input_verse = st.number_input("Verse", min_value=1, max_value=176, value=1, step=1)

    top_n = st.slider("Number of Similar Verses", min_value=1, max_value=50, value=10, step=5)

    submitted = st.form_submit_button("➡️Find Similar Verses")

if submitted:
    results = top_verse(input_book, input_chapter, input_verse, top_n)
    searched_verse = data.loc[
        (data['Book Name'] == input_book) &
        (data['c'].astype(str) == str(input_chapter)) &
        (data['v'].astype(str) == str(input_verse))
    ]

    if not searched_verse.empty:
        st.write(f"**Input Verse:** {searched_verse.iloc[0]['t']}")
        st.write("### 🔍 Similar Verses:")
        # Highlight key verses
        # The following 3 lines can be one option for displaying results in a way other than a table

        #for index, row in results.iterrows():
        #    st.markdown(f"**{row['Book']} {row['Chapter']}:{row['Verse']}** - {row['Text']}")
        #    st.markdown("---")
        # If the above 3 lines are uncommented, you may want to comment out the very last line below to avoid displaying the results twice
    else:
        st.write("Verse not found.")
    # One option for displaying results
    #st.table(results)
    # Another option for displaying results
    #st.dataframe(results, hide_index=True, use_container_width=True)
    # Apply text wrapping in display of results
    st.dataframe(results.style.set_properties(subset=['Text'], **{'white-space': 'normal'}))