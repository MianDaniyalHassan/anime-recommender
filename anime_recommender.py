import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Anime Recommender System",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        color: #4ECDC4;
        font-size: 1.5rem;
        margin-top: 2rem;
    }
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_anime_dataset():
    """
    Generate a mock anime dataset for the recommender system.
    In a real implementation, you would load this from a CSV file.
    """
    anime_data = {
        'title': [
            'Naruto', 'One Piece', 'Attack on Titan', 'Death Note', 'Dragon Ball Z',
            'My Hero Academia', 'Demon Slayer', 'One Punch Man', 'Fullmetal Alchemist',
            'Hunter x Hunter', 'Tokyo Ghoul', 'Bleach', 'JoJo\'s Bizarre Adventure',
            'Mob Psycho 100', 'Your Name', 'Spirited Away', 'Princess Mononoke',
            'Akira', 'Ghost in the Shell', 'Cowboy Bebop', 'Evangelion', 'Steins;Gate',
            'Code Geass', 'Monster', 'Parasyte', 'Black Clover', 'Fire Force',
            'Dr. Stone', 'Promised Neverland', 'Violet Evergarden', 'A Silent Voice',
            'Weathering with You', 'Your Lie in April', 'Clannad', 'Anohana',
            'Haikyuu!!', 'Kuroko\'s Basketball', 'Slam Dunk', 'Initial D', 'Food Wars'
        ],
        'genre': [
            'Action Shounen', 'Action Adventure', 'Action Drama Horror', 'Psychological Thriller',
            'Action Shounen', 'Action Superhero Shounen', 'Action Supernatural Shounen',
            'Action Comedy Superhero', 'Action Adventure Drama', 'Action Adventure Shounen',
            'Action Horror Supernatural', 'Action Shounen', 'Action Adventure Bizarre',
            'Supernatural Comedy', 'Romance Drama', 'Adventure Fantasy Family',
            'Adventure Fantasy Drama', 'Action Sci-Fi Thriller', 'Action Sci-Fi Cyberpunk',
            'Action Space Western', 'Mecha Psychological', 'Sci-Fi Thriller Time-Travel',
            'Mecha Political Drama', 'Psychological Horror Thriller', 'Horror Sci-Fi Psychological',
            'Action Magic Shounen', 'Action Supernatural Shounen', 'Adventure Sci-Fi Comedy',
            'Psychological Thriller Horror', 'Drama Romance Fantasy', 'Drama Romance School',
            'Romance Drama Fantasy', 'Drama Music Romance', 'Drama Romance Slice-of-Life',
            'Drama Supernatural Romance', 'Sports Volleyball Shounen', 'Sports Basketball Shounen',
            'Sports Basketball Drama', 'Sports Racing Action', 'Comedy Food Ecchi'
        ],
        'rating': [
            8.7, 9.0, 9.0, 9.0, 8.7, 8.7, 8.7, 8.7, 9.1, 9.0, 8.7, 8.1, 8.5,
            8.6, 8.2, 9.3, 8.4, 8.0, 8.0, 8.8, 8.5, 9.0, 8.9, 9.0, 8.7, 8.2,
            7.7, 8.7, 8.9, 8.5, 8.2, 8.1, 8.7, 8.9, 8.7, 8.7, 8.6, 9.1, 8.1, 8.5
        ],
        'year': [
            2002, 1999, 2013, 2006, 1989, 2016, 2019, 2015, 2003, 2011, 2014, 2004,
            2012, 2016, 2016, 2001, 1997, 1988, 1995, 1998, 1995, 2011, 2006, 2004,
            2014, 2017, 2019, 2019, 2019, 2018, 2016, 2019, 2011, 2007, 2011, 2012,
            2003, 1990, 1998, 2015
        ],
        'episodes': [
            720, 1000, 87, 37, 291, 138, 44, 24, 64, 148, 48, 366, 190, 37, 1, 1, 1,
            1, 1, 26, 26, 24, 50, 74, 24, 170, 48, 24, 12, 13, 1, 1, 22, 44, 11,
            75, 25, 101, 39, 24
        ]
    }
    
    return pd.DataFrame(anime_data)

def calculate_similarity_matrix(df):
    """
    Calculate similarity matrix using TF-IDF on genres and cosine similarity.
    """
    # Use TF-IDF to vectorize genres
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(df['genre'])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(genre_matrix)
    
    return similarity_matrix

def get_recommendations(title, df, similarity_matrix, genre_filter=None, n_recommendations=10):
    """
    Get anime recommendations based on similarity scores.
    """
    try:
        # Get the index of the selected anime
        idx = df[df['title'] == title].index[0]
        
        # Get similarity scores for all anime
        sim_scores = list(enumerate(similarity_matrix[idx]))
        
        # Sort by similarity score (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Remove the selected anime itself
        sim_scores = sim_scores[1:]
        
        # Filter by genre if specified
        if genre_filter and genre_filter != "All Genres":
            filtered_scores = []
            for i, score in sim_scores:
                if genre_filter.lower() in df.iloc[i]['genre'].lower():
                    filtered_scores.append((i, score))
            sim_scores = filtered_scores
        
        # Get top N recommendations
        sim_scores = sim_scores[:n_recommendations]
        
        # Get anime indices
        anime_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Create recommendations dataframe
        recommendations = df.iloc[anime_indices].copy()
        recommendations['similarity_score'] = similarity_scores
        
        return recommendations
        
    except IndexError:
        st.error(f"Anime '{title}' not found in the database.")
        return pd.DataFrame()

def plot_similarity_scores(recommendations):
    """
    Create a bar chart of similarity scores for recommendations.
    """
    if recommendations.empty:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    bars = ax.barh(range(len(recommendations)), recommendations['similarity_score'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(recommendations))))
    
    # Customize the plot
    ax.set_yticks(range(len(recommendations)))
    ax.set_yticklabels(recommendations['title'], fontsize=10)
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_title('Anime Recommendations - Similarity Scores', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_genre_distribution(recommendations):
    """
    Create a pie chart showing genre distribution of recommendations.
    """
    if recommendations.empty:
        return None
    
    # Extract all genres from recommendations
    all_genres = []
    for genres in recommendations['genre']:
        genre_list = [g.strip() for g in genres.split()]
        all_genres.extend(genre_list)
    
    # Count genre occurrences
    genre_counts = pd.Series(all_genres).value_counts()
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use only top 8 genres to avoid clutter
    top_genres = genre_counts.head(8)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_genres)))
    wedges, texts, autotexts = ax.pie(top_genres.values, labels=top_genres.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    
    ax.set_title('Genre Distribution in Recommendations', fontsize=14, fontweight='bold')
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.tight_layout()
    return fig

def plot_rating_comparison(selected_anime, recommendations, df):
    """
    Create a bar chart comparing ratings of selected anime vs recommendations.
    """
    if recommendations.empty:
        return None
    
    # Get selected anime rating
    selected_rating = df[df['title'] == selected_anime]['rating'].iloc[0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    titles = ['Selected: ' + selected_anime] + list(recommendations['title'])
    ratings = [selected_rating] + list(recommendations['rating'])
    colors = ['red'] + ['skyblue'] * len(recommendations)
    
    # Create bar chart
    bars = ax.bar(range(len(titles)), ratings, color=colors, alpha=0.7)
    
    # Customize plot
    ax.set_xlabel('Anime', fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)
    ax.set_title('Rating Comparison: Selected Anime vs Recommendations', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(titles)))
    ax.set_xticklabels(titles, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 10)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def main():
    """
    Main Streamlit application.
    """
    # App header
    st.markdown('<h1 class="main-header">🎌 Anime Recommender System</h1>', unsafe_allow_html=True)
    st.markdown("Discover your next favorite anime based on similarity to your preferences!")
    
    # Load data
    df = load_anime_dataset()
    similarity_matrix = calculate_similarity_matrix(df)
    
    # Sidebar for user inputs
    st.sidebar.header("🔍 Find Your Next Anime")
    
    # Anime selection
    selected_anime = st.sidebar.selectbox(
        "Select an anime you enjoyed:",
        df['title'].sort_values().tolist(),
        help="Choose an anime you've watched and enjoyed to get similar recommendations"
    )
    
    # Genre filter
    all_genres = set()
    for genres in df['genre']:
        genre_list = [g.strip() for g in genres.split()]
        all_genres.update(genre_list)
    
    genre_filter = st.sidebar.selectbox(
        "Filter by genre (optional):",
        ["All Genres"] + sorted(list(all_genres)),
        help="Optionally filter recommendations by a specific genre"
    )
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=15,
        value=10,
        help="How many anime recommendations would you like?"
    )
    
    # Get recommendations button
    if st.sidebar.button("🎯 Get Recommendations", type="primary"):
        st.session_state.show_recommendations = True
        st.session_state.selected_anime = selected_anime
        st.session_state.genre_filter = genre_filter
        st.session_state.n_recommendations = n_recommendations
    
    # Display selected anime info
    if selected_anime:
        selected_info = df[df['title'] == selected_anime].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### 📺 Selected Anime")
            st.write(f"**Title:** {selected_info['title']}")
            st.write(f"**Genre:** {selected_info['genre']}")
            st.write(f"**Rating:** {selected_info['rating']}/10")
            st.write(f"**Year:** {selected_info['year']}")
            st.write(f"**Episodes:** {selected_info['episodes']}")
    
    # Show recommendations if button was clicked
    if hasattr(st.session_state, 'show_recommendations') and st.session_state.show_recommendations:
        
        # Get recommendations
        with st.spinner("Finding similar anime..."):
            recommendations = get_recommendations(
                st.session_state.selected_anime, 
                df, 
                similarity_matrix, 
                st.session_state.genre_filter, 
                st.session_state.n_recommendations
            )
        
        if not recommendations.empty:
            st.markdown('<h2 class="subheader">🎯 Recommendations for you</h2>', unsafe_allow_html=True)
            
            # Display recommendations table
            display_df = recommendations[['title', 'genre', 'rating', 'year', 'episodes', 'similarity_score']].copy()
            display_df['similarity_score'] = display_df['similarity_score'].round(3)
            display_df.columns = ['Title', 'Genre', 'Rating', 'Year', 'Episodes', 'Similarity']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400,
                column_config={
                    "Title": st.column_config.TextColumn("Title", width="large"),
                    "Genre": st.column_config.TextColumn("Genre", width="large"),
                    "Rating": st.column_config.NumberColumn("Rating", format="%.1f"),
                    "Year": st.column_config.NumberColumn("Year", format="%d"),
                    "Episodes": st.column_config.NumberColumn("Episodes", format="%d"),
                    "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1)
                }
            )
            
            # Visualization section
            st.markdown('<h2 class="subheader">📊 Visual Analysis</h2>', unsafe_allow_html=True)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["📈 Similarity Scores", "🎭 Genre Distribution", "⭐ Rating Comparison"])
            
            with tab1:
                fig1 = plot_similarity_scores(recommendations)
                if fig1:
                    st.pyplot(fig1)
                    st.markdown("*Higher similarity scores indicate more similar anime based on genre analysis.*")
            
            with tab2:
                fig2 = plot_genre_distribution(recommendations)
                if fig2:
                    st.pyplot(fig2)
                    st.markdown("*This shows the distribution of genres in your recommendations.*")
            
            with tab3:
                fig3 = plot_rating_comparison(st.session_state.selected_anime, recommendations, df)
                if fig3:
                    st.pyplot(fig3)
                    st.markdown("*Compare the rating of your selected anime with the recommendations.*")
            
            # Statistics
            st.markdown('<h2 class="subheader">📈 Statistics</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_rating = recommendations['rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.1f}/10")
            
            with col2:
                avg_similarity = recommendations['similarity_score'].mean()
                st.metric("Average Similarity", f"{avg_similarity:.3f}")
            
            with col3:
                total_episodes = recommendations['episodes'].sum()
                st.metric("Total Episodes", f"{total_episodes:,}")
            
            with col4:
                unique_genres = len(set([g.strip() for genres in recommendations['genre'] for g in genres.split()]))
                st.metric("Unique Genres", unique_genres)
        
        else:
            st.warning("No recommendations found. Try adjusting your filters.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**How it works:** This recommender system uses TF-IDF vectorization on anime genres "
        "and calculates cosine similarity to find anime with similar characteristics. "
        "The similarity score ranges from 0 to 1, where 1 indicates perfect similarity."
    )
    
    # Additional info in expander
    with st.expander("ℹ️ About this dataset"):
        st.write(f"**Total anime in database:** {len(df)}")
        st.write(f"**Rating range:** {df['rating'].min():.1f} - {df['rating'].max():.1f}")
        st.write(f"**Year range:** {df['year'].min()} - {df['year'].max()}")
        st.write("**Note:** This is a demo with mock data. In production, you would use a real anime database.")

if __name__ == "__main__":
    main()