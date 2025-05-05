
# pairwise apke feature me similarity nikalne k liye use hoti hai iski aik sub library hai cosine similarity
# har datatype alag alag hogi tu uske liye alag alag vectorizer use hoti hai
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Upload your dataset first using the upload button in Colab or save as 'movies.csv'
 # Make sure the dataset is uploaded here
df = pd.read_csv('movies.csv')

# Fix column names if needed (strip whitespaces)
df.columns = [col.strip() for col in df.columns]

# Clean numeric columns with '$', commas, or spaces
def clean_currency(val):
    if isinstance(val, str):
        return float(val.replace('$', '').replace(',', '').strip())
    return val

# Convert numeric columns properly
numeric_cols = ['Audience score %', 'Profitability', 'Rotten Tomatoes %', 'Worldwide Gross']
for col in numeric_cols:
    df[col] = df[col].apply(clean_currency)

# Fill missing values
df = df.fillna({
    'Genre': '',
    'Lead Studio': '',
    'Audience score %': 0,
    'Profitability': 0,
    'Rotten Tomatoes %': 0,
    'Worldwide Gross': 0,
    'Year': 0
})

# Normalize numeric values
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Merge text-based features
def combine_features(row):
    return f"{row['Genre']} {row['Lead Studio']} {int(row['Year'])}"

df['combined_features'] = df.apply(combine_features, axis=1)

# Apply TF-IDF on combined text
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Combine text and numerical features
combined_matrix = np.hstack([tfidf_matrix.toarray(), df[numeric_cols].values])

# Compute cosine similarity
cosine_sim = cosine_similarity(combined_matrix)

# Recommendation function
def get_recommendations(title, df=df, cosine_sim=cosine_sim):
    if title not in df['Film'].values:
        return f"Movie '{title}' not found in dataset."
    
    idx = df[df['Film'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['Film'].iloc[movie_indices].tolist()

# Example usage
movie_title = "Twilight"  # Change this to a title from your dataset
print(f"\nTop 5 movies similar to '{movie_title}':")
recommendations = get_recommendations(movie_title)
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")