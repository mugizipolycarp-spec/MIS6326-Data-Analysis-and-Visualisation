# Netflix Netflix Movies and TV Shows Analyis
# Mugizi Polycarp

import pandas as pd
from collections import Counter

# 1st FUNCTION to Load and clean the dataset (removes missing values)

def load_dataset(filepath):    
    df = pd.read_csv(filepath)
    
    # Drop rows missing values
    df.dropna(subset=['title', 'rating', 'release_year', 'listed_in'], inplace=True)
    return df

# 2nd FUNCTION to Analyze genre ratings distribution

def analyze_ratings(df): #uses a dictionary to group ratings into categories
                          
    # Define rating groups using a dictionary
    rating_groups = {
        'Kids': ['G', 'PG', 'TV-Y', 'TV-Y7', 'TV-G', 'TV-PG'],
        'Teens': ['PG-13', 'TV-14'],
        'Adults': ['R', 'NC-17', 'TV-MA']
    }

    # Initialize counts for each category
    group_counts = {'Kids': 0, 'Teens': 0, 'Adults': 0, 'Unclassified': 0}

    # Loop through ratings and assign to a category
    for rating in df['rating']:
        found = False
        for group, values in rating_groups.items():
            if rating in values:
                group_counts[group] += 1
                found = True
                break
        # If rating doesn’t fit any known category
        if not found:
            group_counts['Unclassified'] += 1
   
# CLASS called Show. Represents a Netflix Show with methods to check if it is family-friendly

class Show:
   
    def __init__(self, title, show_type, rating, genre, release_year):
        self.title = title
        self.show_type = show_type
        self.rating = rating
        self.genre = genre
        self.release_year = release_year

    def is_family_friendly(self):
        """Return True if rating is family-friendly."""
        family_ratings = ['G', 'PG', 'TV-G', 'TV-Y', 'TV-Y7', 'TV-PG']
        return self.rating in family_ratings

    def __repr__(self):
        return f"{self.title} ({self.release_year}) - {self.rating}"


if __name__ == "__main__":
    
    # Load dataset from your local path
    filepath = r"D:\Msc\datasets\1\netflix_titles.csv"
    df = load_dataset(filepath)

    # Analyze top genres
    all_genres = [genre for sublist in df['listed_in'].str.split(', ') for genre in sublist]
    genre_counts = Counter(all_genres)
    top_genres = genre_counts.most_common(10)

    print("Top 10 Netflix Genres:")
    for genre, count in top_genres:
        print(f"{genre}: {count}")

    # Analyze release years
    print("\n Number of Titles by Release Year:")
    release_counts = df['release_year'].value_counts().sort_index()
    for year, count in release_counts.tail(10).items():  # show last 10 years
        print(f"{year}: {count}")

    # Analyze ratings using dictionary grouping
    rating_summary = analyze_ratings(df)
    print("\n🎬 Rating Groups Summary:")
    print(rating_summary)

    # Create list of Show objects and filter family-friendly ones
    shows = [
        Show(row['title'], row['type'], row['rating'], row['listed_in'], row['release_year'])
        for _, row in df.iterrows()
    ]

    family_friendly = [s for s in shows if s.is_family_friendly()]

    # Output results using loops and if-statement
    print(f"\n Total Family-Friendly Shows: {len(family_friendly)}")
    print("Sample Family-Friendly Shows:")
    for i, show in enumerate(family_friendly[:10]):
        if show.is_family_friendly():
            print(f"{i+1}. {show}")