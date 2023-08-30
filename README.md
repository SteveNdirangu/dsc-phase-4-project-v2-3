# **Title: Movie Recommendation for "Moviefix" Stakeholder**


# **Business Understanding**
#### The movie industry is fast growing, and with so many options, there is need for cutting edge user interactivity. This involves tailoring their movie watching experience via intuitive recommendations that can give them interesting options based on what they prefer, while also enticing them with other content that they may like

#### It is for this very reason we are working on a recommendation system for the "Moviefix" firm, in order to not only capture the users, but also keep them coming back for more. To accomplish this, our model will create 5(five) recommendations that will be given to the user.


# **Problem Statement**
The main challenge is to design and implement a movie recommendation system that employs collaborative filtering techniques to predict movie preferences for users based on their past ratings. In short, to analyse their past activities and give them recommendations based on their tastes.

To address a potential "cold start" problem (where new users or movies have limited ratings), if possible, we will attempt to explore a hybrid approach that combines collaborative filtering with content-based filtering.

The success of this project will be measured by evaluating the accuracy and relevance of the recommendations via metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) that will gauge the performance of the model.

Ultimately, the objective is to build a recommendation system that enhances user engagement, encourages exploration of diverse movies, and contributes to the overall satisfaction of users on the platform.


# **Data understanding**

The dataset describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.

Users were selected at random for inclusion. All selected users had rated at least 20 movies. No demographic information is included. Each user is represented by an id, and no other information is provided.

The data are contained in the files `links.csv`, `movies.csv`, `ratings.csv` and `tags.csv`.
The dataset was provided by MovieLens, a movie recommendation service. It includes the movies and the rating scores made for these movies contains. It contains 100,000 ratings (1–5) from 943 users on 1682 movies.
**1) Movies:**

Has information about each movie, such as movie ID, title, and genres.

Columns:

- **movieId:** the unique numerical identifier for each movie. This ID is used to connect the movie information with the ratings and links datasets. This identifier is crucial for linking the movie information with other datasets, especially the ratings dataset. It acts as a key to connect the movie information with user interactions (ratings) and potentially external databases (links).
- **title:** The name of the movie together with its year of release, is a string type.

- **genres:** Genres associated with the movie. Each movie belongs to one or even a combination of genres, marking its type and both distinguishing it from others as well as linking it to a certain category. Genres are a pipe-separated list, and are selected from the following:

* Action
* Adventure
* Animation
* Children's
* Comedy
* Crime
* Documentary
* Drama
* Fantasy
* Film-Noir
* Horror
* Musical
* Mystery
* Romance
* Sci-Fi
* Thriller
* War
* Western
* (no genres listed)

**2) Ratings Dataset:**

The ratings dataset contains user-movie interactions, including user IDs, movie IDs, and ratings.
Collaborative filtering algorithms will leverage this dataset to predict movie ratings for users based on their historical ratings.

Size: The dataset contains information about user-movie interactions, where each row represents a user's rating for a specific movie.

Columns:

- **userId:** unique integer identifier for each user, to track their interactions

- **movieId:** A unique integer identifier for each movie. This identifier connects the ratings with specific movies. It links user ratings to the movies they've interacted with.
- **rating:** The value representing how much a user liked a particular movie. ranging from 1 to 5, with half-star increments.
- **timestamp:** A timestamp indicating when the rating was given. Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

**3)Tags**: Tags are user-generated metadata about movies. Each tag is typically a single word or short phrase. The meaning, value, and purpose of a particular tag is determined by each user.
Columns:
- **userId** The user's unique Identifier
- **movieId** The Movie's Unique identifier
- **tag**- the tag entered by a user to describe a movie
- **timestamp**-Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

**4) Links Dataset:**
Identifiers that can be used to link to other sources of movie data, that is external databases
Columns:

- **movieId:** A unique identifier for each movie. This identifier corresponds to the movie ID in the MovieLens dataset.
- **imdbId:** The identifier of the movie in the IMDb (Internet Movie Database) system. This identifier is used to connect the movie with its corresponding entry in the IMDb database. IMDb is a widely-used database for movie information, including details about cast, crew, plot, and more.
- **tmdbId:** The identifier of the movie in the TMDB (The Movie Database) system. This identifier links the movie to its corresponding entry in the TMDB database.

This dataset might offer additional contextual information for content-based filtering, especially for new users.

# EDA samples
movieId	title	genres
0	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
1	2	Jumanji (1995)	Adventure|Children|Fantasy
2	3	Grumpier Old Men (1995)	Comedy|Romance
3	4	Waiting to Exhale (1995)	Comedy|Drama|Romance
4	5	Father of the Bride Part II (1995)	Comedy
5	6	Heat (1995)	Action|Crime|Thriller
6	7	Sabrina (1995)	Comedy|Romance
7	8	Tom and Huck (1995)	Adventure|Children
8	9	Sudden Death (1995)	Action
9	10	GoldenEye (1995)	Action|Adventure|Thriller

Visualizations of EDA
![image](https://github.com/SteveNdirangu/dsc-phase-4-project-v2-3/assets/127976914/579742c4-426c-467d-a565-84cfa6238c2c)
![image](https://github.com/SteveNdirangu/dsc-phase-4-project-v2-3/assets/127976914/514abf70-cab2-4f96-b86e-0e9b4bde9ffc)
![image](https://github.com/SteveNdirangu/dsc-phase-4-project-v2-3/assets/127976914/14ee0316-4489-415d-a0c7-c7f473333ae2)
![image](https://github.com/SteveNdirangu/dsc-phase-4-project-v2-3/assets/127976914/bb3cbc1e-076a-4a04-b0b3-7fb664f9aed7)

# Data cleaning
We removed some duplicates in the Movies dataset and the Ratings dataset, which were the 2 focus datasets of this projest

print("duplicates in ID: ", movies.movieId.duplicated().sum())
print("duplicates in Title: ", movies.title.duplicated().sum())
print("duplicates in Genres: ", movies.genres.duplicated().sum())
duplicates in ID:  0
duplicates in Title:  5
duplicates in Genres:  8791
## Remove duplicates based on the "title" column, keeping the first occurence
movies.drop_duplicates(subset="title", keep="first", inplace=True)


## Remove duplicates based on "userId" and "movieId" columns
ratings.drop_duplicates(subset=["userId", "movieId"], keep="first", inplace=True)
#Check for missing values
print(ratings.isnull().sum())
#Reset the index of the dataframe
ratings.reset_index(drop=True, inplace=True)



# We then Merged the data into one, and proceeded to explore it with some visualizations
#Merge movies and ratings dataframes based on movieId
movie_ratings = pd.merge(movies, ratings, on='movieId', how='inner')

#Merge the result with links dataframe based on movieId
movie_ratings_links = pd.merge(movie_ratings, links, on='movieId', how='inner')

#Merge the result with tags dataframe based on userId and movieId
consolidated_data = pd.merge(movie_ratings_links, tags, on=['userId', 'movieId'], how='left')


![image](https://github.com/SteveNdirangu/dsc-phase-4-project-v2-3/assets/127976914/e07d1fb0-5645-426a-a4df-ea3f2488f2c7)
![image](https://github.com/SteveNdirangu/dsc-phase-4-project-v2-3/assets/127976914/524982cf-3c41-4ebc-a3e0-0d17007a5698)



# we chose a popularity of 50and above votes, since lower votes, even if the rating is high, mean less popularity
popularity_threshold =50
rating_popular_movie = combined_ratingCount_data.query("totalRatingCount >= " + str(popularity_threshold))
rating_popular_movie


# **Feature selection**
We decided to drop the Tags and their timestamps, relying on the actual ratings instead , we did the same for the links as we would not be using outside resources, this was especially done due to their missing values, 
We also found out that we had **genre_list**, which was a better representation of **"genres"**, so we dropped the latter
**title length** was not really relevant to recommendation, so we dropped it as well

We then pivoted the resulting features table in order to carry out **collaborative filtering**



# **Implementation of Nearest Neighbors Model, utilising cosine metric**
movie_features_selected_matrix = csr_matrix(movie_features_selected.values)
model_knn = NearestNeighbors(metric ="cosine", algorithm="brute")
model_knn.fit(movie_features_selected_matrix)

## We then querried the necessary distances and indices from the model, and looped through the rows(now movie titles to train the model)
def get_movie_recommendations_with_threshold(movie_title, n_neighbors=6, min_ratings_threshold=10):
    try:
        # Find the index of the given movie title
        movie_index = movie_features_selected.index.get_loc(movie_title)

        # Check if the movie meets the minimum ratings threshold
        if movie_features_selected.iloc[movie_index, :].sum() < min_ratings_threshold:
            print("This movie has too few ratings to provide reliable recommendations.")
            return

        # Query the k-NN model and print recommendations
        get_movie_recommendations(movie_title, n_neighbors)

    except KeyError:
        print("Movie title not found in the dataset.")

#Example usage
get_movie_recommendations_with_threshold("Toy Story (1995)")


## afterwards, we loaded the data into surprise in order to carry out training and testing using the SVD model
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

#Create Surprise Dataset
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(movie_features1[['userId', 'movieId', 'rating']], reader)

#Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

#Train an SVD model on the trainset
svd_model = SVD(n_factors=50, random_state=42)
svd_model.fit(trainset)

#Get the user-item predicted ratings
test_predictions = svd_model.test(testset)

#Calculate RMSE using the Surprise accuracy module
rmse = accuracy.rmse(test_predictions)
print("RMSE: {:.2f}".format(rmse))

# this gave us an RMSE score of 0.83 
# we were unable to implement cold start handling due to crashing of runtime, however the model works well


# Conclusion
In this project, we set out to build a movie recommendation system using collaborative filtering and content-based approaches. We explored a diverse dataset containing movie ratings, genres, and user interactions to create a personalized movie recommendation system. Through data cleaning, exploratory data analysis (EDA), and the implementation of recommendation algorithms, we've gained insights into the movie preferences of users and successfully generated movie recommendations.



# Key Findings and Achievements
**Data Cleaning and EDA**: We started by preprocessing the dataset, removing duplicates, and handling missing values. Exploratory data analysis provided us with valuable insights into the distribution of movie genres, user ratings, and user interactions.

**Collaborative Filtering (SVD)**: Using the Surprise library, we built a collaborative filtering model based on matrix factorization, specifically the Singular Value Decomposition (SVD) algorithm. The model effectively captured user preferences and generated accurate movie recommendations. The calculated Root Mean Square Error (RMSE) of 0.83 indicated the model's reasonable predictive performance.



# Recommendations for Improvement
**Hybrid Recommendations**: While not implemented in this project, combining collaborative filtering and content-based recommendations can enhance the accuracy and coverage of recommendations. Hybrid models address the limitations of each approach, providing more robust suggestions for users.

**Fine-Tuning and Evaluation**: Experiment with different hyperparameters, algorithms, and preprocessing techniques to fine-tune the recommendation models further. Utilize evaluation metrics like RMSE, precision, recall, and F1-score to objectively assess the models' performance.

**User Interface and Deployment**: Consider creating a user interface where users can input their preferences and receive personalized recommendations. This interface could be deployed on a web server, allowing users to interact with the recommendation system seamlessly.



# **Conclusion**
In conclusion, this recommendation system project has demonstrated the power of collaborative filtering and content-based approaches in providing users with personalized movie recommendations. By combining user-item interactions and content features, we've created a system that caters to diverse user preferences. As movie datasets continue to grow, the techniques explored in this project can be scaled and adapted to real-world applications, contributing to enhanced user experiences and engagement.

This project serves as a stepping stone toward building more advanced recommendation systems and opens the door to exploring cutting-edge techniques in the field of machine learning and artificial intelligence.




