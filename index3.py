# Task 4 - Recommendation System using SVD (Collaborative Filtering)


#  Install surprise library if not already installed
# !pip install scikit-surprise

#from surprise import SVD, Dataset, Reader
#from surprise.model_selection import train_test_split, accuracy


#  Load built-in dataset
data = Dataset.load_builtin('ml-100k')

#  Train-Test Split
trainset, testset = train_test_split(data, test_size=0.25)

# Use the SVD algorithm
model = SVD()
model.fit(trainset)

#  Make Predictions
predictions = model.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

#  Sample recommendation for a user
from collections import defaultdict

def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_recommendations = get_top_n(predictions, n=5)

# Print top 5 recommendations for a sample user
sample_user = list(top_recommendations.keys())[0]
print(f"Top 5 Recommendations for User {sample_user}:")
for item_id, score in top_recommendations[sample_user]:
    print(f"Movie ID: {item_id}, Predicted Rating: {score:.2f}")
