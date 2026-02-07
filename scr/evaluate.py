import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import get_recommendations

import pandas as pd

user_id = 4
top_movies = get_recommendations(userId=user_id, N=10, alpha=0.5)

print(f"Top 10 recommended movies for user {user_id}:\n")
print(top_movies)