#reduce dimensionality on articles 
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data_dir = "./cleaned_articles"
all_documents = []
for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        print(f"Processing file: {filename}")
        try:
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
                articles = json.load(file)
                for article in articles:
                    content = article.get("Content") or article.get("content")
                    if content:
                        cleaned_content = re.sub(r"\\u[\dA-Fa-f]{4}", " ", content)
                        cleaned_content = re.sub(r"\s+", " ", cleaned_content)
                        cleaned_content = cleaned_content.strip()  
                        
                        all_documents.append(cleaned_content)
        except json.JSONDecodeError as e:
            print(f"Error processing {filename}: {e}")

print(f"collected {len(all_documents)} articles")
#all_documents = all_documents[:100]
#1 initialize centroids 
# while len(centroids) < k:
#     point = random.choice(points)
#     if point not in centroids: 
#         centroids.append(point)
#         clusters[point] = []
vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
    max_features=1000 
)

t0 = time.time()
X_tfidf = vectorizer.fit_transform(all_documents)
print(f"vectorization took {time.time() - t0:.3f} seconds")
print(f"number of documents: {X_tfidf.shape[0]}")

svd = TruncatedSVD(n_components=2)
X_reduced = svd.fit_transform(X_tfidf)

components = svd.components_ 
# components = pca.components_ 
#figure out which words contribute to the x and y axis 
feature_names = vectorizer.get_feature_names_out()
num_top_words = 10

for i, component in enumerate(components):
    top_indices = component.argsort()[-num_top_words:][::-1]
    bottom_indices = component.argsort()[:num_top_words]
    
    print(f"\nComponent {i + 1} Top Words:")
    print("Top Positive Words:")
    for index in top_indices:
        print(f"{feature_names[index]}: {component[index]:.4f}")
    
    print("Top Negative Words:")
    for index in bottom_indices:
        print(f"{feature_names[index]}: {component[index]:.4f}")


k = 4
kmeans = KMeans(n_clusters=k, n_init=5, max_iter=3, random_state=42, verbose=1)
kmeans.fit(X_reduced)
labels = kmeans.labels_
# for point in points:
#     for i, expoint in enumerate(extremes):
#         if point not in centroids and distance(point, expoint) < distance(centroids[i], expoint):
#             centroids[i] = point

# for point in points: 
#     nearest = centroids[0]
#     for centroid in centroids:
#         if distance(point, centroid) < distance(point, nearest):
#             nearest = centroid
#     clusters[nearest].append(point)

reduced_2D = X_reduced[:, :2]
plt.scatter(reduced_2D[:, 0], reduced_2D[:, 1], c=labels, cmap='viridis', marker='o')
plt.title(f'K-Means Clustering (k={k})')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar()
plt.show()

#pca = PCA(n_components=2)
#X_reduced = pca.fit_transform(X_tfidf)
#plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
#plt.show()

'''
output:
Processing file: cleaned_the_verge.json
Processing file: ksl_articles_9-27.json
Processing file: cleaned_cbs.json
Processing file: ksl_articles_9-26.json
Processing file: cleaned_cnbc.json
Processing file: cleaned_ksl_articles_9-24.json
Processing file: cleaned_abc.json
Collected 452 articles.
Vectorization done in 0.161 seconds.
Number of samples (documents): 452

Component 1 Top Words:
Top Positive Words:
ai: 0.3429
company: 0.1666
meta: 0.1395
apple: 0.1310
openai: 0.1189
google: 0.1172
amazon: 0.1084
microsoft: 0.0949
people: 0.0936
just: 0.0935
Top Negative Words:
espresso: 0.0032
ounce: 0.0053
earbuds: 0.0054
cup: 0.0054
ultra: 0.0059
simple: 0.0061
rich: 0.0061
monitor: 0.0063
wireless: 0.0063
portable: 0.0064

Component 2 Top Words:
Top Positive Words:
ai: 0.4440
apple: 0.1838
meta: 0.1551
google: 0.1378
iphone: 0.1169
video: 0.1086
microsoft: 0.1012
feature: 0.0931
features: 0.0901
copilot: 0.0889
Top Negative Words:
trump: -0.2186
harris: -0.1853
court: -0.1156
president: -0.1090
cbs: -0.1036
election: -0.1017
news: -0.0979
hurricane: -0.0920
state: -0.0873
israel: -0.0822
Initialization complete
Iteration 0, inertia 4.997287631031201.
Iteration 1, inertia 3.4149792564384893.
Iteration 2, inertia 3.264986579564695.
Initialization complete
Iteration 0, inertia 4.232484262349725.
Iteration 1, inertia 3.3297140503372686.
Iteration 2, inertia 3.206396008273561.
Initialization complete
Iteration 0, inertia 4.08572890152711.
Iteration 1, inertia 3.4533798221893193.
Iteration 2, inertia 3.232867509727416.
Initialization complete
Iteration 0, inertia 4.61776620960281.
Iteration 1, inertia 3.2176534529834413.
Iteration 2, inertia 3.16050603014733.
'''