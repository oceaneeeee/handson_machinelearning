from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
X_digits, y_digits = load_digits(return_X_y=True)
X_train, y_train, X_test, y_test = train_test_split(X_digits, y_digits)
k = 50
kmeans = KMeans(n_clusters=k)
X_digits_dist = kmeans.fit_transform(X_train)
print(f"X_digits_dist={X_digits_dist}")
print(f"len X digits dist = {X_digits_dist.shape}")
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]
y_representative_digits = np.array([4,8,0,6,8,3,7,7,9,2,5,5,8,5,2,1,2,9,6,1,1,6,9,0,8,3,0,7,4,1,6,5,2,4,1,8,6,3,9,2,4,2,9,4,7,6,2,3,1,1])
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
print(kmeans.labels_)
print(y_train_propagated[:10])
print(kmeans.labels_==1)
print(y_train_propagated[kmeans.labels_==1][:10])
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
print(y_train_propagated)
print(np.arange(len(X_train)))
X_cluster_dist = X_digits_dist[np.arange(len(X_train)),kmeans.labels_]
print(f"X_cluster dist = {X_cluster_dist}")
print(f"X digit dist = {X_digits_dist}")
print(f"X cluster dist shape = {X_cluster_dist.shape}")
