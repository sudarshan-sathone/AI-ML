# Day 15 Report: Unsupervised Learning & User Segmentation

## Overview
For Day 15, the task was to explore unsupervised learning using the K-Means clustering algorithm. The goal was to identify hidden patterns in data without using any predefined labels and group similar data points together.

---

## Approach

### 1. Dataset
A synthetic dataset was generated using `make_blobs` to simulate user behavior patterns. The dataset consisted of 300 data points distributed across 5 natural clusters.

---

### 2. Finding Optimal Number of Clusters (Elbow Method)
To determine the best number of clusters (K), the Elbow Method was used. The Within-Cluster Sum of Squares (WCSS) was calculated for values of K ranging from 1 to 10.

The graph showed a clear "elbow" at **K = 5**, indicating that 5 clusters would be optimal for this dataset.

---

### 3. Applying K-Means Clustering
K-Means was applied with:
- Number of clusters: 5
- Initialization method: `k-means++`

The algorithm successfully grouped the data into 5 distinct clusters.

---

### 4. Visualization
The clusters were visualized using a scatter plot:
- Each cluster was represented with different labels
- Cluster centroids were highlighted separately

This made it easy to observe how the data points were grouped.

---

### 5. Stability Test (Experiment)

K-Means was run multiple times using different initialization methods:
- `k-means++`
- `random`

### Observation:
The cluster centers remained consistent across runs, with only minor differences in ordering.

### Conclusion:
This indicates that:
- The dataset is well-structured
- K-Means converges to similar solutions even with random initialization in this case

However, generally:
- `k-means++` is preferred as it selects better initial centroids
- It reduces randomness and improves stability

---

## Key Learnings

- K-Means clustering helps discover hidden patterns in unlabeled data
- Choosing the correct value of K is important for meaningful clustering
- The Elbow Method is a simple and effective way to find optimal K
- Initialization plays a key role in clustering performance
- Visualization helps in understanding how clusters are formed

---

## Reflection

If clustering identifies users who attend late-night events and prefer high-intensity sports, this insight can be used to improve recommendations on the MeetMux platform.

Instead of generic suggestions, the system can recommend:
- Late-night sports events
- Fitness-based meetups
- High-energy group activities

This makes the platform more personalized and relevant for users. As a result, users are more likely to engage with events that match their interests, leading to better user satisfaction and higher activity on the platform.

---

## Conclusion

This task provided a clear understanding of how unsupervised learning works in real scenarios. It demonstrated how clustering can be used to segment users and improve decision-making in applications like recommendation systems.