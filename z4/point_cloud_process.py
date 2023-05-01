import open3d as o3d
import numpy as np
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import cv2

matplotlib.use("Qt5Agg")

PLY_PTH = "/home/d618/.dev/pvso-bf/z4/open3d_data/extract/RedwoodIndoorLivingRoom1/livingroom.ply"
NW_PTH = "output.pcd"
KITCHEN = "TLS_kitchen.ply"

# %% Sample point cloud
ply = o3d.io.read_point_cloud(KITCHEN)
o3d.visualization.draw_geometries([ply])

# %% Nacitanie mracna bodov
pcd = o3d.io.read_point_cloud(KITCHEN)
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)

# %% RANSAC
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01, ransac_n=3, num_iterations=1000
)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# %% DBSCAN
labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))
cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])


segment_models = {}
segments = {}
max_plane_idx = 20
rest = pcd
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    segments[i] = rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass", i, "/", max_plane_idx, "done.")

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest])

labels = np.array(segments[i].cluster_dbscan(eps=0.01 * 10, min_points=10))

candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]

best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])

rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(
    list(np.where(labels != best_candidate)[0])
)
segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))

labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])


# %% K-MEANS

pcd = o3d.io.read_point_cloud(KITCHEN)

# Convert point cloud to a NumPy array
points = np.asarray(pcd.points)

# Apply k-means clustering
n_clusters = 10  # Change this value to set the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)

# Assign each point to a cluster
labels = kmeans.labels_

# Visualize the clustering result
clusters = [o3d.geometry.PointCloud() for _ in range(n_clusters)]

for i, label in enumerate(labels):
    clusters[label].points.append(points[i])

# Set random colors for each cluster
for cluster in clusters:
    color = np.random.rand(3)
    cluster.colors = o3d.utility.Vector3dVector(
        np.tile(color, (len(cluster.points), 1))
    )

# Display the clustered point cloud
o3d.visualization.draw_geometries(clusters)
# %%
