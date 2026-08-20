import numpy as np
import os
import gc
import pandas as pd
from scipy.spatial.distance import cdist
from cluster_som_utils import *
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import RegularPolygon
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


class ClusterSOM:
            def __init__(self, x, y, input_dim, learning_rate=0.1, radius=1.0, topology="hexagonal", weights_file="weights.txt", reshape_weights=True):
                self.x = x
                self.y = y
                self.input_dim = input_dim
                self.learning_rate = learning_rate
                self.radius = radius
                self.topology = topology
                if isinstance(weights_file, np.ndarray):
                    self.weights = weights_file
                else:
                    if not os.path.isfile(weights_file):
                        raise FileNotFoundError(f"Please provide the weights file or the correct weights file path.")
                    loaded_weights = np.loadtxt(f'{weights_file}', skiprows=2)   
                    if reshape_weights:
                        self.weights = np.reshape(loaded_weights, (x, y, input_dim))
                    else:
                        self.weights = loaded_weights

            
            def KMeans(self, optimization=True, score_type='silhouette', method='Kneedle', folder=None, plot=False, nc=None, clr=None, random_state=42):
                """
                Performs KMeans clustering on the SOM weights.
                Args:
                    optimization (bool): If True, automatically determines the optimal number of clusters.
                    score_type (str): Metric used for cluster evaluation (default: 'silhouette').
                    method (str): Method for optimal cluster selection (default: 'Kneedle').
                    folder (str, optional): Directory to save clustering results CSV.
                    plot (bool): If True, plots clustering results.
                    nc (int, optional): Number of clusters (used if optimization is False).
                    clr (list, optional): List of colors for plotting clusters.
                    random_state (int): Random seed for reproducibility.
                Returns:
                    tuple: (number of clusters, cluster labels, KMeans model, evaluation score)
                """

                if clr is None:
                    clr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
                reshaped_weights = self.weights.reshape(self.x*self.y, self.input_dim)

                if optimization:
                    opti_n_clust = NumOptiClust(reshaped_weights, method, clr, folder, plot)
                    print("Number of optimal cluster is", opti_n_clust, "proceeding with clustering...")
                    clustering_fit = KMeans(n_clusters=opti_n_clust, random_state=random_state, init='k-means++').fit(reshaped_weights)
                    self.clustering_fit = clustering_fit
                else:
                    clustering_fit = KMeans(n_clusters=nc, random_state=random_state, init='k-means++').fit(reshaped_weights)
                    self.clustering_fit = clustering_fit
                
                self.clustering_predict = clustering_fit.predict(reshaped_weights)
                clustering = pd.DataFrame(self.clustering_predict, columns=['ClusterNumber'])

                if folder is not None:
                    clustering.to_csv(folder + "/clustering_results.csv", index=False)
                    print("Clustering done, clustering results saved in " + folder + ", proceeding with evaluation...")

                evaluation = KMeansEvaluation(reshaped_weights, self.clustering_predict, score_type)
                print('Done')
                
                return clustering_fit.n_clusters, self.clustering_predict, clustering_fit, evaluation

            def get_bmus(self, new_data, nb=1, verbose=True, memory_fraction=0.25):
                """
                Finds the Best Matching Units (BMUs) for the given input data.
                Parameters:
                    new_data (np.ndarray): Input data samples to map to BMUs.
                    nb (int, optional): Number of BMUs to return per sample (1 or 2). Default is 1.
                    verbose (bool, optional): If True, prints batch size information. Default is True.
                    memory_fraction (float, optional): Fraction of memory to use for batch processing. Default is 0.25.
                Returns:
                    np.ndarray or tuple: Indices of BMUs for each sample. If nb == 1, returns a single array.
                                         If nb != 1, returns a tuple of arrays (bmu1, bmu2).
                """

                reshaped_weights = self.weights.reshape(self.x*self.y, self.input_dim)
                n_nodes = self.x * self.y
                batch_size = estimate_batch_size(n_nodes, max_mem_frac=memory_fraction)
                if verbose:
                    print(f"Using batch size: {batch_size}")

                bmu1 = np.empty(new_data.shape[0], dtype=np.int32)
                if nb != 1:
                    bmu2 = np.empty(new_data.shape[0], dtype=np.int32)

                for i, start in enumerate(range(0, new_data.shape[0], batch_size)):
                    end = min(start + batch_size, new_data.shape[0])
                    batch = new_data[start:end]
                    dists = cdist(batch, reshaped_weights, metric='euclidean')
                    sorted_idx = np.argsort(dists, axis=1)
                    bmu1[start:end] = sorted_idx[:, 0]
                    if nb != 1:
                        bmu2[start:end] = sorted_idx[:, 1]
                    del batch, dists, sorted_idx
                    gc.collect()

                if nb == 1:
                    return bmu1
                else:
                    return bmu1, bmu2
                
            def get_hits(self, new_data):
                bmus = self.get_bmus(new_data)
                hits = np.zeros((self.x, self.y), dtype=int)
                for i in range(self.x * self.y):
                    hits[i // self.y, i % self.y] = np.sum(bmus == i)
                return hits

            def predict(self, new_data, bmu_indices=None, scaler_fit_old=None, scaler_fit_new=None):  
                """
                Predict the cluster labels for new data using the trained Self-Organizing Map (SOM).
                Parameters:
                new_data (array-like): An array where each row represents a data point to be clustered.
                Returns:
                list: A list of cluster labels corresponding to each data point in new_data.
                Raises:
                ValueError: If the KMeans clustering has not been run before calling this method.
                """

                if scaler_fit_new is not None:
                    reshaped_weights = self.weights.reshape(self.x*self.y, self.input_dim)
                    reshaped_weights = scaler_fit_old.inverse_transform(reshaped_weights)
                    reshaped_weights = scaler_fit_new.transform(reshaped_weights)
                    self.weights = reshaped_weights.reshape(self.x, self.y, self.input_dim)
                else:
                    reshaped_weights = self.weights.reshape(self.x*self.y, self.input_dim)

                if bmu_indices is None:
                    bmu_indices = self.get_bmus(new_data)
                else:
                    bmu_indices = bmu_indices

                if not hasattr(self, 'clustering_predict'):
                    raise ValueError("You need to run a clustering method (e.g. KMeans()) before calling predict().")
                
                cluster_labels = []
                for bmu_index in bmu_indices:
                    if isinstance(bmu_index, (tuple, list, np.ndarray)) and len(bmu_index) == 2:
                        clustering_predict = self.clustering_predict.reshape(self.x, self.y)
                        cluster_labels.append(clustering_predict[bmu_index[0], bmu_index[1]])
                    else:
                        cluster_labels.append(self.clustering_predict[bmu_index])
                
                return cluster_labels
            
            def get_quality_measures(self, data, memory_fraction):
                """
                Calculates topographic and quantization errors for the SOM.
                Args:
                    data (np.ndarray): Input data samples.
                    memory_fraction (float): Fraction of memory to use for BMU computation.
                Returns:
                    tuple: (topographic_error, quantization_error)
                        - topographic_error (float): Proportion of samples with non-adjacent BMUs.
                        - quantization_error (float): Average distance between data samples and their BMUs.
                """
                
                n_nodes = self.x * self.y
                bmu1, bmu2 = self.get_bmus(data, nb=2, memory_fraction=memory_fraction)
                coords = hex_coords_axial(self.x, self.y)[:n_nodes]
                bmu1_coords = coords[bmu1]
                bmu2_coords = coords[bmu2]

                dist = hex_distance(bmu1_coords, bmu2_coords)

                topographic_error = np.mean(dist > 1)
                print(f"Topographic error: {topographic_error:.4f}")
                quantization_error = np.mean(np.linalg.norm(data - self.weights[bmu1], axis=1))
                print(f"Quantization error: {quantization_error:.4f}")

                return topographic_error, quantization_error

            def plot_umatrix(self, save=False, folder=None):
                """
                Plots the U-Matrix (Unified Distance Matrix) of the SOM, visualizing the average distance between neighboring neurons.
                Parameters
                ----------
                save : bool, optional
                    If True, saves the plot as 'umatrix.png' in the specified folder.
                folder : str, optional
                    Path to the folder where the plot will be saved if 'save' is True.
                Notes
                -----
                - Supports both rectangular and hexagonal topologies.
                - Displays a colorbar indicating the average neighborhood distance.
                """
                
                um = np.zeros((self.weights.shape[0],
                            self.weights.shape[1],
                            8))  # 2 spots more for hexagonal topology

                ii = [[0, -1, -1, -1, 0, 1, 1, 1]]*2
                jj = [[-1, -1, 0, 1, 1, 1, 0, -1]]*2

                if self.topology == 'hexagonal':
                    ii = [[1, 1, 1, 0, -1, 0], [0, 1, 0, -1, -1, -1]]
                    jj = [[1, 0, -1, -1, 0, 1], [1, 0, -1, -1, 0, 1]]

                for x in range(self.weights.shape[0]):
                    for y in range(self.weights.shape[1]):
                        w_2 = self.weights[x, y]
                        e = y % 2 == 0   # only used on hexagonal topology
                        for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                            if (x+i >= 0 and x+i < self.weights.shape[0] and
                                    y+j >= 0 and y+j < self.weights.shape[1]):
                                w_1 = self.weights[x+i, y+j]
                                um[x, y, k] = np.linalg.norm(w_2-w_1)

                um = um.sum(axis=2)

                if self.topology == 'hexagonal':
                        norm = Normalize(vmin=um[:,:].min(), vmax=um[:,:].max())

                        xx, yy = np.meshgrid(range(self.weights.shape[1]), range(self.weights.shape[0]))
                        xx, yy = 1.*xx, 1.*yy
                        for i, k in enumerate(xx):
                            if i % 2 == 1:
                                xx[i] += 0.5

                        f = plt.figure(figsize=(12,12*(np.sqrt(3) / 2)))
                        ax = f.add_subplot(111)
                        ax.set_aspect('equal')

                        # iteratively add hexagons
                        for i in range(self.weights.shape[0]):
                            for j in range(self.weights.shape[1]):
                                wy = yy[(i, j)] * np.sqrt(3) / 2
                                hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/ np.sqrt(3), facecolor=cm.bone_r(norm(um[i, j])), alpha=.9, edgecolor='black', linewidth=0.05)
                                ax.add_patch(hex)

                        # Create a colorbar
                        cmap = cm.bone_r
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])
                        # Add the colorbar to the plot
                        cbar = plt.colorbar(sm, ax=ax, pad=0.06, fraction=0.04, extend='both')
                        cbar.set_label('Average Neighbourhood Distance')
                        xrange = np.arange(self.weights.shape[0])
                        yrange = np.arange(self.weights.shape[1])
                        ax.set_xticks(xrange-.5, xrange+1)
                        ax.set_yticks(yrange * np.sqrt(3) / 2, yrange+1)
                        ax.set_xlim(-1, self.weights.shape[1])
                        ax.set_ylim(-0.5, self.weights.shape[0]*(np.sqrt(3) / 2)) 
                        ax.axis('off')
                else:
                    im = plt.imshow(um, cmap='bone_r', origin='lower')
                    cbar = plt.colorbar(im, label='Average Neighbourhood Distance', fraction=0.04, pad=0.06, extend='both')
                
                
                if save:
                    plt.savefig(folder + "/umatrix.png", dpi=300)
                    print("U-Matrix saved in " + folder)
            
            def plot_clustered_map(
                self, colors, save=False, folder=None, with_umatrix=False,
                ax=None, cbar=True, cbar_label='Cluster',
                cbar_fraction=0.09, cbar_pad=0.015,
                cbar_labelsize=34, cbar_ticksize=28
            ):
                """
                Plot the clustered map of the Self-Organizing Map (SOM).
                Parameters:
                colors (list): List of colors for clusters.
                save (bool): Whether to save the plot as a .png file.
                folder (str): The folder to save the plot in.
                with_umatrix (bool): Overlay the U-matrix as background.
                """
                if not hasattr(self, 'clustering_predict'):
                    raise ValueError("You need to run a clustering method (e.g. KMeans()) before calling plot_clustered_map().")
                
                num_clusters = np.unique(self.clustering_predict.reshape(self.x, self.y)).size
                colors = colors[:num_clusters]
                cmap = mcolors.ListedColormap(colors)
                norm = mpl.colors.BoundaryNorm(np.arange(num_clusters + 1) - 0.5, num_clusters)

                if ax is None:
                    fig = plt.figure(figsize=(12, 12 * (np.sqrt(3) / 2)))
                    ax = fig.add_subplot(111)
                else:
                    fig = ax.figure

                if self.topology == 'hexagonal':
                    xx, yy = np.meshgrid(range(self.weights.shape[1]), range(self.weights.shape[0]))
                    xx, yy = 1.*xx, 1.*yy
                    for i, k in enumerate(xx):
                        if i % 2 == 1:
                            xx[i] += 0.5

                    ax.set_aspect('equal')

                    if with_umatrix:
                        # Compute U-matrix
                        um = np.zeros((self.weights.shape[0], self.weights.shape[1], 8))
                        ii = [[0, -1, -1, -1, 0, 1, 1, 1]]*2
                        jj = [[-1, -1, 0, 1, 1, 1, 0, -1]]*2
                        if self.topology == 'hexagonal':
                            ii = [[1, 1, 1, 0, -1, 0], [0, 1, 0, -1, -1, -1]]
                            jj = [[1, 0, -1, -1, 0, 1], [1, 0, -1, -1, 0, 1]]
                        for x in range(self.weights.shape[0]):
                            for y in range(self.weights.shape[1]):
                                w_2 = self.weights[x, y]
                                e = y % 2 == 0
                                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                                    if (0 <= x+i < self.weights.shape[0] and 0 <= y+j < self.weights.shape[1]):
                                        w_1 = self.weights[x+i, y+j]
                                        um[x, y, k] = np.linalg.norm(w_2-w_1)
                        um = um.sum(axis=2)
                        norm_um = Normalize(vmin=um[:,:].min(), vmax=um[:,:].max())
                        # Draw U-matrix as background hexagons
                        for i in range(self.weights.shape[0]):
                            for j in range(self.weights.shape[1]):
                                wy = yy[(i, j)] * np.sqrt(3) / 2
                                hex_bg = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                                                        facecolor=cm.bone_r(norm_um(um[i, j])), alpha=.7, edgecolor='black', linewidth=0.05)
                                ax.add_patch(hex_bg)
                        # Overlay clusters with partial transparency
                        for i in range(self.weights.shape[0]):
                            for j in range(self.weights.shape[1]):
                                wy = yy[(i, j)] * np.sqrt(3) / 2
                                hex_fg = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                                                        facecolor=cmap(norm(self.clustering_predict.reshape(self.x, self.y)[i, j])), alpha=.5, edgecolor='black', linewidth=0.05)
                                ax.add_patch(hex_fg)
                    else:
                        for i in range(self.weights.shape[0]):
                            for j in range(self.weights.shape[1]):
                                wy = yy[(i, j)] * np.sqrt(3) / 2
                                hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                                                     facecolor=cmap(norm(self.clustering_predict.reshape(self.x, self.y)[i, j])), alpha=.9, edgecolor='black', linewidth=0.05)
                                ax.add_patch(hex)

                    xrange = np.arange(self.weights.shape[0])
                    yrange = np.arange(self.weights.shape[1])
                    ax.set_xticks(xrange-.5, xrange+1)
                    ax.set_yticks(yrange * np.sqrt(3) / 2, yrange+1)
                    ax.set_xlim(-1, self.weights.shape[1])
                    ax.set_ylim(-0.5, self.weights.shape[0]*(np.sqrt(3) / 2)) 
                    ax.axis('off')

                    if cbar:
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])
                        cbar_obj = fig.colorbar(sm, ax=ax, pad=cbar_pad, fraction=cbar_fraction, extend='both')
                        cbar_obj.set_label(cbar_label, fontsize=cbar_labelsize)
                        cbar_obj.ax.tick_params(labelsize=cbar_ticksize)
                        cbar_obj.set_ticks(np.arange(num_clusters))
                        cbar_obj.set_ticklabels([f'{i+1}' for i in range(num_clusters)])
                else:
                    if with_umatrix:
                        # Compute U-matrix
                        um = np.zeros((self.weights.shape[0], self.weights.shape[1], 8))
                        ii = [[0, -1, -1, -1, 0, 1, 1, 1]]*2
                        jj = [[-1, -1, 0, 1, 1, 1, 0, -1]]*2
                        for x in range(self.weights.shape[0]):
                            for y in range(self.weights.shape[1]):
                                w_2 = self.weights[x, y]
                                e = y % 2 == 0
                                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                                    if (0 <= x+i < self.weights.shape[0] and 0 <= y+j < self.weights.shape[1]):
                                        w_1 = self.weights[x+i, y+j]
                                        um[x, y, k] = np.linalg.norm(w_2-w_1)
                        um = um.sum(axis=2)
                        # Plot U-matrix as background
                        im = ax.imshow(um, cmap='bone_r', origin='lower', alpha=0.7)
                        # Overlay clusters with partial transparency
                        ax.imshow(self.clustering_predict.reshape(self.x, self.y), cmap=cmap, origin='lower', alpha=0.5)
                    else:
                        ax.imshow(self.clustering_predict.reshape(self.x, self.y), cmap=cmap, origin='lower')
                    norm1 = mpl.colors.BoundaryNorm(np.arange(num_clusters + 1) - 0.5, num_clusters)
                    if cbar:
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm1)
                        sm.set_array([])
                        cbar_obj = fig.colorbar(sm, ax=ax, pad=cbar_pad, fraction=cbar_fraction)
                        cbar_obj.set_label(cbar_label, fontsize=cbar_labelsize)
                        cbar_obj.ax.tick_params(labelsize=cbar_ticksize)
                        cbar_obj.set_ticks(np.arange(num_clusters))
                        cbar_obj.set_ticklabels([f'{i}' for i in range(num_clusters)])

                    ax.set_aspect('equal')
                    ax.axis('off')

                if save:
                    fig.savefig(folder + "/clustered_map.png", dpi=300)
                    print("Clustered map saved in " + folder)
            
            def plot_feature_map(self, feature_index, feature, scaler_fit=None, save=False, folder=None, cmap='viridis_r', clusters=None):
                """
                Plot the feature map of the Self-Organizing Map (SOM).
                Parameters:
                feature_index (int): The index of the feature to plot.
                feature (str): The name of the feature to plot.
                save (bool): Whether to save the plot as a .png file.
                folder (str): The folder to save the plot in.
                """
                
                fdict = {
                    'Np': r'$N_p$ [cm$^{-3}$]',
                    'logNp': r'log($N_p$ [cm$^{-3}$])',
                    'Tp': r'$T_p$ [K]',
                    'logTp': r'log($T_p$ [K])',
                    'B': r'$B$ [nT]',
                    'logB': r'log($B$ [nT])',
                    'V': r'$V_p$ [km/s]',
                    'logV': r'log($V_p$ [km/s])',
                    'Phi': r'$\Phi_B$ [deg]',
                    'logPhi': r'log($\Phi_B$ [deg])',
                    'logBeta': r'log($\beta$)',
                    'Bz': r'$B_z$ [nT]',
                    'logBz': r'log($B_z$ [nT])',
                    'Bx': r'$B_x$ [nT]',
                    'logBx': r'log($B_x$ [nT])',
                    'By': r'$B_y$ [nT]',
                    'logBy': r'log($B_y$ [nT])',
                    'Vx': r'$V_x$ [km/s]',
                    'logVx': r'log($V_x$ [km/s])',
                    'Vy': r'$V_y$ [km/s]',
                    'logVy': r'log($V_y$ [km/s])',
                    'Vz': r'$V_z$ [km/s]',
                    'logVz': r'log($V_z$ [km/s])',
                    'Va': r'$V_a$ [km/s]',
                    'logVa': r'log($V_A$ [km/s])',
                    'Pt': r'$P_{tot}$ [pPa]',
                    'logPt': r'log($P_{tot}$ [pPa])',
                }
                
                if feature_index >= self.input_dim:
                    raise ValueError("Feature index out of bounds.")
                
                if scaler_fit is not None:
                    reshaped_weights = self.weights.reshape(self.x*self.y, self.input_dim)
                    weights = scaler_fit.inverse_transform(reshaped_weights).reshape(self.x, self.y, self.input_dim)
                else:
                    weights = self.weights

                if self.topology == 'hexagonal':
                        norm = Normalize(vmin=weights[:, :, feature_index].min(), vmax=weights[:, :, feature_index].max())

                        xx, yy = np.meshgrid(range(self.weights.shape[1]), range(self.weights.shape[0]))
                        xx, yy = 1.*xx, 1.*yy
                        for i, k in enumerate(xx):
                            if i % 2 == 1:
                                xx[i] += 0.5

                        f = plt.figure(figsize=(12,12*(np.sqrt(3) / 2)))
                        ax = f.add_subplot(111)
                        ax.set_aspect('equal')

                        # iteratively add hexagons
                        for i in range(self.weights.shape[0]):
                            for j in range(self.weights.shape[1]):
                                wy = yy[(i, j)] * np.sqrt(3) / 2
                                hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/ np.sqrt(3), facecolor=cmap(norm(weights[i, j, feature_index])), alpha=.9, edgecolor='black', linewidth=0.05)
                                ax.add_patch(hex)

                        # Create a colorbar
                        cmap = cmap
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])
                        # Add the colorbar to the plot
                        cbar = plt.colorbar(sm, ax=ax, pad=0.06, fraction=0.04, extend='both')
                        if scaler_fit is not None:
                            cbar.set_label(fdict.get(feature, feature))
                        else:
                            cbar.set_label(fdict.get(feature, feature)+' (scaled)')

                        if weights[:, :, feature_index].max() > 1e5:
                            cbar.set_ticks(np.arange(weights[:, :, feature_index].min(), weights[:, :, feature_index].max(), 1e5))
                            cbar.set_ticklabels([f"{int(tick/1e5)}" for tick in cbar.get_ticks()])
                            cbar.set_label(fdict.get(feature, feature) + r' ($\times 10^5$)')
                           

                        xrange = np.arange(self.weights.shape[0])
                        yrange = np.arange(self.weights.shape[1])
                        ax.set_xticks(xrange-.5, xrange+1)
                        ax.set_yticks(yrange * np.sqrt(3) / 2, yrange+1)
                        ax.set_xlim(-1, self.weights.shape[1])
                        ax.set_ylim(-0.5, self.weights.shape[0]*(np.sqrt(3) / 2)) 
                        ax.axis('off')
                else:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(weights[:, :, feature_index], cmap='viridis_r', origin='lower')
                    plt.colorbar(label=fdict.get(feature, feature), fraction=0.04, pad=0.06, extend='both')

                if save:
                    plt.savefig(folder + f"/{feature}_map.png", dpi=300)
                    print(f"{feature} map saved in " + folder)
                
                plt.show()
            
            def hits(self, bmus, data):
                hits = []

                if bmus is None:
                    bmus = self.get_bmus(data)
                else:
                    bmus = bmus

                for i in range(self.x * self.y):
                    hits.append(np.sum(bmus == i))
                
                hits = np.array(hits).reshape(self.x, self.y)

                return hits
            
            def hits_class(self, data, bmus=None, classes=None):
                hits = np.zeros((self.x, self.y, len(classes)))

                if bmus is None:
                    bmus = self.get_bmus(data)
                else:
                    bmus = bmus

                for i in range(self.x * self.y):
                    for j, cl in enumerate(classes):
                        hits[i // self.y, i % self.y, j] = np.sum((bmus == i) & (data['Cluster'] == cl))
                
                return hits
                        
            def plot_hitmap(self, data, bmus=None, save=False, filename=None, cmap='Blues'):

                hitmap = self.hits(bmus)
                norm = Normalize(vmin=hitmap.min(), vmax=hitmap.max())
                cmap = cm.get_cmap(cmap)

                if self.topology == 'hexagonal':
                    xx, yy = np.meshgrid(range(self.weights.shape[1]), range(self.weights.shape[0]))
                    xx, yy = 1.*xx, 1.*yy
                    for i, k in enumerate(xx):
                        if i % 2 == 1:
                            xx[i] += 0.5

                    f = plt.figure(figsize=(12,12*(np.sqrt(3) / 2)))
                    ax = f.add_subplot(111)
                    ax.set_aspect('equal')

                    # iteratively add hexagons
                    for i in range(self.weights.shape[0]):
                        for j in range(self.weights.shape[1]):
                            wy = yy[(i, j)] * np.sqrt(3) / 2
                            hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/ np.sqrt(3), facecolor=cmap(norm(hitmap[i, j])), alpha=.9, edgecolor='black', linewidth=0.05)
                            ax.add_patch(hex)

                    xrange = np.arange(self.weights.shape[0])
                    yrange = np.arange(self.weights.shape[1])
                    ax.set_xticks(xrange-.5, xrange+1)
                    ax.set_yticks(yrange * np.sqrt(3) / 2, yrange+1)
                    ax.set_xlim(-1, self.weights.shape[1])
                    ax.set_ylim(-0.5, self.weights.shape[0]*(np.sqrt(3) / 2)) 
                    ax.axis('off')

                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, pad=0.06, fraction=0.04, extend='both')
                    cbar.set_label('Hitmap')
                else:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(hitmap.reshape(self.x, self.y), cmap=cmap, origin='lower')
                    plt.colorbar(label='Hitmap', fraction=0.04, pad=0.06, extend='both')
                if save:
                    plt.savefig(filename if filename is not None else "./hitmap.png", dpi=300)
                    print("Hitmap saved in " + (filename if filename is not None else "./hitmap.png"))
                plt.show()
                
            def plot_clustered_data(data, colors, begin, end, labels, events=None, event_type=None, save=False, folder=None, filename=None, show=True):
                fdict = {'Np': r'$Np$ [cm$^{-3}$]', 'Tp': r'$Tp$ [K]', 'B': r'$B$ [nT]', 'V': r'$V_{SW}$ [km/s]', 'Phi': r'$\Phi_B$ [deg]', 'logBeta': r'log($\beta$)', 'Bz': r'$B_z$ [nT]', 'Bx': r'$B_x$ [nT]', 'By': r'$By$ [nT]', 'Vx': r'$V_x$ [km/s]', 'Vy': r'$V_y$ [km/s]', 'Vz': r'$V_z$ [km/s]', 'Va': r'$V_a$ [km/s]'}

                features = data.drop(columns=['Time']).columns

                data['Cluster'] = labels

                data = data[(data['Time'] >= begin) & (data['Time'] < end)]

                if events is not None:
                    for event in events:
                        event =  event[(event['Start'] >= begin) & (event['Start'] < end)]
 
                cmap = mpl.colors.ListedColormap(colors[:len(np.unique(labels))])
                norm = mpl.colors.BoundaryNorm(np.arange(len(np.unique(labels)) + 1) - 0.5, len(np.unique(labels)))
                # Create a colorbar with the discrete colormap
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])

                fig, ax = plt.subplots(len(features), 1, figsize=(12, 1.25*len(features)), sharex=True)
        
                for i, feature in enumerate(features):
                    for j in range(len(np.unique(labels))):
                        cluster_data = data[data['Cluster'] == j]
                        ax[i].scatter(cluster_data['Time'], cluster_data[feature], s=0.2, c=colors[j])
                    
                    if events is not None:
                        ecolors = ['#d9d9d9', '#a6a6a6', '#000000']  # Greyscale colors up to black
                        for (k, etype) in enumerate(events):
                            for ev in range(len(etype)):
                                ax[i].axvspan(etype['Start'].iloc[ev], 
                                               etype['End'].iloc[ev], 
                                               color=ecolors[k], alpha=0.3)
                            

                    if event_type is not None:
                        for (h,et) in enumerate(event_type):
                                ax[i].plot([], [], color=ecolors[h], alpha=0.3, label=et)
                                ax[i].legend(loc='upper right')
                    
                    ax[i].set_ylabel(fdict.get(feature, feature))
        
                ax[-1].set_xlabel('Time')
                
                # Add the colorbar to the plot
                cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(len(np.unique(labels))), extend='both', pad=0.05, fraction=0.05)
                cbar.set_label('Cluster')
                ax[-1].tick_params(axis='x', rotation=20)

                data = data.drop(columns=['Cluster'])
                
                if save:
                    plt.savefig(f'{folder}/{filename}', dpi=100)
                
                if show:
                    plt.show()
            

            def plot_hexmap(
                self, neurons_content, cmap, bar_min=0, bar_max=100,
                ax=None, cbar=True, cbar_label='Hits',
                cbar_fraction=0.09, cbar_pad=0.015,
                cbar_labelsize=34, cbar_ticksize=28,
                plot_boundaries=False, n_clusters=None, clusters_colors=None
            ):
                norm = Normalize(vmin=bar_min, vmax=bar_max)
                neurons_content = neurons_content.reshape(self.x, self.y)

                xx, yy = np.meshgrid(range(self.weights.shape[1]), range(self.weights.shape[0]))
                xx, yy = 1.0 * xx, 1.0 * yy
                for i in range(xx.shape[0]):
                    if i % 2 == 1:
                        xx[i] += 0.5

                if ax is None:
                    fig = plt.figure(figsize=(12, 12 * (np.sqrt(3) / 2)))
                    ax = fig.add_subplot(111)
                else:
                    fig = ax.figure

                ax.set_aspect('equal')

                for i in range(self.weights.shape[0]):
                    for j in range(self.weights.shape[1]):
                        wy = yy[i, j] * np.sqrt(3) / 2
                        hexagon = RegularPolygon(
                            (xx[i, j], wy), numVertices=6, radius=0.95 / np.sqrt(3),
                            facecolor=cmap(norm(neurons_content[i, j])),
                            alpha=0.9, edgecolor='black', linewidth=0.05
                        )
                        ax.add_patch(hexagon)

                # Overplot cluster boundaries if requested
                if plot_boundaries and n_clusters is not None:
                    clusters_reshaped = self.clustering_predict.reshape(self.x, self.y)
                    for i in range(self.x):
                        for j in range(self.y):
                            wy = yy[i, j] * np.sqrt(3) / 2
                            # Check if this neuron is on a cluster boundary
                            is_boundary = False
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < self.x and 0 <= nj < self.y:
                                        if clusters_reshaped[ni, nj] != clusters_reshaped[i, j]:
                                            is_boundary = True
                                            break
                                if is_boundary:
                                    break
                            
                            if is_boundary:
                                hexagon_boundary = RegularPolygon(
                                    (xx[i, j], wy), numVertices=6, radius=0.95 / np.sqrt(3),
                                    facecolor='none', edgecolor=clusters_colors[clusters_reshaped[i, j]], linewidth=2.0, alpha=0.7
                                )
                                ax.add_patch(hexagon_boundary)

                ax.set_xlim(-1, self.weights.shape[1])
                ax.set_ylim(-0.5, self.weights.shape[0] * (np.sqrt(3) / 2))
                ax.axis('off')

                if cbar:
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cb = fig.colorbar(sm, ax=ax, pad=cbar_pad, fraction=cbar_fraction, extend='both')
                    cb.set_label(cbar_label, fontsize=cbar_labelsize)
                    cb.ax.tick_params(labelsize=cbar_ticksize)
                
                

        

            

            
