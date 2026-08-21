"""
swksom: Solar Wind K-means Self-Organizing Maps

A library for classifying solar wind observations using Self-Organizing Maps
combined with K-means clustering. Predicts cluster assignments for solar wind
data using a pre-trained toroidal SOM model.
"""

from .ClusterSOM import ClusterSOM

__version__ = "0.1.0"
__author__ = "Your Name"
__all__ = ["ClusterSOM"]
