"""
swksom: Solar Wind Classification with Self-Organizing Maps

Classify solar wind observations into CSW, Ejecta, HAW, or SSW using a pre-trained 
Self-Organizing Map (SOM) model. The pipeline includes automatic preprocessing 
(Box-Cox and log transforms with RobustScaler) and visualization of results.

Quick start:
    from swksom import ClusterSOM
    import pickle
    
    with open('./models/toroidal_som.pkl', 'rb') as f:
        model = pickle.load(f)
    
    predictions = model.predict(scaled_data)

See example.ipynb for complete usage tutorial.
"""

from .ClusterSOM import ClusterSOM

__version__ = "0.1.0"
__author__ = "Your Name"
__all__ = ["ClusterSOM"]
