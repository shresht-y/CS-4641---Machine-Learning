import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt

# Set plotly renderer
rndr_type = "jupyterlab+png"
pio.renderers.default = rndr_type


class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:  # 5 points
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose

        Hint: np.linalg.svd by default returns the transpose of V
              Make sure you remember to first center your data by subtracting the mean of each feature.

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        #subtract the mean for each feature 
        # n = instances 
        # d = features
        self.U, self.S, self.V = np.linalg.svd(X - np.mean(X, 0), False)
        return None 
    
        

    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:  # 2 pts
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
        """
        u, s, v = np.linalg.svd(data - np.mean(data, 0), False)
        return u[:,:K] * s[:K]

    def transform_rv(
        self, data: np.ndarray, retained_variance: float = 0.99
    ) -> np.ndarray:  # 3 pts
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.

        """
        u, s, v = np.linalg.svd(data - np.mean(data, 0), False)
        variance = np.square(s)
        total_var = np.sum(variance)
        var_threshold = total_var*retained_variance
        k = np.sum(np.cumsum(variance)<=var_threshold)
        k+=1
        return u[:,:k]*s[:k]
            

    def get_V(self) -> np.ndarray:
        """Getter function for value of V"""

        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) -> None:  # 5 pts
        """
        You have to plot two different scatterplots (2d and 3d) for this function. For plotting the 2d scatterplot, use your PCA implementation to reduce the dataset to only 2 features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using plotly.
        Hint: Refer to https://plotly.com/python/line-and-scatter/ for making scatter plots with plotly.
        Hint: We recommend converting the data into a pandas dataframe before plotting it. Refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html for more details.

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels

        Return: None
        """
        #2d plot 
        self.fit(X)
        fitted_x = self.transform(X,2)
        df = pd.DataFrame(fitted_x, columns=["Feature 1", "Feature 2"])
        df['Label'] = y
        fig = px.scatter(df, x='Feature 1', y='Feature 2', color="Label", title=fig_title)
        fig.show()
        
        #for three features 
        fitted_x_3d = self.transform(X, 3)
        df2 = pd.DataFrame(fitted_x_3d, columns=['Feature 1', 'Feature 2', 'Feature 3'])
        df2['Label'] = y
        fig2 = px.scatter_3d(df2, x='Feature 1', y='Feature 2', z='Feature 3', color='Label', title=fig_title)
        fig2.update_traces(marker={'size':5})
        fig2.show()

        

        
        
