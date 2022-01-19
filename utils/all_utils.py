import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib # FOR SAVING MY MODEL AS A BINARY FILE
from matplotlib.colors import ListedColormap
import os
import logging

plt.style.use("fivethirtyeight") # THIS IS STYLE OF GRAPHS

def prepare_data(df):
    """It is used to separate out independent and dependent features        

    Args:
        df ([pd.DataFrame]): It's is the Pandas DataFrame

    Returns:
        tuples: It returns the tuples of dependent and independent variables
    """

    logging.info("Preparing the data by segragating the independent and dependent variables")

    X = df.drop("y", axis=1)

    y = df["y"]

    return X, y

def save_model(model,filename):
    """This saves the train model

    Args:
        model (python object): trained model to
        filename (str): Path to save the trained model
    """
    logging.info("Saving the trained model")
    model_dir = "models"
    os.makedirs(model_dir,exist_ok=True)
    filePath = os.path.join(model_dir,filename)
    joblib.dump(model,filePath)
    logging.info(f"Saved the trained model {filePath}")

def save_plot(df,file_name,model):
    """[summary]

    Args:
        df : It's a Data Frame
        file_name : It is a path to save the plot
        model : Trained model
    """
    def _create_base_plot(df):
        logging.info("Creating the base plot")
        df.plot(kind="scatter",x="x1",y="x2",c="y",s=100,cmap="winter")
        plt.axhline(y=0,color="black",linestyle='--',linewidth=1)
        plt.axvline(x=0,color="black",linestyle='--',linewidth=1)
        figure = plt.gcf() # Get current figure
        figure.set_size_inches(10,8)
  
    def _plot_decision_regions(X,y,classifier,resolution=.2):
        logging.info("Plotting the decision region")
        colors = ("red","blue","lightgreen","gray","cyan")
        cmap = ListedColormap(colors[:len(np.unique(y))])

        X = X.values
        x1 = X[:,0]
        x2 = X[:,1]

        x1_min,x1_max = x1.min() - 1, x1.max() + 1
        x2_min,x2_max = x2.min() - 1, x2.max() + 1

        xx1,xx2 = np.meshgrid(np.arange(x1_min, x1_max,resolution),
                            np.arange(x1_min, x1_max,resolution))
        print(xx1)
        print(xx1.ravel())

        z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
        z = z.reshape(xx1.shape)
        plt.contourf(xx1,xx2,z,alpha=0.2,cmap=cmap)
        plt.xlim(xx1.min(),xx1.max())
        plt.ylim(xx2.min(),xx2.max())

        plt.plot()

    X,y = prepare_data(df)

    _create_base_plot(df)
    _plot_decision_regions(X,y,model)

    plot_dirs = 'plots'
    os.makedirs(plot_dirs,exist_ok=True)
    plotPath = os.path.join(plot_dirs,file_name)
    plt.savefig(plotPath)
    logging.info(f"Saving the plot at {plotPath}")
