import matplotlib.pyplot as plt
import numpy as np
import itertools
from mpl_toolkits.mplot3d import Axes3D

class Plotter():

    def __init__(self):
        pass

    @staticmethod
    def plot_scatter_plot(X,Y,area,labels,X_legend='X',Y_legend='Y',alpha=0.5):

        plt.scatter(X, Y, s=area, c=labels.astype(float), alpha=alpha)
        plt.xlabel(X_legend, fontsize=18)
        plt.ylabel(Y_legend, fontsize=16)

        plt.show()


    @staticmethod
    def scatter_3D(X,Y,Z,labels,X_legend='X',Y_legend='Y',Z_legend='Z',alpha=0.5):
        fig = plt.figure(1, figsize=(8, 6))
        plt.clf()
        ax = Axes3D(fig, rect=(0, 0, .95, 1), elev=48, azim=134)

        plt.cla()

        ax.set_xlabel(X_legend)
        ax.set_ylabel(Y_legend)
        ax.set_zlabel(Z_legend)

        ax.scatter(X, Y, Z, c=labels.astype(float))

    @staticmethod
    def plot_confussion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """

        plt.figure()
        np.set_printoptions(precision=2)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()