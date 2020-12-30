import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn.functional as nnf
import matplotlib.cm as colormap

def plot_decision_boundary(X,class_names,model,ax, steps=1000, cmap='tab20b'):
    #cmap = plt.get_cmap(cmap)
    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    if(X.shape[1] ==2):
        xx, yy = np.meshgrid(x_span, y_span)
        zz=0
        output = model.module.fc(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
        top_prob, labels = nnf.softmax(output, dim=1).topk(1, dim=1)
        z = labels.reshape(xx.shape)
        #    fig, ax = plt.subplots()
        ax.contourf(xx, yy, z, cmap=colormap.rainbow, alpha=.4)
    else:  #dim = 3
        zmin,zmax = X[:,2].min()-1, X[:,2].max()+1
        z_span = np.linspace(zmin,zmax,steps)
        xx,yy,zz = np.meshgrid(x_span,y_span,z_span)
        output = model.module.fc(torch.from_numpy(np.c_[xx.ravel(), yy.ravel(),zz.ravel()]).float())
        top_prob, labels = nnf.softmax(output, dim=1).topk(1, dim=1)
        z = labels.reshape(xx.shape)
        #    fig, ax = plt.subplots()
        ax.contourf(xx, yy, zz, z, cmap=colormap.rainbow, alpha=.4)


