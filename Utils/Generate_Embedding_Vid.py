# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:47:01 2020

@author: jpeeples
"""
#Need to update code for results script

import numpy as np
import os
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import matplotlib.animation as animation
import matplotlib.cm as colormap
import cv2
# import pdb
import moviepy.video.io.ImageSequenceClip
from Utils.Plot_Decision_Boundary import plot_decision_boundary
import sklearn.metrics


def Generate_Embed_Vid(embeddings, labels, sub_dir, model, class_names=None, embed_dim=2):
    print('Generate embedding visual...')

    # Turn interactive plotting off, don't show plots
    plt.ioff()
    
    ## Add images you want in video
    frame_array = []

    for epoch in range(len(embeddings)):
        count = 0
        fig = plt.figure(figsize=(14, 6))
        silhouette_scores = []
        CH_scores = []
        for phase in embeddings[epoch].keys():

            GT_vals = labels[epoch][phase]
            silhouette_score = sklearn.metrics.silhouette_score(embeddings[epoch][phase], GT_vals, metric='euclidean',
                                                                random_state=1)
            silhouette_c_score = sklearn.metrics.silhouette_score(embeddings[epoch][phase], GT_vals, metric='cosine',
                                                                  random_state=1)
            calinski_score = sklearn.metrics.calinski_harabasz_score(embeddings[epoch][phase], GT_vals)
            # Generate figure for embedding
            if embed_dim == 3:
                ax = fig.add_subplot(1, 3, count + 1, projection='3d')
            else:
                ax = fig.add_subplot(1, 3, count + 1)

            if epoch == len(embeddings) - 1:
                plot_decision_boundary(embeddings[29][phase], class_names, model, ax)
                CH_scores.append(calinski_score)
                silhouette_scores.append(silhouette_score)
                silhouette_scores.append(silhouette_c_score)
            if class_names is not None:
                colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
                for texture in range (0, len(class_names)):
                    x = embeddings[epoch][phase][[np.where(GT_vals==texture)],0]
                    y = embeddings[epoch][phase][[np.where(GT_vals==texture)],1]
                    if embed_dim == 2:
                        ax.scatter(x, y, color = colors[texture,:],
                                          label=class_names[texture])
                    else: #3D
                        z = embeddings[epoch][phase][[np.where(GT_vals==texture)],2]
                        ax.scatter(x, y, z, color = colors[texture,:],
                                          label=class_names[texture])
                            
                ax.set_title(phase.capitalize())
                ax.set_xlabel('Silhouette: ' + str(silhouette_score) + '\nCalinski Harabasz: ' + str(calinski_score))
            else:
                colors = colormap.rainbow(np.linspace(0, 1, len(np.unique(labels))))
                for texture in np.unique(labels):
                    x = embeddings[epoch][phase][[np.where(GT_vals==texture)],0]
                    y = embeddings[epoch][phase][[np.where(GT_vals==texture)],1]
                    if embed_dim == 2:
                        ax.scatter(x, y, color = colors[texture,:],
                                          label=class_names[texture])
                    else: #3D
                        z = embeddings[epoch][phase][[np.where(GT_vals==texture)],2]
                        ax.scatter(x, y, z, color = colors[texture,:],
                                          label=class_names[texture])
                ax.set_title(phase.capitalize())
                ax.set_xlabel('Silhouette: ' + str(silhouette_score) + '\nCalinski Harabasz: ' + str(calinski_score))
            # Counter for figure
            count += 1
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.subplots_adjust(right=0.80) 
        
        if class_names is not None:
            ax.legend(class_names,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            ax.legend(np.unique(labels),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
        plt.suptitle('Embeddings for Epoch {} of {}'.format(epoch+1,len(embeddings)))
        
        # # redraw the canvas
        fig.canvas.draw()
        
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # img is rgb, convert to opencv's default bgr
        # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
        #Display figure (to check)
        # cv2.imshow("plot",img)

        ## Training loop:
        frame_array.append(img)

        if epoch == len(embeddings) - 1:
            fig.savefig(sub_dir + 'FinalEmbedding.png', dpi=fig.dpi)
            np.savetxt((sub_dir + 'Final_Silhouette_Scores.txt'), silhouette_scores, fmt='%.2f')
            np.savetxt((sub_dir + 'Final_CH_Scores.txt'), CH_scores, fmt='%.2f')

        # Close figure and image
        plt.close(fig)

    fps = 2  ## Frame rate
    pathOut = sub_dir + 'Embedding.mp4' ## where you want to save the video
    
    # ## Create the video
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frame_array, fps=fps)
    clip.write_videofile(pathOut)

    return silhouette_scores, CH_scores

# ani = SubplotAnimation()
# # ani.save('test_sub.mp4')
# plt.show()