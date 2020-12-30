# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:47:01 2020

@author: jpeeples
"""
#Need to update code for results script

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import matplotlib.cm as colormap
import cv2
import pdb
import moviepy.video.io.ImageSequenceClip


def Generate_Hist_Vid(centers,widths,num_bins,sub_dir):
    
    print('Generate histogram visual...')

    # Turn interactive plotting off, don't show plots
    plt.ioff()
    
    ## Add images you want in video
    frame_array = []
    
    #Colors for scatter map (should colors be based off bins or feature maps)
    #Maybe use color bar to show bins that belong to each feature map?
    # colors = colormap.rainbow(np.linspace(0, 1, centers.shape[-1]))
    # pdb.set_trace()
    color_array = np.linspace(0, 1, int(centers.shape[-1]/num_bins))
    color_array = np.repeat(color_array,num_bins)
    colors = colormap.rainbow(color_array)

    for epoch in range(len(centers)):
        fig =  plt.figure()
 
        plt.scatter(centers[epoch], widths[epoch],c=colors)

        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # fig.subplots_adjust(right=0.80) 
        plt.xlabel('Bin Centers')
        plt.ylabel('Bin Widths')
    
        plt.suptitle(('{}-Bin Histogram for {} ' +
                     'Feature Maps Epoch {} of {}').format(num_bins,
                                                          int(centers.shape[-1]/num_bins),
                                                          epoch,
                                                          len(centers)-1))
        
        # # redraw the canvas
        fig.canvas.draw()
        
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # img is rgb, convert to opencv's default bgr
        # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
        #Display figure (to check)
        # pdb.set_trace()
        # cv2.imshow("plot",img)

        ## Training loop:
        frame_array.append(img)
        
        #Close figure and image
        plt.close(fig)
        
  
    fps = 2  ## Frame rate
    pathOut = sub_dir + 'Histogram.mp4' ## where you want to save the video
    
    # ## Create the video
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frame_array, fps=fps)
    clip.write_videofile(pathOut)
              
