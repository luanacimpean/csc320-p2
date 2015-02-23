###########################################################################
## Handout painting code.
###########################################################################
from PIL import Image
from pylab import *
from canny import *
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
import matplotlib.image as mpimg
import scipy as sci
import os


np.set_printoptions(threshold = np.nan)  

def colorImSave(filename, array):
    imArray = sci.misc.imresize(array, 3., 'nearest')
    if (len(imArray.shape) == 2):
        sci.misc.imsave(filename, cm.jet(imArray))
    else:
        sci.misc.imsave(filename, imArray)

def markStroke(mrkd, p0, p1, rad, val):
    # Mark the pixels that will be painted by
    # a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1).
    # These pixels are set to val in the ny x nx double array mrkd.
    # The paintbrush is circular with radius rad>0
    
    sizeIm = mrkd.shape
    sizeIm = sizeIm[0:2];
    nx = sizeIm[1]
    ny = sizeIm[0]
    p0 = p0.flatten('F')
    p1 = p1.flatten('F')
    rad = max(rad,1)
    # Bounding box
    concat = np.vstack([p0,p1])
    bb0 = np.floor(np.amin(concat, axis=0))-rad
    bb1 = np.ceil(np.amax(concat, axis=0))+rad
    # Check for intersection of bounding box with image.
    intersect = 1
    if ((bb0[0] > nx) or (bb0[1] > ny) or (bb1[0] < 1) or (bb1[1] < 1)):
        intersect = 0
    if intersect:
        # Crop bounding box.
        bb0 = np.amax(np.vstack([np.array([bb0[0], 1]), np.array([bb0[1],1])]), axis=1)
        bb0 = np.amin(np.vstack([np.array([bb0[0], nx]), np.array([bb0[1],ny])]), axis=1)
        bb1 = np.amax(np.vstack([np.array([bb1[0], 1]), np.array([bb1[1],1])]), axis=1)
        bb1 = np.amin(np.vstack([np.array([bb1[0], nx]), np.array([bb1[1],ny])]), axis=1)
        # Compute distance d(j,i) to segment in bounding box
        tmp = bb1 - bb0 + 1
        szBB = [tmp[1], tmp[0]]
        q0 = p0 - bb0 + 1
        q1 = p1 - bb0 + 1
        t = q1 - q0
        nrmt = np.linalg.norm(t)
        [x,y] = np.meshgrid(np.array([i+1 for i in range(int(szBB[1]))]), np.array([i+1 for i in range(int(szBB[0]))]))
        d = np.zeros(szBB)
        d.fill(float("inf"))
        
        if nrmt == 0:
            # Use distance to point q0
            d = np.sqrt( (x - q0[0])**2 +(y - q0[1])**2)
            idx = (d <= rad)
        else:
            # Use distance to segment q0, q1
            t = t/nrmt
            n = [t[1], -t[0]]
            tmp = t[0] * (x - q0[0]) + t[1] * (y - q0[1])
            idx = (tmp >= 0) & (tmp <= nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = abs(n[0] * (x[np.where(idx)] - q0[0]) + n[1] * (y[np.where(idx)] - q0[1]))
            idx = (tmp < 0)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q0[0])**2 +(y[np.where(idx)] - q0[1])**2)
            idx = (tmp > nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q1[0])**2 +(y[np.where(idx)] - q1[1])**2)

            #Pixels within crop box to paint have distance <= rad
            idx = (d <= rad)
        #Mark the pixels
        if np.any(idx.flatten('F')):
            xy = (bb0[1]-1+y[np.where(idx)] + sizeIm[0] * (bb0[0]+x[np.where(idx)]-2)).astype(int)
            sz = mrkd.shape
            m = mrkd.flatten('F')
            m[xy-1] = val
            mrkd = m.reshape(mrkd.shape[0], mrkd.shape[1], order = 'F')

            '''
            row = 0
            col = 0
            for i in range(len(m)):
                col = i//sz[0]
                mrkd[row][col] = m[i]
                row += 1
                if row >= sz[0]:
                    row = 0
            '''
            
            
            
    return mrkd

def paintStroke(canvas, x, y, p0, p1, colour, rad):
    # Paint a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1)
    # on the canvas (ny x nx x 3 double array).
    # The stroke has rgb values given by colour (a 3 x 1 vector, with
    # values in [0, 1].  The paintbrush is circular with radius rad>0
    sizeIm = canvas.shape
    sizeIm = sizeIm[0:2]
    idx = markStroke(np.zeros(sizeIm), p0, p1, rad, 1) > 0
    # Paint
    if np.any(idx.flatten('F')):
        canvas = np.reshape(canvas, (np.prod(sizeIm),3), "F")
        xy = y[idx] + sizeIm[0] * (x[idx]-1)
        canvas[xy-1,:] = np.tile(np.transpose(colour[:]), (len(xy), 1))
        canvas = np.reshape(canvas, sizeIm + (3,), "F")
    return canvas


if __name__ == "__main__":
    # Read image and convert it to double, and scale each R,G,B
    # channel to range [0,1].
    imRGB = array(Image.open('orchid.jpg'))
    imRGB = double(imRGB) / 255.0
    cannied = canny(imRGB[:,:,0], 2)

    plt.clf()
    plt.axis('off')
    
    sizeIm = imRGB.shape
    sizeIm = sizeIm[0:2]
    # Set radius of paint brush and half length of drawn lines
    rad = 3
    halfLen = 5
    
    # Set up x, y coordinate images, and canvas.
    [x, y] = np.meshgrid(np.array([i+1 for i in range(int(sizeIm[1]))]), np.array([i+1 for i in range(int(sizeIm[0]))]))
    canvas = np.zeros((sizeIm[0],sizeIm[1], 3))
    canvas.fill(-1) ## Initially mark the canvas with a value out of range.
    # Negative values will be used to denote pixels which are unpainted.
    
    # Random number seed
    np.random.seed(29645)
    
    time.time()
    time.clock()

    '''
        Takes an input image in the range [0, 1] and generate a gradient image
        with edges marked by 1 pixels.
    '''
    imin = (imRGB[:,:,0]).copy() * 255.0

    # Create the gauss kernel for blurring the input image
    # It will be convolved with the image
    # wsize should be an odd number
    wsize = 5
    sigma = 4
    gausskernel = gaussFilter(sigma, window = wsize)
    # fx is the filter for vertical gradient
    # fy is the filter for horizontal gradient
    # Please not the vertical direction is positive X

    fx = createFilter([0,  1, 0,
                       0,  0, 0,
                       0, -1, 0])
    fy = createFilter([ 0, 0, 0,
                       -1, 0, 1,
                        0, 0, 0])

    imout = conv(imin, gausskernel, 'valid')
    # print "imout:", imout.shape
    gradxx = conv(imout, fx, 'valid')
    gradyy = conv(imout, fy, 'valid')

    gradx = np.zeros(imin.shape)
    grady = np.zeros(imin.shape)
    padx = (imin.shape[0] - gradxx.shape[0]) / 2.0
    pady = (imin.shape[1] - gradxx.shape[1]) / 2.0
    gradx[padx:-padx, pady:-pady] = gradxx
    grady[padx:-padx, pady:-pady] = gradyy
    
    # Net gradient is the square root of sum of square of the horizontal
    # and vertical gradients

    grad = hypot(gradx, grady)
    theta = arctan2(grady, gradx)
    theta = 180 + (180 / pi) * theta
    
    # Only significant magnitudes are considered. All others are removed
    xx, yy = where(grad < 3)
    theta[xx, yy] = 0
    grad[xx, yy] = 0
    
    k = -1
    unpainted = np.where (canvas == -1)
#     canvas = paintStroke(canvas, x,y, np.array([2,2]), np.array([2,2]), np.array([1,0,0]), 1)

    
    while (len(unpainted[1]) > 1):
        k += 1

        unpainted = np.where(canvas == -1)
        rando = randint(0,len(unpainted[0]) - 1)
        
        cntr = np.array([unpainted[1][rando], unpainted[0][rando]])
#         cntr = np.amin(np.vstack((cntr, np.array([sizeIm[1], sizeIm[0]]))), axis=0)

        # Orientation of paint brush strokes
        stroke_theta =  theta[cntr[1],cntr[0]]
        # print 'stroke_theta is', stroke_theta
        # Set vector from center to one end of the stroke.
        delta = np.array([cos(radians(stroke_theta)), sin(radians(stroke_theta))])
        

        length1, length2 = (halfLen, halfLen) 

        # if the center is at a canny edge:
        if (cannied[cntr[1]][cntr[0]] == 1):
            # Grab colour from image at center position of the stroke.
            colour = np.reshape(imRGB[cntr[1], cntr[0], :],(3,1))
            # Add the stroke to the canvas
            nx, ny = (sizeIm[1], sizeIm[0])
            # length1, length2 = (halfLen, halfLen)        
            canvas = paintStroke(canvas, x, y, cntr+1, cntr+1, colour, rad)
            # print 'stroke of length 1', k

        else:
            len1 = 0
            while (True):
                candidate_len = len1 + 1
                if (candidate_len >= length1):
                    break
                p = cntr - delta * len1
                if (p[0] >= len(canvas[0]) or p[0] < 0):
                    break
                if (p[1] >= len(canvas) or p[1] < 0):
                    break
                if cannied[p[1]][p[0]] == 1:
                    break
                len1 = candidate_len
            p1 = cntr - delta * len1
            
            len2 = 0
            while (True):
                candidate_len = len2 + 1
                if (candidate_len >= length2):
                    break
                p = cntr + delta * len2
                if (p[0] >= len(canvas[0]) or p[0] < 0):
                    break
                if (p[1] >= len(canvas) or p[1] < 0):
                    break
                if cannied[p[1]][p[0]] == 1:
                    break
                len2 = candidate_len
            p2 = cntr + delta * len2

                
            colour = np.reshape(imRGB[cntr[1], cntr[0], :],(3,1))
            # Add the stroke to the canvas
            nx, ny = (sizeIm[1], sizeIm[0])
            canvas = paintStroke(canvas, x, y, p1+1, p2+1, colour, rad)
            
        # # finding a negative pixel
        # # Randomly select stroke center
        # cntr = np.floor(np.random.rand(2,1).flatten() * np.array([sizeIm[1], sizeIm[0]])) + 1
        # cntr = np.amin(np.vstack((cntr, np.array([sizeIm[1], sizeIm[0]]))), axis=0)
        # # Grab colour from image at center position of the stroke.
        # colour = np.reshape(imRGB[cntr[1]-1, cntr[0]-1, :],(3,1))
        # # Add the stroke to the canvas
        # nx, ny = (sizeIm[1], sizeIm[0])
        # length1, length2 = (halfLen, halfLen)        
        # canvas = paintStroke(canvas, x, y, cntr - delta * length2, cntr + delta * length1, colour, rad)
        # #print imRGB[cntr[1]-1, cntr[0]-1, :], canvas[cntr[1]-1, cntr[0]-1, :]
        # print 'stroke', k
        
        unpainted = np.where (canvas == -1)
    print "done!"
    time.time()
    
    canvas[canvas < 0] = 0.0
    plt.clf()
    plt.axis('off')
    plt.imshow(canvas)
    show()
#     plt.pause(3)
#     colorImSave('output.png', canvas)
