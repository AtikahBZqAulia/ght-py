import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import cmath
import os
from scipy.ndimage.filters import sobel
from collections import defaultdict

def load_image(path):
    return cv.imread("semicircle3.jpg", 0)

def load_template(path2):
    return cv.imread("semicircle2.jpg", 0)

def get_edge_coordinates(image):
    coordinates = []
    x, y = image.shape
    for a in range(x):
        for b in range(y):
            if (image[a, b]!=0):
                coordinates.append((a, b))
    return coordinates
    print(len(coordinates))
    
def get_reference_points(coordinates):
    x = 0
    y = 0
    for i in range(len(coordinates)):
        x = x + coordinates[i][0]
        y = y + coordinates[i][1]
    x = x/len(coordinates)
    y = y/len(coordinates)
    return (int(x), int(y))

def get_gradient_orientation(image):
    gradient = cv.Sobel(image,cv.CV_64F,1,0,ksize=5)  
    abs_gradient = np.absolute(gradient)
    
    return abs_gradient

def build_r_table(image, refPoint, coordinates):
    templateCanny = cv.Canny(template, 100, 200)
    gradient = get_gradient_orientation(templateCanny)
    
    R_table = {}
    for i, point in enumerate(coordinates):
        rx = refPoint[0] - point[0]
        ry = refPoint[1] - point[1]
        r = (rx, ry)
        phi = gradient[point[0], point[1]]
        if(phi not in list(R_table.keys())):
           R_table[phi] = [r]
        else:
            R_table[phi].append(r)
    R_table['refPoint'] = refPoint

    return R_table

def get_r_tableWithRotation(templateCanny,angle=2):
    r_tableWithRotation = []
    rows, cols = templateCanny.shape
    for i  in range(int(360/angle)):
        theta = angle*i
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_template = cv.warpAffine(templateCanny, M, (cols, rows))
        template_coordinates = get_edge_coordinates(rotated_template)
        template_reference = get_reference_points(template_coordinates)
        r_table_theta = build_r_table(rotated_template, template_reference, template_coordinates)
        r_tableWithRotation.append(r_table_theta)
    return r_tableWithRotation

# def get_r_TableWithScale():
#     r_tableWithScale = []
#     return 

def accumulate_gradients(imgCanny, R_table, angle=2):
    rows = imgCanny.shape[0]
    cols = imgCanny.shape[1]
    edgeCoordinates = get_edge_coordinates(imgCanny)
    gradient = get_gradient_orientation(imgCanny)
    
    accumulator = np.ndarray((int(360/angle), rows, cols))
    for theta, R_Table_Theta in enumerate(R_table):
        for i, edge in enumerate(edgeCoordinates):
            phi = gradient[edge[0], edge[1]]
            if (phi in list(R_Table_Theta.keys())):
                temp = R_Table_Theta[phi]
                for radius, vector in enumerate(temp):
                    x = edge[0] + vector[0]
                    y = edge[0] + vector[1]
                    if (x>=0 and x<rows) and (y>=0 and y<cols):
                        accumulator[theta, x, y]+=1
            else:
                continue
    
    return accumulator

def reconstruct_image(R_table, theta, a, b, imgCanny, angle=2):
    R_table = R_table[int(theta)]
    output = np.ones_like(imgCanny)*255
    closure = np.zeros_like(imgCanny)*255
    closureCoordinates = []

    rows = imgCanny.shape[0]
    cols = imgCanny.shape[1]
    edgeCoordinates = get_edge_coordinates(image)
    gradient = get_gradient_orientation(imgCanny)

    for i, edge in enumerate(edgeCoordinates):
        phi = gradient[edge[0], edge[1]]
        if (phi in list(R_table.keys())):
            temp = R_table[phi]
            for radius, vector in enumerate(temp):
                x = a - vector[0]
                y = b - vector[1]
                if (x>=0 and x<rows) and (y>=0 and y<cols):
                    cv.rectangle(output, (y, x), (y+2, x+2), 0x0000FF, -1)
                    closureCoordinates.append((y,x))
        else:
            continue
    cv.fillConvexPoly(closure, np.int32(closureCoordinates), (1.0, 1.0, 1.0), 16, 0)

    return output, closure

def generalized_hough(image, template, angle=2):
    output = np.zeros(image.shape, image.dtype)
    imgCanny = cv.Canny(image, 100, 200, output)
    cv.imshow('test image', imgCanny)
    cv.waitKey()
    templateCanny = cv.Canny(template, 100, 200, output)
    cv.imshow('template image', templateCanny)
    cv.waitKey()
    r_table = get_r_tableWithRotation(templateCanny, angle=2)
    img_accuTable = accumulate_gradients(imgCanny, r_table, angle=2)
    theta, a, b = np.unravel_index(img_accuTable.argmax(), img_accuTable.shape)
    output, closure = reconstruct_image(r_table, theta, a, b, imgCanny, angle)
    cv.imshow("output", output)
    cv.waitKey()
    cv.imshow("closure", closure)
    cv.waitKey()
    return output, closure

# def test_ght(gh, reference_image, query):
#     query_image = cv.imread(query, flatten=True)
#     accumulator = gh(query_image)

#     plt.clf()
#     plt.gray()
    
#     fig = plt.figure()
#     fig.add_subplot(2,2,1)
#     plt.title('Reference image')
#     plt.imshow(reference_image)
    
#     fig.add_subplot(2,2,2)
#     plt.title('Tested image')
#     plt.imshow(query_image)
    
#     fig.add_subplot(2,2,3)
#     plt.title('Accumulator')
#     plt.imshow(accumulator)
    
#     fig.add_subplot(2,2,4)
#     plt.title('Detection')
#     plt.imshow(query_image)
    
#     # top 5 results in red
#     m = n_max(accumulator, 5)
#     y_points = [pt[1][0] for pt in m]
#     x_points = [pt[1][1] for pt in m] 
#     plt.scatter(x_points, y_points, marker='o', color='r')

#     # top result in yellow
#     i,j = np.unravel_index(accumulator.argmax(), accumulator.shape)
#     plt.scatter([j], [i], marker='x', color='y')
    
#     d,f = os.path.split(query)[0], os.path.splitext(os.path.split(query)[1])[0]
#     plt.savefig(os.path.join(d, f + '_output.png'))
    
#     return 

if __name__ == '__main__':
    path = "C:/Users/LENOVO/skripsi"
    image = load_image(path)
    template = load_template(path)
    generalized_hough(image, template, angle=2)
