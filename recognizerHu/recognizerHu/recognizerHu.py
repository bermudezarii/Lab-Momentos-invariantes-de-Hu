# -*- coding: utf-8 -*-

import numpy as np
import cv2


###############################################################################
## Convert the samples to negative                                           ##
## Parameters:                                                               ##
##  -name_directory_load: name of the folder where are the samples           ##
##  -name_diractory_save: name of the folder where save the negatives images ##
##  -max_count: number of images                                             ##
###############################################################################


def negative_samples(name_directory_load, name_directory_save, max_count):

    for i in range(1, max_count + 1):
        for j in range(1, 4):
            
            name_img = name_directory_load + "/vowel_" + str(i) + "_" + str(j) + ".jpg"
            
            img = cv2.imread(name_img, 0)
    
            cv2.bitwise_not(img, img)
            
            cv2.imwrite(name_directory_save + "/vowel_" + str(i) + "_" + str(j) + ".jpg", img)

    
#**************   Geometric Transformations  ***************** #
            
def rotation_image(image, degrees):
    rows,cols = image.shape
    matrix = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0), degrees ,1)
    rotation = cv2.warpAffine(image, matrix,(cols,rows))

    return rotation
    
def translation_image(image, x, y):

    rows,cols = image.shape
    matrix = np.float32([[1,0,x],[0,1,y]])
    translation = cv2.warpAffine(image,matrix,(cols,rows))

    return translation

def scaling_image(image, value):
    row,cols = image.shape[:2]
    scaling = cv2.resize(image,(value*row, value*cols), interpolation = cv2.INTER_CUBIC)

    return scaling
    

#refencia: https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html


#################################################################
##  Calculate the seven moments of hu for an image             ##
##  and return the array with the moments.                     ##
#################################################################

def hu_moments(image):
    
    moments_image = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments_image).flatten()

    return hu_moments

#################################################################
##  Calculate the seven moments of hu for the 5                ##
##  vowels and return these resulting arrays.                  ##
#################################################################

def all_hu_moments(vowel_a, vowel_e, vowel_i, vowel_o, vowel_u):
    hu_moments_a = hu_moments(vowel_a)
    hu_moments_e = hu_moments(vowel_e)
    hu_moments_i = hu_moments(vowel_i)
    hu_moments_o = hu_moments(vowel_o)
    hu_moments_u = hu_moments(vowel_u)

    return hu_moments_a, hu_moments_e, hu_moments_i, hu_moments_o, hu_moments_u

#######################################################################
##                                                                   ##
##  Check if the moments are invariant for the same image but with   ##
##  a certain degree of rotation.                                    ##
##                                                                   ##
##  Return  true if it finds that all invariant moments are equal    ##
##   with  the invariant moment initially received.                  ##
##																	 ##
#######################################################################
    
def invariant_moments_to_rotation(image, original_hu_moments):

    rotation_90 = rotation_image(image, 90)
    rotation_180 = rotation_image(image, 180)
    rotation_270 = rotation_image(image, 270)
    rotation_360 = rotation_image(image, 360)

    #cv2.imshow("90", rotation_90)
    #cv2.imshow("180", rotation_180)
    #cv2.imshow("270", rotation_270)
    #cv2.imshow("330", rotation_330)

    #calculate the moments of hu with each image rotated

    hu_moments_90 = hu_moments(rotation_90)
    hu_moments_180 = hu_moments(rotation_180)
    hu_moments_270 = hu_moments(rotation_270)
    hu_moments_360 = hu_moments(rotation_360)

    print(hu_moments_90)
    print(hu_moments_180)
    print(hu_moments_270)
    print(hu_moments_360)
    
    print(np.allclose(original_hu_moments, hu_moments_90))
    #Verified if all arrays are equals
    
    if(np.allclose(original_hu_moments, hu_moments_90) and np.allclose(original_hu_moments, hu_moments_180) and
       np.allclose(original_hu_moments, hu_moments_270) and np.allclose(original_hu_moments, hu_moments_360)):
        return True
    else:
        return False
    


#def invariant_moments_to_scaling(image, original_hu_moments): 
    

#negative_samples("Samples","Negative-Samples", 150)
#print(hu_moments(cv2.imread("Negative-Samples/vowel_101_2.jpg", 0)))
#cv2.imshow("..",(rotate_image(cv2.imread("Negative-Samples/vowel_101_2.jpg", 0),90)))
#cv2.imshow("..",(translation_image(cv2.imread("Negative-Samples/vowel_101_2.jpg", 0),15,15)))

cv2.imshow("..",(scaling_image(cv2.imread("Negative-Samples/vowel_101_2.jpg", 0),4)))
cv2.waitKey()
#vowels 

vowel_a = cv2.imread("Negative-Samples/vowel_134_3.jpg", 0)
#vowel_e = cv2.imread("Negative-Samples/vowel_101_2.jpg", 0)
#vowel_i = cv2.imread("Negative-Samples/vowel_85_3.jpg", 0)
#vowel_o = cv2.imread("Negative-Samples/vowel_53_1.jpg", 0)
#vowel_u = cv2.imread("Negative-Samples/vowel_1_3.jpg", 0)

original_hu_moments = hu_moments(vowel_a)
print(invariant_moments_to_rotation(vowel_a, original_hu_moments))

