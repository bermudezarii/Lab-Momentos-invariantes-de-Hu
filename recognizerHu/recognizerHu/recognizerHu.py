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
##																	                ##
#######################################################################
    
def invariant_moments_to_rotation(image, original_hu_moments):

    rotation_90 = rotation_image(image, 90)
    rotation_180 = rotation_image(image, 180)
    rotation_270 = rotation_image(image, 270)


    #cv2.imshow("90", rotation_90)
    #cv2.imshow("180", rotation_180)
    #cv2.imshow("270", rotation_270)


    #calculate the moments of hu with each image rotated

    hu_moments_90 = hu_moments(rotation_90)
    hu_moments_180 = hu_moments(rotation_180)
    hu_moments_270 = hu_moments(rotation_270)

    strings = ["" for i in range(7)]
    for i in range(7): 
        strings[i] += str(i+1) + "   &   "X
        strings[i] += str(hu_moments_90[i]) + "   &   "
        strings[i] += str(hu_moments_180[i]) + "   &   "
        strings[i] += str(hu_moments_270[i])+ "   \\\\  "
        print(strings[i])

    print(hu_moments_90)
    print(hu_moments_180)
    print(hu_moments_270)

    
    #Verified if all arrays are equals
    
    if(np.allclose(original_hu_moments, hu_moments_90) and np.allclose(original_hu_moments, hu_moments_180) and
       np.allclose(original_hu_moments, hu_moments_270)):
        return True
    else:
        return False
    
#######################################################################
##                                                                   ##
##  Check if the moments are invariant for the same image but with   ##
##  a certain degree of scaling                                      ##
##                                                                   ##
##  Return  true if it finds that all invariant moments are equal    ##
##   with  the invariant moment initially received.                  ##
##																	          ##
#######################################################################   
def invariant_moments_to_scaling(image, original_hu_moments): 
    scaling_2 = scaling_image(image,2)
    scaling_4 = scaling_image(image,4)
    scaling_8 = scaling_image(image,8)
    scaling_16 = scaling_image(image,16)
    
    hu_moments_2 = hu_moments(scaling_2)
    hu_moments_4 = hu_moments(scaling_4)
    hu_moments_8 = hu_moments(scaling_8)
    hu_moments_16 = hu_moments(scaling_16)
 
    strings = ["" for i in range(7)]
    for i in range(7): 
        strings[i] += str(i+1) + "   &   "
        strings[i] += str(hu_moments_2[i]) + "   &   "
        strings[i] += str(hu_moments_4[i])+ "   \\\\  "
        print(strings[i])
    
    strings = ["" for i in range(7)]
    for i in range(7): 
        strings[i] += str(i+1) + "   &   "
        strings[i] += str(hu_moments_8[i]) + "   &   "
        strings[i] += str(hu_moments_16[i])+ "   \\\\  "
        print(strings[i])

    print(hu_moments_2)
    print(hu_moments_4)
    print(hu_moments_8)
    print(hu_moments_16)
    
    
   
## thos function calculates the difference between two numpy arrays     
def compare(original, new): 
    diffs = original - new 
    print("diffs: " + str(diffs))
    
    
    
#######################################################################
##                                                                   ##
##  Check if the moments are invariant for the same image but with   ##
##  a certain degree of translation                                  ##
##                                                                   ##
##  Return  true if it finds that all invariant moments are equal    ##
##   with  the invariant moment initially received.                  ##
##																	         ##
#######################################################################       
def invariant_moments_to_translation(image, original_hu_moments): 
    translation_2 = translation_image(image,2,10)
    translation_4 = translation_image(image,4,10)
    translation_8 = translation_image(image,8,10)
    translation_16 = translation_image(image,16,10)
    
    hu_moments_2 = hu_moments(translation_2)
    hu_moments_4 = hu_moments(translation_4)
    hu_moments_8 = hu_moments(translation_8)
    hu_moments_16 = hu_moments(translation_16)
    
    strings = ["" for i in range(7)]
    for i in range(7): 
        strings[i] += str(i+1) + "   &   "
        strings[i] += str(hu_moments_2[i]) + "   &   "
        strings[i] += str(hu_moments_4[i])+ "   \\\\  "
        print(strings[i])
    
    strings = ["" for i in range(7)]
    for i in range(7): 
        strings[i] += str(i+1) + "   &   "
        strings[i] += str(hu_moments_8[i]) + "   &   "
        strings[i] += str(hu_moments_16[i])+ "   \\\\  "
        print(strings[i])
    
    print(hu_moments_2)
    print(hu_moments_4)
    print(hu_moments_8)
    print(hu_moments_16)
    
    
    
    
"""
COMMENTS ABOUT LAB 3: 
    The intention in this lab was to demonstrate that hu moments gives us 
    similar numbers no matter the transformation (in rotation, scaling, and translation)
    
    At the end we concluded that the different moments with the transformations aren't 
    equal at all, but pretty similar, which is understandable because 
    to get some meaningful answers we take a log transform
    
    The fact they are not identical is to be expected, in openCV for some images, 
    the original and transformed images are a bit diferent.
"""
print("Welcome to Hu Moment Calculator")
print("Here for each vowel there is it will be calculated the 7 original moments of hu, from the original image.")
print("Then you'll see the 7 moments in an array")

original_moments_vowels = []


##we took one example of each vocal to process the different transformations 
print("VOWEL A")
##reading image with opencv and the file address
vowel_a = cv2.imread("Negative-Samples/vowel_134_3.jpg", 0)
print("Original hu moments")
original_hu_moments = hu_moments(vowel_a)
original_moments_vowels.append(original_hu_moments)
print(original_hu_moments)
#calculating each example of hu momments 
print("Hu moments in: ROTATION")
print(invariant_moments_to_rotation(vowel_a, original_hu_moments))
print("Hu moments in: TRANSLATION")
invariant_moments_to_translation(vowel_a, original_hu_moments)
print("Hu moments in: SCALING")
invariant_moments_to_scaling(vowel_a, original_hu_moments)

print("VOWEL E")
##reading image with opencv and the file address
vowel_e = cv2.imread("Negative-Samples/vowel_101_2.jpg", 0)
print("Original hu moments")
original_hu_moments = hu_moments(vowel_e)
original_moments_vowels.append(original_hu_moments)
print(original_hu_moments)
#calculating each example of hu momments 
print("Hu moments in: ROTATION")
print(invariant_moments_to_rotation(vowel_e, original_hu_moments))
print("Hu moments in: TRANSLATION")
invariant_moments_to_translation(vowel_e, original_hu_moments)
print("Hu moments in: SCALING")
invariant_moments_to_scaling(vowel_e, original_hu_moments)


print("VOWEL I")
##reading image with opencv and the file address
vowel_i = cv2.imread("Negative-Samples/vowel_85_3.jpg", 0)
print("Original hu moments")
original_hu_moments = hu_moments(vowel_i)
original_moments_vowels.append(original_hu_moments)
print(original_hu_moments)
#calculating each example of hu momments 
print("Hu moments in: ROTATION")
print(invariant_moments_to_rotation(vowel_i, original_hu_moments))
print("Hu moments in: TRANSLATION")
invariant_moments_to_translation(vowel_i, original_hu_moments)
print("Hu moments in: SCALING")
invariant_moments_to_scaling(vowel_i, original_hu_moments)



print("VOWEL O")
##reading image with opencv and the file address
vowel_o = cv2.imread("Negative-Samples/vowel_53_1.jpg", 0)
print("Original hu moments")
original_hu_moments = hu_moments(vowel_o)
original_moments_vowels.append(original_hu_moments)
print(original_hu_moments)
#calculating each example of hu momments 
print("Hu moments in: ROTATION")
print(invariant_moments_to_rotation(vowel_o, original_hu_moments))
print("Hu moments in: TRANSLATION")
invariant_moments_to_translation(vowel_o, original_hu_moments)
print("Hu moments in: SCALING")
invariant_moments_to_scaling(vowel_o, original_hu_moments)



print("VOWEL U")
##reading image with opencv and the file address
vowel_u = cv2.imread("Negative-Samples/vowel_1_3.jpg", 0)
print("Original hu moments")
original_hu_moments = hu_moments(vowel_u)
original_moments_vowels.append(original_hu_moments)
print(original_hu_moments)
#calculating each example of hu momments 
print("Hu moments in: ROTATION")
print(invariant_moments_to_rotation(vowel_u, original_hu_moments))
print("Hu moments in: TRANSLATION")
invariant_moments_to_translation(vowel_u, original_hu_moments)
print("Hu moments in: SCALING")
invariant_moments_to_scaling(vowel_u, original_hu_moments)

    
