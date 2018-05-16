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


    #calculate the moments of hu with each image rotated

    hu_moments_90 = hu_moments(rotation_90)
    hu_moments_180 = hu_moments(rotation_180)
    hu_moments_270 = hu_moments(rotation_270)

    strings = ["" for i in range(7)]
    for i in range(7): 
        strings[i] += str(i+1) + "   &   "
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

##print("Welcome to Hu Moment Calculator")
##print("Here for each vowel there is it will be calculated the 7 original moments of hu, from the original image.")
##print("Then you'll see the 7 moments in an array")

##original_moments_vowels = []
##
####we took one example of each vocal to process the different transformations 
##print("VOWEL A")
####reading image with opencv and the file address
##vowel_a = cv2.imread("Negative-Samples/vowel_134_3.jpg", 0)
##print("Original hu moments")
##original_hu_moments = hu_moments(vowel_a)
##original_moments_vowels.append(original_hu_moments)
##print(original_hu_moments)
###calculating each example of hu momments 
##print("Hu moments in: ROTATION")
##print(invariant_moments_to_rotation(vowel_a, original_hu_moments))
##print("Hu moments in: TRANSLATION")
##invariant_moments_to_translation(vowel_a, original_hu_moments)
##print("Hu moments in: SCALING")
##invariant_moments_to_scaling(vowel_a, original_hu_moments)
##
##print("VOWEL E")
####reading image with opencv and the file address
##vowel_e = cv2.imread("Negative-Samples/vowel_101_2.jpg", 0)
##print("Original hu moments")
##original_hu_moments = hu_moments(vowel_e)
##original_moments_vowels.append(original_hu_moments)
##print(original_hu_moments)
###calculating each example of hu momments 
##print("Hu moments in: ROTATION")
##print(invariant_moments_to_rotation(vowel_e, original_hu_moments))
##print("Hu moments in: TRANSLATION")
##invariant_moments_to_translation(vowel_e, original_hu_moments)
##print("Hu moments in: SCALING")
##invariant_moments_to_scaling(vowel_e, original_hu_moments)
##
##
##print("VOWEL I")
####reading image with opencv and the file address
##vowel_i = cv2.imread("Negative-Samples/vowel_85_3.jpg", 0)
##print("Original hu moments")
##original_hu_moments = hu_moments(vowel_i)
##original_moments_vowels.append(original_hu_moments)
##print(original_hu_moments)
###calculating each example of hu momments 
##print("Hu moments in: ROTATION")
##print(invariant_moments_to_rotation(vowel_i, original_hu_moments))
##print("Hu moments in: TRANSLATION")
##invariant_moments_to_translation(vowel_i, original_hu_moments)
##print("Hu moments in: SCALING")
##invariant_moments_to_scaling(vowel_i, original_hu_moments)
##
##
##
##print("VOWEL O")
####reading image with opencv and the file address
##vowel_o = cv2.imread("Negative-Samples/vowel_53_1.jpg", 0)
##print("Original hu moments")
##original_hu_moments = hu_moments(vowel_o)
##original_moments_vowels.append(original_hu_moments)
##print(original_hu_moments)
###calculating each example of hu momments 
##print("Hu moments in: ROTATION")
##print(invariant_moments_to_rotation(vowel_o, original_hu_moments))
##print("Hu moments in: TRANSLATION")
##invariant_moments_to_translation(vowel_o, original_hu_moments)
##print("Hu moments in: SCALING")
##invariant_moments_to_scaling(vowel_o, original_hu_moments)
##
##
##
##print("VOWEL U")
####reading image with opencv and the file address
##vowel_u = cv2.imread("Negative-Samples/vowel_1_3.jpg", 0)
##print("Original hu moments")
##original_hu_moments = hu_moments(vowel_u)
##original_moments_vowels.append(original_hu_moments)
##print(original_hu_moments)
###calculating each example of hu momments 
##print("Hu moments in: ROTATION")
##print(invariant_moments_to_rotation(vowel_u, original_hu_moments))
##print("Hu moments in: TRANSLATION")
##invariant_moments_to_translation(vowel_u, original_hu_moments)
##print("Hu moments in: SCALING")
##invariant_moments_to_scaling(vowel_u, original_hu_moments)


######################################################################
#                                                                    #
#                                                                    #
#                 IMPLEMENTION OF THE RECOGNIZER                     #
#                                                                    #
#                                                                    #
######################################################################

################################################################
#  Get the all moments of letters  and save the moments in txt #
#  directoy_img: corresponds to the path of the images to load #
#  max_letters: total of the letters to load                   #  
################################################################

def get_all_moments_of_samples(directory_img, max_letters):
    
    all_moments = []

    for i in range(1, max_letters + 1):
        for j in range(1, 4):

            name_img = directory_img + "/vowel_" + str(i) + "_" + str(j) + ".jpg"
            vowel = cv2.imread(name_img, 0)
            vowel_moment = hu_moments(vowel)

            all_moments.append(-np.sign(vowel_moment)*np.log10(np.abs(vowel_moment)))

    np.savetxt("all_moments.txt", all_moments, delimiter="\n")
    
    return all_moments


#################################################################
#Function that calculates the mean of a set of arrays           #
#                                                               #
#Parameters:                                                    #
#   moments: array list of int array                            #
#                                                               #
#Returns:                                                       #
#   mean_moments: mean of all moments                           #
#################################################################

def get_mean(moments):
    np_array_moments = np.array(moments)
    mean_moments = np.mean(np_array_moments, axis=0)[np.newaxis]
    print(mean_moments)
    return np.array(mean_moments[0])


#################################################################
#Function that calculates the standard desviation of a set of   #
#  arrays                                                       #
#                                                               #
#Parameters:                                                    #
#   moments: array list of int array                            #
#                                                               #
#Returns:                                                       #
#   std_moments: std of all moments                             #
#################################################################

def get_standard_deviation(moments):
    np_array_moments = np.array(moments)
    std_moments = np.std(np_array_moments, axis=0)[np.newaxis]
    print(std_moments)
    return std_moments[0]


#################################################################
#Function  performs training calculating the mean and the       #
#standard deviation and saves this result in txt                #                                                  #
#                                                               #
#Parameters:                                                    #
#   moments: array list of int array                            #
#                                                               #                            #
#################################################################

def generate_training(all_moments):

    total_elements = len(all_moments)
    mean = []
    std = []
    vocals_samples = []
    
    
    start = 0
    end = 72
    while(end < total_elements):
        data_extracted = np.array(all_moments[start:end])
        #print("data extracted:" + str(append_lists(data_extracted_h,data_extracted_v).shape))
        for data in data_extracted: 
            vocals_samples.append(data)
        mean_data = get_mean(data_extracted)
        mean.append(mean_data)
        std_data = get_standard_deviation(data_extracted)
        std.append(std_data)
        
        start += 90
        end += 90 

    mean = np.array(mean)
    std = np.array(std)
    vocals_samples = np.array(vocals_samples)
    print(vocals_samples)
    np.savetxt("Mean.txt", mean, delimiter="\n")
    np.savetxt("Std.txt", std, delimiter="\n")
    np.savetxt("Trained.txt", vocals_samples, delimiter="\n")
    return np.array(mean), std, vocals_samples, ["u", "o", "i", "e", "a"]

#################################################################
# from previous txt saved in the folder 
# reads the saved numpy arrays 
#################################################################

def read_training():
    m_mean = np.loadtxt("Mean.txt", delimiter="\n")
    m_std = np.loadtxt("Std.txt", delimiter="\n")
    all_moments = np.loadtxt("all_moments.txt", delimiter="\n")
    trained = np.loadtxt("Trained.txt", delimiter="\n")

    m_mean = m_mean.reshape((5,7))
    m_std = m_std.reshape((5,7))
    all_moments = all_moments.reshape((450,7))
    trained = trained.reshape((360,7))

    return m_mean, m_std, all_moments, trained


    
#################################################################
# check if an array is inside of other two arrays 
# like a range
#################################################################

def inside_range(value, top, down): 
    value = np.round(value,0)
    top = np.round(top,0)
    down = np.round(down,0)
    for i in range(len(value)): 
        if ((value[i] >= down[i]) == False or (value[i]<= top[i]) == False):
            return False
    return True


#################################################################
#Takes the moments given, this is substracted to every mean of  #
#each vowel, then with distance, we calculate the nearest from  #
#each and then we choose, then with standart deviation we see if#
#its inside of the ranges and with that define the confidence   #
#                                                               #
#Parameters:                                                    #
#   moments: the moments of an image to be recognized           #
#   mean: list of vowel mean                                    #
#   std: list of vowel std                                      #
#   order: list of vowels                                       #
#################################################################

def recognize(moments, mean, std, order, training):
    n = nearest_centroid(moments, training) #best_method :D 60 40 
    #n = k_neighbors(3, moments, training) 
    #n =range_method(moments, mean, std, order) 
    return order[n]
    
def range_method(moments, mean, std, order):
    print(moments)
    m_np = np.array(moments)
    a = mean-m_np
    distance_norm = np.linalg.norm(a, axis=1)
    n = np.argmin(distance_norm)    
    print("n es:" + str(n))    
    result = 0 
    distance = []
    inside = []
    for i in range(len(mean)):
        mean_plus_std =  mean[i] + std[i]
        mean_minus_std = mean[i] - std[i]
        if(inside_range(m_np, mean_plus_std, mean_minus_std)): 
            inside.append(order[i])
            distance.append(mean[i] - m_np)    
            result = i  
    a = distance
    if(len(a) == 0):
        print("")
        print("Low confidence")
        return order[n]
    else: 
        distance_norm = np.linalg.norm(a, axis=1)  
        result = np.argmin(distance_norm) 
        print("n es:" + str(result) )    
        print(inside)
        if(inside[result] == order[n]):
            print("")
            print("High confidence")
        else:
            print("")
            print("Low confidence")
            print("result:" + order[n])

    #print("  & \multicolumn{1}{l|}{ " + order[n] + " } ")
    return inside[result] # si devuelve inside son los rangos, si es order es la distancia 


#def k_neighbors(neighbors): 
def nearest_centroid(moments, training):
    m_np = np.array(moments)
    a = training-m_np
    distance_norm = np.linalg.norm(a, axis=1)
    n = np.argmin(distance_norm)  
    return (int(n/72))


def get_min(distance):
    n = np.argmin(distance)
    distance[n] = 50000000
    return int(n/72), distance


def k_neighbors(k, new_image, projected_images):
    a = projected_images-new_image
    distance_norm = np.linalg.norm(a, axis=1)
    neighbors = []
    for i in range(k):
        neighbors.append(get_min(distance_norm)[0])
        new_distance_norm = get_min(distance_norm)[1]
        distance_norm = new_distance_norm
    unique, counts = np.unique(neighbors, return_counts=True)
    for i in range(len(unique)):
        print("Del sujeto: " + str(unique[i]) + " hay " +
              str(counts[i]) + " apariciones")
    j = np.argmax(counts)
    return unique[j]


#################################################################
# given  all_moments that are all the values 
# takes 72 vowel moments to evaluate them in the sistem 

#################################################################

def get_testing_samples(all_moments, order): 
    total_elements = len(all_moments)
    expected = []
    testing_samples = []
    start = 72
    end = 90
    j = 0 
    while(end <= total_elements):
        data_extracted = all_moments[start:end]
        #print(data_extracted)
        #print(data_extracted.shape) 
        vocal = order[j]
        vocals = [vocal for i in range(18)]
        expected.append(vocals)
        testing_samples.append(data_extracted)
        start += 90
        end += 90 
        j += 1
    testing_samples = np.array(testing_samples)
    #print(testing_samples)
    return testing_samples, expected
   
    
def test():
    mean, std, all_moments, training= read_training()
    
    order = ["u", "o", "i", "e", "a"]
    samples, expected = get_testing_samples(all_moments, order)
    true = 0
    false = 0
    for i in range(samples.shape[0]):
        set_vocals = samples[i]
        expected_ = expected[i]
        for j in range(len(expected_)): 
            #print("\multicolumn{1}{|l|}{" + expected_[j] + "}")
            result = recognize(set_vocals[j],mean,std,order, training)
            print("expected:" + expected_[j])
            
            if (result == expected_[j]):
                print("True")
                #print(" & \multicolumn{1}{l|}{True} \\\\ \hline")
                true += 1
            else:
                print("False")
                #print(" & \multicolumn{1}{l|}{False} \\\\ \hline")
                false += 1
    print("Fiabilidad:")
    print("Elementos de exito: " + str(true/90*100) + "%")
    print("Elementos de fracaso: "+ str(false/90*100) + "%")
    print(all_moments.shape)
    print(false)
    print(true)


## test 
all_moments = get_all_moments_of_samples("Negative-Samples", 150)

generate_training(all_moments)

#mean, std, all_moments = read_training()

#order = ["u", "o", "i", "e", "a"]
#testing_samples, expected = get_testing_samples(all_moments,order)

test()

"""a1 = cv2.imread("a1.png", 0)
cv2.bitwise_not(a1, a1)
ha1 = hu_moments(a1)

cv2.imwrite("a1n.png", a1)

a2 = cv2.imread("a2.png", 0)
cv2.bitwise_not(a2, a2)
ha2 = hu_moments(a2)

cv2.imwrite("a2n.png", a2)


a3 = cv2.imread("a3.png", 0)
cv2.bitwise_not(a3, a3)
ha3 = hu_moments(a3)

cv2.imwrite("a3n.png", a3)

print(ha1)
print("")
print(ha2)
print("")
print(ha3) """










