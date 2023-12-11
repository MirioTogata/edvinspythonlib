
'''
Functions and classes for course Image Analysis
'''
import imageio
import numpy as np
import matplotlib.pyplot as plt

def plotimg(filepath,colmap="gray"):
    '''Plots an image from the given filepath, and returns the image'''
    img = imageio.imread(filepath)
    plt.imshow(img,cmap=colmap)
    return img

def todecimal(x):
    return int(x, 2)

def tobin(x):
    return bin(x)[2:]

def histogram(x):
    #probably flatten first
    plt.hist(x, bins=100)
    plt.show()

def changevectorbasis (v, basis):
    #v is a vector, basis is a matrix
    #returns the coordinates of v in the new basis
    return np.linalg.solve(basis, v)

def thinlensequation(b = 0, g = 0, f = 0):
    #b is the distance from the object to the lens in meters
    #g is the distance from the lens to the image in meters
    #f is the focal length of the lens in meters
    #returns the thing not given, in meters
    if b == 0:
        return 1/(1/f - 1/g)
    elif g == 0:
        return 1/(1/f - 1/b)
    elif f == 0:
        return 1/(1/b + 1/g)
    else:
        return "Error: Only one of b, g or f can be 0"
    
def runlengthencoding(x):
    #x is a matrix/image
    #returns a list of tuples (count,number)
    runlength = 1
    result = []
    new = np.array(x).flatten()
    for i in range(1,len(new)):
        if new[i] == new[i-1]:
            runlength += 1
        else:
            result.append((runlength, new[i-1]))
            runlength = 1
        if i == len(new)-1:
            result.append((runlength, new[i]))
    return result, len(result)*2

def chaincode(center,chaincode):
    #given a center and chaincode, returns the binary image
    image = np.zeros((50,50))
    image[center[0]][center[1]] = 1
    current = center
    for i in chaincode:
        if int(i) == 0:
            current = (current[0], current[1]+1)
        elif int(i) == 1:
            current = (current[0]-1, current[1]+1)
        elif int(i) == 2:
            current = (current[0]-1, current[1])
        elif int(i) == 3:
            current = (current[0]-1, current[1]-1)
        elif int(i) == 4:
            current = (current[0], current[1]-1)
        elif int(i) == 5:
            current = (current[0]+1, current[1]-1)
        elif int(i) == 6:
            current = (current[0]+1, current[1])
        elif int(i) == 7:
            current = (current[0]+1, current[1]+1)
        image[current[0]][current[1]] = 1
    plt.matshow(image)

def stretchimage(image,mind=0,maxd=255):
    #image is a numpy array
    #we strech values between 0 and 255
    #returns the stretched image
    return (image - np.min(image)) * (maxd-mind) / (np.max(image) - np.min(image))

def stretchfunction(imagemin, imagemax,mind=0,maxd=255):
    a = (maxd-mind) / (imagemax - imagemin)
    b = -a * imagemin
    return a,b

def gammamapping(image, gamma):
    #image is a numpy array
    #gamma is a float
    #returns the gamma mapped image
    return np.power(image/255, gamma)*255

def logmapping(image,gamma): #NOT TESTED
    #image is a numpy array
    #gamma is a float
    #returns the log mapped image
    c = 255/np.log(1+255)
    return c*np.log(1+image/255)*255

def bilinear(x,x1,x2,x3,x4):
    '''
    Given a point(x) in a square and the corners of the square(x1,x2,x3,x4);
    Returns what the value of x would be if it was bilinearly interpolated.
    x is a list (x,y). x1,x2,x3,x4 are lists (x,y,value)
    '''
    dx = x[0] - x1[0]
    dy = x[1] - x1[1]
    return x1[2]*(1-dx)*(1-dy) + x2[2]*dx*(1-dy) + x3[2]*(1-dx)*dy + x4[2]*dx*dy

def calculateR(x,y,theta):
    '''calculates r for polar coordinates'''
    rad = 0.0175*theta
    return x*np.cos(rad) + y*np.sin(rad)

def calculateentropy(x):
    '''Calculates the entropy when given the probability of each class'''
    return -np.sum(x*np.log2(x))