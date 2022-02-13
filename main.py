import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from HeatTransfer import nextTemp
from OperationObject import *
import pickle
import pandas as pd
import re

def reshapeImage(img, domX, domY, dx, dy):
    '''
    img : the original image : np.ndarry
    domx : the spacing for x elements : float
    domy : the spacing for y elements : float
    dx : the new x spacing : float
    dy : the new y spacing : float
    '''

    # Get the total x and y length of the domain
    yLen = img.shape[1] * domY
    xLen = img.shape[0] * domX

    # Solvle for the required number of cells for the new domain
    yCellCount = int(np.ceil(yLen / dy))
    xCellCount = int(np.ceil(xLen / dx))

    # Create the new domain
    return cv2.resize(img, (yCellCount, xCellCount))


def transformTemplate(tem, xLen, yLen, x, y, rot):
    '''
    xLen : number of cells in the x of the final image
    yLen : number of cells in the y of the final image
    tem : the template image
    x : the desired x position
    y : the desired y position
    rot : the desired rotation
    '''

    # Get the indecies of the focal point as the max heat flux cell
    focusIndex = np.unravel_index(tem.argmax(), tem.shape)

    # Get the height and width of the image
    height, width = tem.shape

    # Compute the transformation matrix about the focal point
    R1 = cv2.getRotationMatrix2D(np.float32([focusIndex[1], focusIndex[0]]),rot,1)

    # Get the sin and cos angles of the rotation matrix
    RotMatCos = np.abs(R1[0][0])
    RotMatSin = np.abs(R1[0][1])

    # Calculate the new height and width such that the image does not get clipped
    newWidth = int((height * RotMatSin) + (width * RotMatCos))
    newHeight = int((height * RotMatCos) + (width * RotMatSin))

    # Update the rotation matrix
    R1[0][2] += (newWidth/2) - focusIndex[1]
    R1[1][2] += (newHeight/2) - focusIndex[0]

    # Apply the rotation to the image
    rotTem = cv2.warpAffine(tem, R1, (newWidth, newHeight))

    # Calculate the index of the focal point of the rotated image
    rotFocusIndex = np.unravel_index(rotTem.argmax(), rotTem.shape)

    # Create the transformation matrix
    T1 = np.float32([[1,0,x-rotFocusIndex[1]],[0,1,y-rotFocusIndex[0]]])

    # Compute and return the tranasformed image
    return cv2.warpAffine(rotTem, T1, (xLen, yLen))


def vectorizedTemp(T, Q, k, alpha, dt, dx, dy):
    xlen, ylen = T.shape

    Td = T[1:ylen-1, 1:xlen-1]

    Typ = T[0:ylen-2,1:xlen-1]
    Tyn = T[2:ylen,1:xlen-1]

    Txp = T[1:ylen-1,0:xlen-2]
    Txn = T[1:ylen-1,2:xlen]

    T[1:ylen-1, 1:xlen-1] += ((Typ + Tyn - 2 * Td) / np.power(dy, 2) + (Txp + Txn - 2 * Td) / np.power(dx, 2) + Q[1:ylen-1, 1:xlen-1]/k) * alpha * dt

    return T



#
def main():
    intensityMap = TransformedImage2*dx*dy*1000000000
    maxTemperatureDomain = np.zeros(TransformedImage2.shape, dtype=np.float64)
    prevTemp = np.ones(TransformedImage2.shape, dtype=np.float64) * intitalTemperature
    currTemp = np.ones(TransformedImage2.shape, dtype=np.float64) * intitalTemperature

    start_time = time.time()


    for t in range(200):
        #currTemp = vectorizedTemp(prevTemp, intensityMap, k, alpha, dt, dx, dy)

        for x in range(1, xCellCount-1):
            for y in range(1, yCellCount-1):
                #currTemp[y, x] = nextTemp(prevTemp[y, x], intensityMap[y, x], prevTemp[y-1, x], prevTemp[y+1, x], prevTemp[y, x-1], prevTemp[y, x+1])
                currTemp[y, x] = nextTemp(  prevTemp[y, x],
                                            intensityMap[y, x],
                                            prevTemp[y-1, x],
                                            prevTemp[y+1, x],
                                            prevTemp[y, x-1],
                                            prevTemp[y, x+1],
                                            dt,
                                            alpha,
                                            dx,
                                            dy,
                                            k)

        prevTemp = currTemp

        #print(np.max(prevTemp))

        print("--- %s seconds ---" % (time.time() - start_time))

        #plt.show()

def plot_routine():
    with open('generationData.p', 'rb') as file:
        data = pickle.load(file)

    generationData = np.zeros(len(data), dtype=np.int16)
    performanceData = np.zeros(len(data), dtype=np.float64)
    for i, row in enumerate(data):
        generation = row[0]
        performance = row[1]
        routine = row[2]

        generationData[i] = generation
        performanceData[i] = performance



        #if generation==599 and True:
        plotRoutine(routine, generation+1, i)



    data = pd.DataFrame({"Generation":generationData, 'Performance':performanceData})

    meanPerformance = data.groupby("Generation").mean()
    stdPerformance = data.groupby("Generation").std()

    print(stdPerformance.shape)


    #plt.errorbar(meanPerformance.index.values, meanPerformance["Performance"], yerr=stdPerformance["Performance"], fmt='.k', ecolor='lightgray')
    #plt.title("Fitness Vs. Generation")
    #plt.xlabel("Generation")
    #plt.ylabel("Fitness")
    #plt.show()


def plotRoutine(rout, gen=None, saveNum=None):
    # Get the current working directory
    CWD = os.getcwd()

    # Get the intensity matrix and domain
    IntensityLoc = os.path.join(CWD, r'heatIntensityMatrix.mat')
    GirdLoc = os.path.join(CWD, r'girdProperties.mat')

    # Set the x and y element size for FEA
    dx = 0.001 #m
    dy = 0.001 #m

    # The dimensions of the domain
    xLen = 0.2
    yLen = 0.2

    # The FEA solver properties
    k = 0.598
    rho = 997
    c = 4184
    dt = 0.1

    # Scale the intensity feild for the FEA spacing
    IntensityTemplate = loadIntensityFeildFromMat(IntensityLoc, GirdLoc, dx, dy)

    # Create the domain (Regular tissue 0, cancerous tissue 1) ~ for now this is a circle
    domain = np.zeros((int(yLen/dy), int(xLen/dx)))
    mask = create_circular_mask(int(yLen/dy), int(xLen/dx), radius=18)
    mask += create_circular_mask(int(yLen/dy), int(xLen/dx), center=(110, 110), radius=12)
    mask += create_circular_mask(int(yLen/dy), int(xLen/dx), center=(90, 110), radius=9)
    mask += create_circular_mask(int(yLen/dy), int(xLen/dx), center=(90, 90), radius=7)
    mask += create_circular_mask(int(yLen/dy), int(xLen/dx), center=(115, 90), radius=9)
    domain += mask

    #plt.imshow(domain)
    #plt.title("Cancerous Domain")
    #plt.show()

    # Create the operation object
    operation = Operation(IntensityTemplate, domain, dx, dy, k, rho, c, dt)

    # Initial Operation Vector
    op = np.array([ 0.5, 0.5, 0.1, 1.0, 0.2,
                    0.5, 0.5, 0.2, 1.0, 0.2,
                    0.5, 0.5, 0.3, 1.0, 0.3,
                    0.5, 0.5, 0.4, 1.0, 0.2,
                    0.5, 0.5, 0.5, 1.0, 0.2,
                    0.5, 0.5, 0.6, 1.0, 0.2,
                    0.4, 0.5, 0.7, 1.0, 0.2,], dtype=np.float32)

    bounds = np.array([ [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.float32)

    optimizer = CMA(mean=op, sigma = 0.06, bounds=bounds)


    op = np.array(rout)

    operation = Operation(IntensityTemplate, domain, dx, dy, k, rho, c, dt)

    operation.addOperationScalar(op[0], op[1], op[2], op[3], op[4])
    operation.addOperationScalar(op[5], op[6], op[7], op[8], op[9])
    operation.addOperationScalar(op[10], op[11], op[12], op[13], op[14])
    operation.addOperationScalar(op[15], op[16], op[17], op[18], op[19])
    operation.addOperationScalar(op[20], op[21], op[22], op[23], op[24])
    operation.addOperationScalar(op[25], op[26], op[27], op[28], op[29])
    operation.addOperationScalar(op[30], op[31], op[32], op[33], op[34])

    temperatureDist = operation.computeOperation()

    fig, (ax1, ax2) = plt.subplots(2, 1)

    if gen != None:
        fig.suptitle('Generation {}'.format(gen), fontsize=8)

    ax1.set_title("Temperature Distribution", fontsize=8)
    ax1.imshow(temperatureDist)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    threshImage = np.where(operation.maxHeat >= operation.criticalTemp, 1, operation.maxHeat)
    threshImage = np.where(operation.maxHeat < operation.criticalTemp, 0, threshImage)
    ax2.set_title("Regions Above Critical Temperature", fontsize=8)
    ax2.imshow(threshImage)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)

    if saveNum != None:
        plt.savefig('/Users/gianpaolopittis/Desktop/Generation Images/{}.png'.format(str(saveNum)))
    plt.close()

def createTimeLapse():
    image_folder = '/Users/gianpaolopittis/Desktop/Generation Images'
    video_name = '/Users/gianpaolopittis/Desktop/video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda f: int(re.sub('\D', '', f)))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width,height))

    for image in images:
        video.write(cv2.resize(cv2.imread(os.path.join(image_folder, image)), (width,height)))


    video.release()
    cv2.destroyAllWindows()

def sinPlot():
    x = np.arange(0,5*np.pi,0.1)   # start,stop,step
    y1 = np.sin(x+np.pi/4)
    y2 = np.sin(x+np.pi/2)
    y3 = np.sin(x+np.pi)
    y4 = np.sin(x+2*np.pi)
    y5 = np.sin(x+np.pi/4)
    y6 = np.sin(x+np.pi/2)

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)

    ax1.plot(x, y1)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    ax2.plot(x, y2)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)

    ax3.plot(x, y3)
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)

    ax4.plot(x, y4)
    ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)

    ax5.plot(x, y5)
    ax5.xaxis.set_visible(False)
    ax5.yaxis.set_visible(False)

    ax6.plot(x, y6)
    ax6.xaxis.set_visible(False)
    ax6.yaxis.set_visible(False)

    plt.show()

if __name__ == "__main__":
    #plot_routine()
    #createTimeLapse()
    sinPlot()
