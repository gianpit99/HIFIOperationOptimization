import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cmaes import CMA


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

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


def loadIntensityFeildFromMat(I_Path, G_Path, dx, dy):
    # Get the intensity matrix and domain
    Intensity = np.array(scipy.io.loadmat(I_Path)["I"], dtype=np.float32) #[W / m^2]
    Domain = scipy.io.loadmat(G_Path)

    # Get the y dimension spacing
    yGrid = Domain['Grid'][0][0][5]
    yGrid = np.squeeze(np.vstack([np.flip(yGrid), yGrid[1:]]))/100

    # Get the x dimension spacing
    xGrid = np.squeeze(Domain['Grid'][0][0][6])/100

    # Get the gird size (This assumes its constant over the whole domain)
    pdy = np.abs(np.diff(yGrid, axis=0))[0] #cm
    pdx = np.abs(np.diff(xGrid, axis=0))[0] #cm

    # Scale the intensity feild for the FEA spacing
    return reshapeImage(Intensity, pdx, pdy, dx, dy)


from HeatTransfer import nextTemp
#    nextTemp(T, Q, Tyn1, Typ1, Txn1, Txp1, dt, alpha, dx, dy, k)
#def nextTemp(T, Q, Tyn1, Typ1, Txn1, Txp1, dx, dy, dt, alpha, k):
#    return ((Typ1 - 2.0 * T + Tyn1)/np.power(dy, 2) + (Txp1 - 2.0 * T + Txn1)/np.power(dx, 2) + Q/k) * alpha * dt + T


def vectorizedTemp(T, Q, k, alpha, dt, dx, dy):
    xlen, ylen = T.shape

    Td = T[1:ylen-1, 1:xlen-1]

    Typ = T[0:ylen-2,1:xlen-1]
    Tyn = T[2:ylen,1:xlen-1]

    Txp = T[1:ylen-1,0:xlen-2]
    Txn = T[1:ylen-1,2:xlen]

    T[1:ylen-1, 1:xlen-1] += ((Typ + Tyn - 2 * Td) / np.power(dy, 2) + (Txp + Txn - 2 * Td) / np.power(dx, 2) + Q[1:ylen-1, 1:xlen-1]/k) * alpha * dt

    return T


class Operation:
    def __init__(self, temp, domain, dx, dy, k, rho, c, dt):


        # The domain with class labels
        self.Domain = domain

        self.dx = dx                    #m
        self.dy = dy                    #m
        self.dt = dt                    #s
        self.k = k
        self.rho = rho
        self.c = c
        self.dt = dt

        self.alpha = k / (rho * c)

        self.xCells = domain.shape[1]
        self.yCells = domain.shape[0]

        self.xLen = domain.shape[1] * dx
        self.yLen = domain.shape[0] * dy

        # Intensity Template
        self.ITemplate = temp

        # The domain to keep track of the max temperature
        self.maxHeat = np.zeros(domain.shape, dtype=np.float64)

        # The queue to hold each operation plan
        self.operationQueue = []

        # Settings
        self.maxHeatTime = 6            #s
        self.maxCoolTime = 5            #s
        self.criticalTemp = 50          #C
        self.upperTemp = 80             #C
        self.bodyTemp = 37              #C`

        # The labels for cancerous and normal tissue in the domain
        self.CancerousClassLabel = 1
        self.TissueClassLabel = 0


        return

    def transformTemplate(self, x, y, rot):
        '''
        xLen : number of cells in the x of the final image
        yLen : number of cells in the y of the final image
        tem : the template image
        x : the desired x position
        y : the desired y position
        rot : the desired rotation
        '''

        # Get the indecies of the focal point as the max heat flux cell
        focusIndex = np.unravel_index(self.ITemplate.argmax(), self.ITemplate.shape)

        # Get the height and width of the image
        height, width = self.ITemplate.shape

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
        rotTem = cv2.warpAffine(self.ITemplate, R1, (newWidth, newHeight))

        # Calculate the index of the focal point of the rotated image
        rotFocusIndex = np.unravel_index(rotTem.argmax(), rotTem.shape)

        # Create the transformation matrix
        T1 = np.float32([[1,0,x-rotFocusIndex[1]],[0,1,y-rotFocusIndex[0]]])

        # Compute and return the tranasformed image
        return cv2.warpAffine(rotTem, T1, (self.xCells, self.yCells))


    def addOperation(self, x, y, rot, ht, ct):
        self.operationQueue.append((x, y, rot, ht, ct))

    def addOperationScalar(self, x, y, rot, ht, ct):
        xUnScaled = int(x * self.xCells)
        yUnScaled = int(y * self.yCells)
        rotUnScaled = rot * 360.0
        htUnScaled = ht * self.maxHeatTime
        ctUnScaled = ct * self.maxCoolTime

        self.addOperation(  xUnScaled,
                            yUnScaled,
                            rotUnScaled,
                            htUnScaled,
                            ctUnScaled)

    def getOperation(self):
        return self.operationQueue.pop()

    def computeOperation(self):
        currentTime = 0
        currentTimeTarget = 0
        count = 0

        prevTemp = np.ones(self.Domain.shape, dtype=np.float64) * self.bodyTemp
        currTemp = np.ones(self.Domain.shape, dtype=np.float64) * self.bodyTemp

        while (len(self.operationQueue) != 0):
            count += 1

            # Get the insonation properties
            x, y, rot, ht, ct = self.getOperation()

            # Compute the heating matrix for the insonation
            Q = self.transformTemplate(x, y, rot) * self.dx * self.dy * 100000000

            # Run the FEA for the required heating cycle
            currentTimeTarget += ht
            while (currentTime <= currentTimeTarget):
                currTemp = vectorizedTemp(  prevTemp,
                                            Q,
                                            self.k,
                                            self.alpha,
                                            self.dt,
                                            self.dx,
                                            self.dy)

                #for x in range(1, self.Domain.shape[1]-1):
                #    for y in range(1, self.Domain.shape[0]-1):
                #        currTemp[y, x] = nextTemp(  prevTemp[y, x],
                #                                    Q[y, x],
                #                                    prevTemp[y-1, x],
                #                                    prevTemp[y+1, x],
                #                                    prevTemp[y, x-1],
                #                                    prevTemp[y, x+1],
                #                                    self.dt,
                #                                    self.alpha,
                #                                    self.dx,
                #                                    self.dy,
                #                                    self.k
                #                                    )
                prevTemp = currTemp
                currentTime += self.dt



            self.maxHeat = np.maximum(self.maxHeat, prevTemp)

            # Run the FEA for the required cooling cycle
            Q = np.zeros(self.Domain.shape, dtype=np.bool)
            currentTimeTarget += ct
            while (currentTime <= currentTimeTarget):
                currTemp = vectorizedTemp(  prevTemp,
                                            Q,
                                            self.k,
                                            self.alpha,
                                            self.dt,
                                            self.dx,
                                            self.dy)
                prevTemp = currTemp
                currentTime += self.dt

                self.tempOperationRoutine(prevTemp, imgCount)
                imgCount+=1

        return prevTemp

    def operationLoss(self):
        TempAboveTresh = np.where(self.maxHeat >= self.criticalTemp, 1, 0)
        TempBelowTresh = np.where(self.maxHeat < self.criticalTemp, 1, 0)

        SqaredDifference = np.square(self.maxHeat - self.criticalTemp)

        # Bad Conditions
        CancerousBelow = np.multiply(self.Domain, TempBelowTresh)
        NCancerousAbove = np.multiply(np.logical_not(self.Domain), TempAboveTresh)
        BadDomainCells = np.logical_or(CancerousBelow, NCancerousAbove)

        # Good Conditions
        CancerousAbove = np.multiply(self.Domain, TempAboveTresh)
        NCancerousBelow = np.multiply(np.logical_not(self.Domain), TempBelowTresh)
        GoodDomainCells = np.logical_or(CancerousAbove, NCancerousBelow)

        return np.sum(np.multiply(BadDomainCells, SqaredDifference)) / np.sum(np.multiply(GoodDomainCells, SqaredDifference))
        #return np.sum(- np.multiply(GoodDomainCells, SqaredDifference) + np.multiply(BadDomainCells, SqaredDifference))

    def showTreshMap(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(self.Domain)

        threshImage = np.where(self.maxHeat >= self.criticalTemp, 1, self.maxHeat)
        threshImage = np.where(self.maxHeat < self.criticalTemp, 0, threshImage)
        ax2.imshow(threshImage)

        plt.show()





def main():
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



    if True:
        op = np.array([0.47168048, 0.56576328, 0.07267478, 0.93457241, 0.50044206,
       0.55891838, 0.44451685, 0.49549059, 0.6350427 , 0.51845252,
       0.47714058, 0.52460771, 0.32905362, 0.72689334, 0.16415627,
       0.58697376, 0.33453024, 0.21717281, 0.73891988, 0.47865519,
       0.52441578, 0.54342482, 0.3571439 , 0.98271025, 0.21968704,
       0.49203765, 0.49688187, 0.57580805, 0.83006375, 0.21623442,
       0.52817129, 0.50942731, 0.90633827, 0.82053844, 0.11329717])



        operation = Operation(IntensityTemplate, domain, dx, dy, k, rho, c, dt)

        operation.addOperationScalar(op[0], op[1], op[2], op[3], op[4])
        operation.addOperationScalar(op[5], op[6], op[7], op[8], op[9])
        operation.addOperationScalar(op[10], op[11], op[12], op[13], op[14])
        operation.addOperationScalar(op[15], op[16], op[17], op[18], op[19])
        operation.addOperationScalar(op[20], op[21], op[22], op[23], op[24])
        operation.addOperationScalar(op[25], op[26], op[27], op[28], op[29])
        operation.addOperationScalar(op[30], op[31], op[32], op[33], op[34])

        operation.computeOperation()

        operation.showTreshMap()
        return


    genProgress = []
    for generation in range(600):
        solutions = []
        for _ in range(optimizer.population_size):
            op = optimizer.ask()

            operation = Operation(IntensityTemplate, domain, dx, dy, k, rho, c, dt)

            # Add an operation proceedures
            operation.addOperationScalar(op[0], op[1], op[2], op[3], op[4])
            operation.addOperationScalar(op[5], op[6], op[7], op[8], op[9])
            operation.addOperationScalar(op[10], op[11], op[12], op[13], op[14])
            operation.addOperationScalar(op[15], op[16], op[17], op[18], op[19])
            operation.addOperationScalar(op[20], op[21], op[22], op[23], op[24])
            operation.addOperationScalar(op[25], op[26], op[27], op[28], op[29])
            operation.addOperationScalar(op[30], op[31], op[32], op[33], op[34])

            # Compute the operation
            operation.computeOperation()

            # Compute the operation loss
            loss = operation.operationLoss()

            # Save the generational loss
            genProgress.append((generation, loss, op))

            solutions.append((op, loss))

            print("Generation: {}, Loss: {}".format(generation, loss))

        optimizer.tell(solutions)

    import pickle
    pickle.dump(genProgress, open('generationData.p', 'wb'))


    for solution in solutions:
        print(solution)



if __name__ == "__main__":
    main()
