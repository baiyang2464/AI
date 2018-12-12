#conding:utf-8
'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from numpy import *
from time import sleep
import csv

#to storage model data
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        
        self.svInd=[]
        self.sVs=[]
        self.labelSV = []

class SVM:
    def __init__(self,trainDataPath='./train.csv',testDataPath='./test.csv',C=1.0,toler=0.0001,maxIter=10000,kTup=('lin',0)):
        self._trainDataPath= trainDataPath
        self._testDataPath = testDataPath
        self._C = C
        self._toler=toler
        self._maxIter = maxIter
        self._kTup = kTup

        #will be initialized in smoP()
        self._oS =[]
        self._dataArr=[]
        self._labelArr=[]

    def loadDataSet(self,fileName):
        dataMat = []; labelMat = []
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
        return dataMat,labelMat

    def selectJrand(self,i,m):
        j=i #we want to select any J not equal to i
        while (j==i):
            j = int(random.uniform(0,m))
        return j

    def clipAlpha(self,aj,H,L):
        if aj > H: 
            aj = H
        if L > aj:
            aj = L
        return aj

    def kernelTrans(self,X, A, kTup): #calc the kernel or transform data to a higher dimensional space
        m,n = shape(X)
        K = mat(zeros((m,1)))
        if kTup[0]=='lin': K = X * A.T   #linear kernel
        elif kTup[0]=='rbf':
            for j in range(m):
                deltaRow = X[j,:] - A
                K[j] = deltaRow*deltaRow.T
            K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
        else: raise NameError('Houston We Have a Problem -- \
        That Kernel is not recognized')
        return K

            
    def calcEk(self,_oS, k):
        fXk = float(multiply(_oS.alphas,_oS.labelMat).T*_oS.K[:,k] + _oS.b)
        Ek = fXk - float(_oS.labelMat[k])
        return Ek
            
    def selectJ(self,i, _oS, Ei):         #this is the second choice -heurstic, and calcs Ej
        maxK = -1; maxDeltaE = 0; Ej = 0
        _oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
        validEcacheList = nonzero(_oS.eCache[:,0].A)[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
                if k == i: continue #don't calc for i, waste of time
                Ek = self.calcEk(_oS, k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k; maxDeltaE = deltaE; Ej = Ek
            return maxK, Ej
        else:   #in this case (first time around) we don't have any valid eCache values
            j = self.selectJrand(i, _oS.m)
            Ej = self.calcEk(_oS, j)
        return j, Ej

    def updateEk(self,_oS, k):#after any alpha has changed update the new value in the cache
        Ek = self.calcEk(_oS, k)
        _oS.eCache[k] = [1,Ek]
            
    def innerL(self,i, _oS):
        Ei = self.calcEk(_oS, i)
        if ((_oS.labelMat[i]*Ei < -_oS.tol) and (_oS.alphas[i] < _oS.C)) or ((_oS.labelMat[i]*Ei > _oS.tol) and (_oS.alphas[i] > 0)):
            j,Ej = self.selectJ(i, _oS, Ei) #this has been changed from selectJrand
            alphaIold = _oS.alphas[i].copy(); alphaJold = _oS.alphas[j].copy();
            if (_oS.labelMat[i] != _oS.labelMat[j]):
                L = max(0, _oS.alphas[j] - _oS.alphas[i])
                H = min(_oS.C, _oS.C + _oS.alphas[j] - _oS.alphas[i])
            else:
                L = max(0, _oS.alphas[j] + _oS.alphas[i] - _oS.C)
                H = min(_oS.C, _oS.alphas[j] + _oS.alphas[i])
            if L==H: print "L==H"; return 0
            eta = 2.0 * _oS.K[i,j] - _oS.K[i,i] - _oS.K[j,j] #changed for kernel
            if eta >= 0: print "eta>=0"; return 0
            _oS.alphas[j] -= _oS.labelMat[j]*(Ei - Ej)/eta
            _oS.alphas[j] = self.clipAlpha(_oS.alphas[j],H,L)
            self.updateEk(_oS, j) #added this for the Ecache
            if (abs(_oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
            _oS.alphas[i] += _oS.labelMat[j]*_oS.labelMat[i]*(alphaJold - _oS.alphas[j])#update i by the same amount as j
            self.updateEk(_oS, i) #added this for the Ecache                    #the update is in the oppostie direction
            b1 = _oS.b - Ei- _oS.labelMat[i]*(_oS.alphas[i]-alphaIold)*_oS.K[i,i] - _oS.labelMat[j]*(_oS.alphas[j]-alphaJold)*_oS.K[i,j]
            b2 = _oS.b - Ej- _oS.labelMat[i]*(_oS.alphas[i]-alphaIold)*_oS.K[i,j]- _oS.labelMat[j]*(_oS.alphas[j]-alphaJold)*_oS.K[j,j]
            if (0 < _oS.alphas[i]) and (_oS.C > _oS.alphas[i]): _oS.b = b1
            elif (0 < _oS.alphas[j]) and (_oS.C > _oS.alphas[j]): _oS.b = b2
            else: _oS.b = (b1 + b2)/2.0
            return 1
        else: return 0

    def smoP(self,dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
        tmp_oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
        for i in range(tmp_oS.m):
            tmp_oS.K[:,i] = self.kernelTrans(tmp_oS.X, tmp_oS.X[i,:],self._kTup)
        iter = 0
        entireSet = True; alphaPairsChanged = 0
        while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:   #go over all
                for i in range(tmp_oS.m):        
                    alphaPairsChanged += self.innerL(i,tmp_oS)
                    print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                iter += 1
            else:#go over non-bound (railed) alphas
                nonBoundIs = nonzero((tmp_oS.alphas.A > 0) * (tmp_oS.alphas.A < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerL(i,tmp_oS)
                    print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                iter += 1
            if entireSet: entireSet = False #toggle entire set loop
            elif (alphaPairsChanged == 0): entireSet = True  
            print "iteration number: %d" % iter
        return tmp_oS

    def calcWs(self,alphas,dataArr,classLabels):
        X = mat(dataArr); labelMat = mat(classLabels).transpose()
        m,n = shape(X)
        w = zeros((n,1))
        for i in range(m):
            w += multiply(alphas[i]*labelMat[i],X[i,:].T)
        return w

    def loadImages(self,filename):
        fr = open(filename)
        hwLabels = []
        rows = -1
        for rows,line in enumerate(fr):
                pass
        rows+=1
        trainingMat = zeros((rows,785))
        fr.close()
        fr = open(filename)
        f_csv=csv.reader(fr)
        i =0
        for row in f_csv:
            for j in range(0,785):
                data = row[j]
                if j==0:
                    if data =='0':hwLabels.append(0)
                    elif data == '1':hwLabels.append(1)
                    elif data == '2':hwLabels.append(2)
                    elif data == '3':hwLabels.append(3)
                    elif data == '4':hwLabels.append(4)
                    elif data == '5':hwLabels.append(5)
                    elif data == '6':hwLabels.append(6)
                    elif data == '7':hwLabels.append(7)
                    elif data == '8':hwLabels.append(8)
                    elif data == '9':hwLabels.append(9)
                else:
                    if data > '0':
                        trainingMat[i][j-1]=1
                    else: trainingMat[i][j-1]=0
            i+=1
        fr.close()
        return trainingMat,hwLabels

    #make label by specific num
    def setLabel(self,num):
        tmpLabel = []
        m = len(self._labelArr)
        for i in range(m):
            if self._labelArr[i]==num:
                tmpLabel.append(1)
            else:tmpLabel.append(-1)
        return tmpLabel

    #train model for nums:0-9
    def trainModel(self):
        num =4
        self._dataArr,self._labelArr = self.loadImages(self._trainDataPath)
        tmpLabel=self.setLabel(num)
        tmp_oS = self.smoP(self._dataArr, tmpLabel,self._C, self._toler,self._maxIter, self._kTup)
        datMat=mat(self._dataArr); labelMat = mat(tmpLabel).transpose()
        tmp_oS.svInd=nonzero(tmp_oS.alphas.A>0)[0]
        tmp_oS.sVs=datMat[tmp_oS.svInd] 
        tmp_oS.labelSV = labelMat[tmp_oS.svInd];
        print "there are %d Support Vectors" % shape(tmp_oS.sVs)[0]
        self._oS.append(tmp_oS)

    def returnMaxNo(self,nums):
        maxn = nums[0]
        k=0
        m = len(nums)
        for i in range(m):
            if maxn<nums[i]:k=i
        return k

    #make predict and get accuracy
    def getPredicton(self):
        dataArr,labelArr = self.loadImages('./test.csv')
        errorCount = 0
        successCount =0
        datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
        m,n = shape(datMat)
        for i in range(m):
            kernelEval = self.kernelTrans(self._oS[0].sVs,datMat[i,:],self._kTup)
            predict=kernelEval.T * multiply(self._oS[0].labelSV,self._oS[0].alphas[self._oS[0].svInd]) + self._oS[0].b
            if sign(predict)==1:
                if labelArr[i]!=4: errorCount += 1    
                else: print"predict right %dst num is 4"%(i+1)
        print "the test error rate is: %f" % (float(errorCount)/m) 

def main():
    trainDataPath = './train.csv'
    testDataPath = './test.csv'
    C= 1.0
    toler = 0.0001
    maxIter = 10000
    kTup=('rbf',20)
    SVM_model = SVM(trainDataPath,testDataPath,C,toler,maxIter,kTup)
    SVM_model.trainModel()
    SVM_model.getPredicton()

if __name__== '__main__':
    main()


