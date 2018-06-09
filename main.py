from numpy import *
import KMeans
datMat=mat(KMeans.loadDataSet('testSet.txt'))
a=[1,2,3]
centroids=KMeans.randCent(datMat,2)
print(centroids)

#将数据聚类4个质心
#print(KMeans.kMeans(datMat,4))

#测试数据集testSet2.txt  使用二分 Kmeans算法
datMat2=mat(KMeans.loadDataSet('testSet2.txt'))
centList,myNewAssments=KMeans.biKmeans(datMat2,3)

print(myNewAssments)