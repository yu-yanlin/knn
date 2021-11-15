from numpy import *
import operator
import time
from os import listdir


def classify(inputPoint, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]     # 已知分类的数据集（训练集）的行数
    # 先 tile 函数将输入点拓展成与训练集相同维数的矩阵，再计算欧氏距离
    diffMat = tile(inputPoint, (dataSetSize, 1)) - dataSet  # 样本与训练集的差值矩阵
    sqDiffMat = diffMat ** 2                    # 差值矩阵平方
    sqDistances = sqDiffMat.sum(axis=1)         # 计算每一行上元素的和
    distances = sqDistances ** 0.5              # 开方得到欧拉距离矩阵
    sortedDistIndicies = distances.argsort()    # 按distances中元素进行升序排序后得到的对应下标的列表
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    # 按classCount字典的第2个元素（即类别出现的次数）从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 文本向量化 32x32 -> 1x1024
def img2vector(filename):
    returnVect = []
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect


# 从文件名中解析分类数字
def classnumCut(fileName):
    fileStr = fileName.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    return classNumStr


# 构建训练集数据向量，及对应分类标签向量
def trainingDataSet():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')         # 获取文件目录内容
    m = len(trainingFileList)       # 训练集数量
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # 获得训练集的标签
        hwLabels.append(classnumCut(fileNameStr))
        # 训练集数据向量
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    return hwLabels, trainingMat


# 测试函数
def handwritingTest():
    hwLabels,trainingMat = trainingDataSet()    # 构建训练集
    testFileList = listdir('digits/testDigits')        # 获取测试集
    errorCount = 0.0                            # 错误数
    mTest = len(testFileList)                   # 测试集总样本数
    t1 = time.time()
    for i in range(mTest):
        fileNameStr = testFileList[i]
        # 获得实际数字
        classNumStr = classnumCut(fileNameStr)
        # 测试文本向量化
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        # 调用knn算法进行测试           测试文本向量化   训练文本向量化    标签
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("classifier 返回值为: %d, 真实的值为: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n测试的总数是: %d" % mTest)             # 输出测试总样本数
    print("错误总数为: %d" % errorCount)           # 输出测试错误样本数
    print("总错误率为: %f" % (errorCount/float(mTest)))  # 输出错误率
    t2 = time.time()
    print("花费时间: %.2fmin, %.4fs." % ((t2-t1)//60, (t2-t1) % 60))      # 测试耗时


if __name__ == "__main__":
    handwritingTest()