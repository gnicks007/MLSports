# -*- coding: utf-8 -*-
"""
@author: NL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree

def dataCols():
    columnNames = ['B1H','B1D','B1A','B2H','B2D','B2A','B3H','B3D',\
                    'B3A','B4H','B4D','B4A','B5H','B5D','B5A','B6H',\
                    'B6D','B6A','B7H','B7D','B7A','B8H','B8D','B8A',\
                    'B9H','B9D','B9A','BbMxH','BbAvH','BbMxD',\
                    'BbAvD','BbMxA','BbAvA','BbMx>2.5','BbAv>2.5',\
                    'BbMx<2.5','BbAv<2.5','BbMxAHH',\
                    'BbAvAHH','BbMxAHA','BbAvAHA']
    return columnNames
    

def convertOddsToProb(dataframe, sequence):
    for x in sequence:
        dataframe[x] = 100/dataframe[x]
    return dataframe


def exportFrameToExcelFile(dataframe, filename, sheet):
    writer = pd.ExcelWriter(filename)
    dataframe.to_excel(writer, sheet)
    writer.save()
    

""" Choose a subset of columns in a DataFrame """
def specifyFrame(dataframe, columnNames):
    sFrame = pd.DataFrame(dataframe, columns=columnNames)
    return sFrame
    
def exportOddsToProbsExcel():
    xsl_file = pd.ExcelFile('data1_odds.xlsx')
    table = xsl_file.parse('sheet1')
    frame = pd.DataFrame(table)
    
    colNames = dataCols()
    newFrame = convertOddsToProb(frame, colNames)
    exportFrameToExcelFile(newFrame, 'data1_prob.xlsx', 'sheet1')

def scattermatrixPlot(dataframe, columns):
    df = specifyFrame(dataframe, columns)
    pd.scatter_matrix(df, alpha=0.2, figsize=(6,6), diagonal='kde')
    
 
def convCatToNum(dataframe):
    length = len(dataframe)
    df = dataframe
    
    for i in range(length):
        temp = dataframe[i]
        
        if temp == 'H':
            df[i] = 0
        elif temp == 'A':
            df[i] = 1
        else:
            df[i] = 2
    return df
        

def decisionTreeAndPlot(X, y, cols):
    length = X.shape[0]
    idx = np.arange(length)
    
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X-mean)/std

    clf = tree.DecisionTreeClassifier().fit(X, y)
    
    plt.subplot(1,1,1)
    
    h = 0.02
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h ),
                         np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.axis("tight")
 
    n_classes = 3
    plot_colors = "bry"
    targetLabels = ['Home wins', 'Away Wins', 'Draw']
    
    newtuple = zip(range(n_classes), plot_colors)
    
    for i, color in newtuple:
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx,1], c=color, label=targetLabels[i], 
                    cmap=plt.cm.Paired)

    plt.axis("tight")
    plt.suptitle("Decision tree using {x} & {y} for EPL2005-2006".format(x=cols[0], y=cols[1]))
    """plt.legend() """
    plt.show()
    
    
def loadFileInFrame(file, sheet):
    """ Import Excel File and Create DataFrame """
    xsl_file = pd.ExcelFile(file)
    table = xsl_file.parse(sheet)
    frame = pd.DataFrame(table)
    return frame
    
    
def main():
    
    frame = loadFileInFrame('data1_prob.xlsx','sheet1')
    
    columnNames = dataCols() 
    columnNames.insert(0, 'FTR')
    scatterFrame = specifyFrame(frame, columnNames)
    
    """ Scatter Plot """
    newFrame = scatterFrame;
    newFrame['FTR'] = convCatToNum(scatterFrame['FTR'])


    """ Convert to Numpy Arrays """
    
    target = newFrame['FTR']
    y = np.asarray(target, dtype="int64")
  
    cols = ['B1H', 'BbMx<2.5']
    combo1 = specifyFrame(newFrame, cols)    
    X = combo1.as_matrix()
     
    length = X.shape[0]
    idx = np.arange(length)
    
    mean = X.mean(axis=0)
    X = X-mean
    
    """do Decision trees with X = H+A, H+D, A+D, A, D, H Save to spreadsheet"""
    for i in [0, 1, 2]:
        idx = np.where(y == i)
        if(i == 0):
            Xhome = X[idx]
        elif(i == 1):
            Xaway = X[idx]
        else:
            Xdraw = X[idx]
            
    
    """ Histogram """
  
    hist, bins = np.histogram(X[:,0], 100)
    width = 0.7 * (bins[1] - bins[0])
    """center = (bins[:-1] + bins[1:]) / 2 """
    center = [0.5 * (bins[i] + bins[i+1]) for i in range(len(hist))]
    plt.bar(center, hist, align='center', width=width)
    plt.show() 
    
    histTotal, bins = np.histogram(X[:,0], 100)
    stay0 = np.where(histTotal!=0)  
    center = np.reshape(center, (100,1))
    
  
    
    plt.scatter(center, hist)
    
       
    """plt.scatter(center[stay0], hist[stay0])"""
    
   
    
main()

