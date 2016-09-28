# -*- coding: utf-8 -*-
"""
@author: NL
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def get_counts(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x]+=1
    return counts


"""
def sum_odds(frame, headings):
    temp = pd.DataFrame(0, )
    
    for heading in headings:
        temp += frame[heading]
     
    return temp/len(headings)
 """       
    
xls_file = pd.ExcelFile('data1_prob.xlsx')
table = xls_file.parse('sheet1')
frame = pd.DataFrame(table)

ftr = frame['FTR']
gamesHome = frame['HomeTeam']
gamesAway = frame['AwayTeam']
goals_home_FT = frame['FTHG']
goals_away_FT = frame['FTAG']
refs = frame['Referee']


num_matches = len(ftr)
ftrStats = get_counts(ftr)
refStats = get_counts(refs)

perHome = ftrStats['H']/num_matches
perAway = ftrStats['A']/num_matches
perDraw = ftrStats['D']/num_matches

total = perHome + perAway + perDraw

teamsAway = get_counts(gamesAway)
teamsHome = get_counts(gamesHome)

x = goals_home_FT
y = goals_away_FT
L = len(x)
"""radius of scatter plot could be frequency"""

color = np.random.rand(L)
plt.scatter(x,y, c=color, alpha=0.5) 
plt.show()

homeHeadings = ['B1H','B2H','B3H','B4H','B5H','B6H','B7H','B8H','B9H','B10H']
awayHeadings = ['B1A','B2A','B3A','B4A','B5A','B6A','B7A','B8A','B9A','B10A']
drawHeadings = ['B1D','B2D','B3D','B4D','B5D','B6D','B7D','B8D','B9D','B10D']

homeOdds = frame['B1H']+frame['B2H']+frame['B3H']+frame['B4H']+frame['B5H']+frame['B6H']+frame['B7H']+frame['B8H']+frame['B9H']
homeOdds/=9
homeOdds-=50

awayOdds = frame['B1A']+frame['B2A']+frame['B3A']+frame['B4A']+frame['B5A']+frame['B6A']+frame['B7A']+frame['B8A']+frame['B9A']
awayOdds/=9
awayOdds-=50

drawOdds = frame['B1H']+frame['B2H']+frame['B3H']+frame['B4D']+frame['B5D']+frame['B6D']+frame['B7D']+frame['B8D']+frame['B9D']
drawOdds/=9
drawOdds-=50

t = np.arange(0,1,0.1)

plt.subplot(2, 1, 1)
plt.plot(t,t,'r--')
plt.scatter(homeOdds, awayOdds)
plt.title('Odds')
plt.xlabel('Homeodds')
plt.ylabel('Awayodds')

plt.subplot(2, 1, 2)
plt.plot(t,t,'r--')
plt.scatter(homeOdds, drawOdds)
plt.xlabel('Homeodds')
plt.ylabel('Drawodds')

"""
plt.subplot(1, 3, 3)
plt.plot(t,t,'r--')
plt.scatter(awayOdds, drawOdds)
plt.xlabel('Awayodds')
plt.ylabel('Drawodds')
"""
plt.show()


"""  Precursor to scatterplot matrix   """

x = frame['B1H']
y = frame['BbMxH']
plt.scatter(x,y)
plt.plot(t,t,'r--')


x = frame['B1H']
y = frame['BbAvH']
plt.scatter(1/x,1/y)
plt.plot(16*t,16*t,'r--')


""" Convert FTR Column to Numerical """
ftrNums = ftr

for i in range(L):
    temp = ftr[i]
    if temp == 'D':
        ftrNums[i] = 1
    elif temp == 'A':
        ftrNums[i] = 1
    else:
        ftrNums[i] = 0


"""

point = frame[['B1H','B1A']].apply(tuple, axis=1)
xlist = point.tolist()
X = np.asarray(xlist)
y = ftrNums

'plt.scatter(*zip(*X))
y = ftrNums.tolist()
"""

""" Do Classification Scheme """
"""iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target"""


point = frame[['BbMx<2.5', 'BbAvAHH']]
X = point.as_matrix()
ytemp = pd.to_numeric(ftrNums)
y = ytemp.as_matrix() 

h = 0.02

C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X,y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

# create a mesh to plot ins
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Home Odds')
    plt.ylabel('Away Odds')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()



