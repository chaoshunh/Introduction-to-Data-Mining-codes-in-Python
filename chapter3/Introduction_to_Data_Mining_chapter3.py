
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns


#load the iris data
data=load_iris()
data.target.shape=(len(data.data),1)
#concatenate the target column and the feature columns
new_data=np.concatenate((data.data,data.target),axis=1)
iris=pd.DataFrame(new_data,columns=['sepal_length','sepal_width','petal_length','petal_width','target'])
print(iris.head(10))

#basic statistics
print(iris.info())

#Mean,SD,median,min and max to each column.
iris_mean=iris.ix[:,0:4].dropna().mean(axis=0)
iris_sd=iris.ix[:,0:4].dropna().std(axis=0)
iris_median=iris.ix[:,0:4].dropna().median(axis=0)
iris_min=iris.ix[:,0:4].dropna().max(axis=0)
iris_max=iris.ix[:,0:4].dropna().min(axis=0)


#Correlation matrix
iris_corr=iris.corr()
print(iris_corr)

#Covariance matrix
iris_cov=iris.cov()
print(iris_cov)

#quantile
iris_quantile=iris.quantile([0.0,0.25,0.5,0.75,1.0])
print(iris_quantile)

#Histogram of three features.
fig=plt.figure(figsize=(8,8))

ax1=fig.add_subplot(2,2,1)
sepal_length=iris.sepal_length
plt.hist(sepal_length,bins=20)
plt.xlabel('sepal_length')
plt.ylabel('count')

sepal_width=iris.sepal_width
ax2=fig.add_subplot(2,2,2)
plt.hist(sepal_width,bins=20)
plt.xlabel('sepal_width')
plt.ylabel('count')

petal_length=iris.petal_length
ax3=fig.add_subplot(2,2,3)
plt.hist(petal_length,bins=20)
plt.xlabel('petal_length')
plt.ylabel('count')

petal_width=iris.petal_width
ax4=fig.add_subplot(2,2,4)
plt.hist(petal_width,bins=20)
plt.xlabel('petal_width')
plt.ylabel('count')
plt.savefig('histogram')
plt.show()

#Box plot
fig=plt.figure()
ax=fig.add_subplot(111)
data=[sepal_length,sepal_width,petal_length,petal_width]
ax.boxplot(data)
ax.set_xticklabels(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
ax.set_ylabel('value cm')
plt.show()

#Empirical Cumulative Distribution Function, ECDF
from matplotlib import mlab
fig,ax=plt.subplots(figsize=(8,4))
mu=200
sigma=25
n_bins=20
#plot the cumulative histogram
n,bins,patches=ax.hist(iris.sepal_length,normed=1,histtype='step',cumulative=True,label='Empirical')
#Add a line showing the expected distribution
y=mlab.normpdf(bins,mu,sigma).cumsum()
y/=y[-1]
ax.plot(bins,y,'k--',linewidth=1.5,label='Theoretical')

ax.grid(True)
ax.legend(loc='right')
ax.set_title('ECDF')
ax.set_xlabel('sepal length')
ax.set_ylabel('Likelihood of occurrence')
plt.show()

#Scatter plot matrix
sns.set(style="ticks")
fig, ax = plt.subplots()
sns.pairplot(iris,hue="target",size=1.2,aspect=1.2)
plt.savefig("Scatter Matrix2.png")
plt.show()

#Plot 3D scatter plot for the first three features.
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
colors=['red','k','blue']
x_vals=iris.sepal_length; y_vals=iris.sepal_width; z_vals=iris.petal_length
ax.scatter(x_vals,y_vals,z_vals,c=iris.target.apply(lambda x: colors[int(x)]))
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_length')
plt.show()

#Scatter plot of  petal_length and petal_width
fig=plt.figure()
ax_scatter=fig.add_subplot(111)
data_0=iris.ix[iris.target==0,['petal_length','petal_width']]
data_1=iris.ix[iris.target==1,['petal_length','petal_width']]
data_2=iris.ix[iris.target==2,['petal_length','petal_width']]

ax_scatter.scatter(data_0.petal_length,data_0.petal_width,c='red',label='Setosa')
ax_scatter.scatter(data_1.petal_length,data_1.petal_width,c='k',label='Vericolour')
ax_scatter.scatter(data_2.petal_length,data_2.petal_width,c='blue',label='Virginica')
ax_scatter.set_xlabel('Petal length')
ax_scatter.set_ylabel('Petal width')
ax_scatter.legend(loc='best')
plt.show()

#Parallel coordinates
from pandas.plotting import parallel_coordinates
#Replace the values 0,1 and 2 in column 'target' by their corresponding flower's names
mapping={0:'Setosa',1:'Virginica',2:'Versicolour'}
iris_new=iris.copy()
iris_new.target=iris_new.target.apply(lambda x: mapping[int(x)])

fig=plt.figure()
parallel_coordinates(iris_new,'target',alpha=0.5)
plt.show()














