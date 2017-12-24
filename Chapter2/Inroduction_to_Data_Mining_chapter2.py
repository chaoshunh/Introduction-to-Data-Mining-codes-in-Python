
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

#The scatter-matrix of iris
sns.set(style="ticks")
sns.pairplot(iris,hue="target")
plt.savefig("Scatter Matrix.png")
plt.show()

#check the basic statistics of iris dataset.
print(iris.info());print(iris.describe())

#Check duplicates rows and remove them
print('number of duplicated:',iris.duplicated().sum())
iris=iris.drop_duplicates()


#Aggregation by taget using mean
grouped_mean=iris.groupby('target').mean()
print(grouped_mean)

#Aggregation by target using median
grouped_median=iris.groupby('target').median()
print(grouped_median)

#Random Sampling
random_sampled=iris.take(np.random.permutation(len(iris))[:20])
print(random_sampled)

#scatter-plot on sampling
sns.set(style="ticks")
sns.pairplot(random_sampled,hue="target")
plt.savefig('sampling')
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

# Pinciple Component Analysis
from sklearn.decomposition import PCA
pca=PCA()
pca.fit(iris)
print(pca.explained_variance_ratio_) 

pca.n_components=2
iris_reduced=pca.fit_transform(iris)
# scatter plot of reduced irirs data sample
fig=plt.figure()
ax1=fig.add_subplot(111)
x_val=iris_reduced[:,0];y_val=iris_reduced[:,1]
colors=['red','k','blue']
ax1.scatter(x_val,y_val,c=iris.target.apply(lambda x:colors[int(x)]))
ax1.set_xlabel('Component1')
ax1.set_ylabel('Component2')
plt.show()

#correlation matrix
correlation_matrix=iris.corr()
mask = np.zeros_like(correlation_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, ax=ax)
plt.savefig('corr')
plt.show()













        




