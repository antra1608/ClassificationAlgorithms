import matplotlib.pyplot as plt
from matplotlib import style 
style.use('ggplot')
import numpy as np
import csv
dataset=[]

with open('dataset.csv','rb') as f:
    mycsv=csv.reader(f)
    mycsv=list(mycsv)
    for x in range(len(mycsv)-1):
        for y in range(1):
            #mycsv[x][y]=float(mycsv[x][y])
            text= mycsv[x][y].split('\t')
            #print text[2]
            dataset.append([round(float(text[0])),round(float(text[1]))])
X=np.asarray(dataset)
#X=np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

#plt.scatter(X[:,0],X[:,1],s=150)
#plt.show()

colors=["k","r","c","b","k"]
#tol=how far centroid can move
class K_Means:
    def __init__(self,k=2,tol=0.001,max_iter=300):
        self.k=k
        self.tol=tol
        self.max_iter=max_iter

    def fit(self,data):
        self.centroids={}

        for i in range(self.k):
            self.centroids[i]=data[i]

        for i in range(self.max_iter):
            self.classifications={}
            for i in range(self.k):
                self.classifications[i]=[]

            for featureset in X:
                distances=[np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification=distances.index(min(distances))
                self.classifications[classification].append(featureset)
 
            prev_centroids=dict(self.centroids)

            for classification in self.classifications:
               # pass
                self.centroids[classification]=np.average(self.classifications[classification],axis=0)
            optimized=True

            for c in self.centroids:
                original_centroid=prev_centroids[c]
                current_centroid=self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.00)>self.tol:
                    optimized=False

 
            if optimized:
                 break

    def predict(self,data):
        distances=[np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification=distances.index(min(distances))
        return classification        


clf=K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker="o",color="g",s=150,linewidth=5)
#'''
for classification in clf.classifications:
    #c=colors[classification]
    #print c
    print "classification"
    print classification
    for featureset in clf.classifications[classification]:
         plt.scatter(featureset[0],featureset[1],marker="x",color=colors[classification],s=150,linewidths=5)
         print featureset 
plt.show()




















