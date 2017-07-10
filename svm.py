import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class Support_Vector_Machine:
    def __init__(self,visualization=True):
         self.visualization=visualization
         self.colors={1:'r',-1:'b'}
         if self.visualization:
             self.fig=plt.figure()
             self.ax=self.fig.add_subplot(1,1,1)
    #train  
    def fit(self,data):
         self.data=data
         #{||w||:[w,b]}
         opt_dict={}
         transforms=[[1,1],[-1,1],[-1,-1],[1,-1]]
         
         all_data=[]
         for yi in self.data:
             for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

         self.max_feature_value=max(all_data)
         self.min_feature_value=min(all_data)

         all_data=None
         
         step_sizes=[self.max_feature_value*0.1,self.max_feature_value*0.01,self.max_feature_value*0.001,
]

   
	 #expensive
         b_range_multiple=2
         b_multiple=5
         latest_optimum=self.max_feature_value*10


         for step in step_sizes:
             print "step size is"
             print step
             w=np.array([latest_optimum,latest_optimum])
             print "w initially"
             print w 
#we can do this bcz optimized
             optimized=False
             while not optimized:
                 for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                    self.max_feature_value*b_range_multiple,
                                    step*b_multiple):
                     for transformation in transforms:
                         w_t=w*transformation
                         found_option=True
                         #weakest link in svm fundamentally
                         for i in self.data:
                             for xi in self.data[i]:
                                 y1=i
                                 if not yi*(np.dot(w_t,xi)+b)>=1:
                                     found_option=False
                                 #print(xi,' : ',yi*(np.dot(w_t,xi)+b))    

                         if found_option:
                             opt_dict[np.linalg.norm(w_t)]=[w_t,b]
              
                
                 print w[0]
                 if w[0]<0: 
		     optimized=True
                     print('Optimized a step')
                 else:
                     w=w-step 
             norms=sorted([n for n in opt_dict])
             opt_choice=opt_dict[norms[0]]
             print 'opt choice'
             print opt_choice
             self.w=opt_choice[0]
             self.b=opt_choice[1]     
             latest_optimum=opt_choice[0][0]+step*2
             print 'latest_optimum'
             print latest_optimum

         for i in self.data:
             for xi in self.data[i]:
                 y1=i
                 print(xi,' : ',yi*(np.dot(self.w,xi)+self.b))    
         
#'''    

    def predict(self,features):
      #sign(x.w+b) 
        classification=np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification!=0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker='*',c=self.colors[classification])
            print classification
        return classification


    def visualize(self):
        print "%%%%%%%%%%%%%%w"
        print self.w
        print "$$$$$$$$$$$$$$b"
        print self.b   

#        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
#        plt.show()
#hyperplane=x.w+b
#v=x.w+b 
#positive suport vector=1
#negative support vector=-1
#dec=0
'''
        def hyperplane(x,w,b,v):
            print "#################hyperplane"
            print w[0]
            print w[1]
            print b
            return (-w[0]*x-b+v) /w[1]
        datarange=(0.9*self.min_feature_value,1.1*self.max_feature_value)
        hyp_x_min=datarange[0]
        hyp_x_max=datarange[1]

        psv1=hyperplane(hyp_x_min,self.w,self.b,1)
        psv2=hyperplane(hyp_x_max,self.w,self.b,1)
        print "psv1"
        print psv1
        print "psv2"
        print psv2
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')

        nsv1=hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2=hyperplane(hyp_x_max,self.w,self.b,-1)
        print "nsv1"
        print nsv1
        print "nsv2"
        print nsv2
              
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')


        db1=hyperplane(hyp_x_min,self.w,self.b,0)
        db2=hyperplane(hyp_x_max,self.w,self.b,0)
        print "db1"
        print db1
        print "db2"
        print db2
        
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'y--')
'''
        #plt.show()



import csv
pos=[]
neg=[]
with open('dataset.csv','rb') as f:
    mycsv=csv.reader(f)
    mycsv=list(mycsv)
    for x in range(len(mycsv)-1):
        for y in range(1):
            #mycsv[x][y]=float(mycsv[x][y])
            text= mycsv[x][y].split('\t')
            print text[2]
            if text[2] is "1":
                pos.append([round(float(text[0])),round(float(text[1]))])
            else:
                neg.append([round(float(text[0])),round(float(text[1]))]) 
#    print pos
data_dict={-1:neg,1:pos}
svm=Support_Vector_Machine()
svm.fit(data=data_dict)

#predict_us=[[0,10],[1,3],[3,4],[5,5],[5,6],[6,-5],[5,8],]
#for p in predict_us:
 #   svm.predict(p)
    
svm.visualize()

