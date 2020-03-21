# anomaly_detection.py
import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.cluster import KMeans

class AnomalyDetection(): 
    def cat2Num(self, df, indices):
        df['features'] = df.features.apply(lambda x: literal_eval(str(x))) #features columns contains string - convert string to list 
        df['feature1'] = df.features.apply(lambda x: x[0]) 
        df['feature2'] = df.features.apply(lambda x: x[1])
        df['feature3'] = df.features.apply(lambda x: [x[0],x[1]])
        feature1_list = list(df['feature1'].unique())
        feature2_list = list(df['feature2'].unique())

        
        feature1_list_len = len(feature1_list)
        feature2_list_len = len(feature2_list)
        
        feature1_matrix = np.identity(feature1_list_len).astype(int)
        feature2_matrix = np.identity(feature2_list_len).astype(int)
        
        feature1_encoding={}
        feature2_encoding={}
        
        index=0
        for item in feature1_list:
            feature1_encoding[item]=list(feature1_matrix[index])
            index=index+1
        #print(feature1_encoding)
        
        index=0
        for item in feature2_list:
            feature2_encoding[item]=list(feature2_matrix[index])
            index=index+1
        #print(feature2_encoding) 
        
        df['feature1_encoding'] = df.feature1.apply(lambda x: feature1_encoding[x])
        df['feature2_encoding'] = df.feature2.apply(lambda x: feature2_encoding[x])
        df['newfeatures'] = df.features.apply(lambda x: x[2:])
        df['features'] = df.apply(lambda x: x['feature1_encoding']+x['feature2_encoding']+x['newfeatures'], axis=1)
        return df    
    
    def scaleNum(self, df, indices):
        for index in indices:
            #print(index)
            df['scalefeature'] = df.features.apply(lambda x: x[index])
            mean=df['scalefeature'].mean()
            std=df['scalefeature'].std()
            if std!=0:
                df['prescalefeatures']=df.features.apply(lambda x: x[:index])         
                df['scalefeature'] = df.features.apply(lambda x:  [(x[index]-mean)/std])
                df['postscalefeatures']=df.features.apply(lambda x: x[index+1:])
                df['features']=df.apply(lambda x: x['prescalefeatures']+x['scalefeature']+x['postscalefeatures'], axis=1)
        return df
                          
            
    def detect(self, df, k, t):
        #X = np.array(df['features'].values.tolist())
        X_list = df['features'].values.tolist()
        X = pd.DataFrame(X_list)
        kmeans = KMeans(n_clusters=k,random_state=32)  #Experimented with random parameter values : Least number of anomalies detected using this parameter! 
        kmeans.fit(X)
        labels = kmeans.labels_
        
        df1 = pd.DataFrame()    
        label_counts = list(np.bincount(labels))#add the cluster array to df as a column
        df1['cluster'] = list(labels)
        df = pd.concat([df['features'],df1['cluster']], axis=1)
        clusters={}     #dictionary values of all clusters and respective counts
        cluster=0       #initial cluster and iterative increase within next loop
        
        #print("Label_counts:",label_counts)
        Nmax = max(label_counts)
        Nmin = min(label_counts)
        #print("Nmax:",Nmax)
        #print("Nmin:",Nmin)
        Denominator = Nmax-Nmin
        #print("Denominator:",Denominator)
        for count in label_counts:
            #print("cluster:",cluster,"count:",count)
            clusters[cluster]=count
            cluster=cluster+1
        #print(clusters)
        
        df['score'] = df.cluster.apply(lambda x: (Nmax-clusters[x])/(Nmax-Nmin))
        df = df[df['score']>t]
        return df
    
 
if __name__ == "__main__":
    
    df = pd.read_csv('logs-features-sample.csv').set_index('id')
    ad = AnomalyDetection()
    
    df1 = ad.cat2Num(df, [0,1])
    print(df1.features.head(5))

    feature_len = len(df1.features[0])

    df2 = ad.scaleNum(df1, [*range(12,feature_len)])  #Passing list of features to be standardized
    #df2 = ad.scaleNum(df1, [12])
    print(df2.features.head(5))

    df3 = ad.detect(df2, 8, 0.97)
    print(df3.head(5))
    print(df3.size)
