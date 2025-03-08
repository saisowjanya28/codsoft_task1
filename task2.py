import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#train=r"C:\Users\91903\OneDrive\Desktop\INT\task2.csv"
#test=r"C:\Users\91903\OneDrive\Desktop\INT\task2.csv"

d1=pd.read_csv(r"C:\Users\91903\OneDrive\Desktop\INT\task2\archive (2)\fraudTest.csv")
d1=d1.sample(n=50000,random_state=42)
d2=pd.read_csv(r"C:\Users\91903\OneDrive\Desktop\INT\task2\archive (2)\fraudTrain.csv")
d2=d2.sample(n=20000,random_state=42)
d=pd.concat([d1,d2],ignore_index=True)

d=d.drop(columns=['trans_num','first','last','dob','street','city','state','zip','merchant','job'])

e=LabelEncoder()
d['category']=e.fit_transform(d['category'])
d['gender']=e.fit_transform(d['gender'])

d['trans_date_trans_time']=pd.to_datetime(d['trans_date_trans_time'])
d['hour']=d['trans_date_trans_time'].dt.hour
d['day']=d['trans_date_trans_time'].dt.day
d['month']=d['trans_date_trans_time'].dt.month
d=d.drop(columns=['trans_date_trans_time'])

x=d.drop(columns=['is_fraud'])
y=d['is_fraud']

s=StandardScaler()
x=s.fit_transform(x)

x1,x2,y1,y2=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

m={"Logistic Regression":LogisticRegression(),
   "RandomForest1":RandomForestClassifier(n_estimators=50,max_depth=10,n_jobs=-1,warm_start=True),
   "RandomForest2":RandomForestClassifier(n_estimators=100,max_depth=15,n_jobs=-1,warm_start=True)}

for n,v in m.items():
    v.fit(x1,y1)
    yp=v.predict(x2)
    yp1=v.predict_proba(x2)[:,1]

    r=pd.DataFrame({'Actual':y2.values,'Prob':yp1})
    r['Actual']=r['Actual'].map({0:'Legit',1:'Fraud'})

    f=r.sample(10,random_state=42)

    print("\n",n,"Model")
    print("Acc:",accuracy_score(y2,yp))
    print("Matrix:\n",confusion_matrix(y2,yp))
    print("\n Cases:")
    print(f)