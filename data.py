

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import xgboost as xgb
data  = pd.read_csv('Data Set CSV.csv')
from sklearn.preprocessing import LabelEncoder
lbl_enc=LabelEncoder()
dataa  = pd.read_csv('Data Set CSV.csv')
dataa.head()
dataa['Q3A']=dataa['Q3A'].replace(to_replace=['18-25','26-35','36-45','46-55','56-67','68+'],value=[5,4,3,2,1,0])
data = data.replace(to_replace =["No","Yes","Female","Male"], value =[0,1,0,1])
df_slice_24A = pd.get_dummies(data["Q24A"], prefix='A_')
df_slice_24A = df_slice_24A.drop(["A__DK", "A__Other"], axis=1)
df_slice_24B = pd.get_dummies(data["Q24B"], prefix='L_')
df_slice_24B = df_slice_24B.drop(["L__DK", "L__Other"], axis=1)
df = data.drop(data[data.Q22_11_other == "Transit"].index)
df = data.drop(data[data.Q22_11_other == "Aeroport"].index)
df = data.drop(data[data.Q22_11_other == "Other"].index)
df = data.reset_index(drop=True)
data["Limba"]=data["Limba"].replace(to_replace=['Rom-Rus','Engleza'],value=[1,2])
data=data.drop('Locality',axis=1)
data=data.drop('Sample',axis=1)
data=data.drop(['Type_of_tourist','id'],axis=1)
data['Q5'] = data['Q5'].replace(to_replace =["Individual (backpacker) – planned trip alone","Group visitor (organized by an agency/ tour-operator)","Business tourism","Moldovan diaspora who traveled in Moldova"], value =[3,2,1,0])
data['Q6_1']=data['Q6_1'].str.lower()
data['Q6_1']=data['Q6_1'].replace(np.nan,'0')
words_to_replace = ['shopping','null','needs','footbal','buy','funeral','documents','shoping','home','teeth','weeding','dk','bussines','employer','cheap','university',]
# #words_to_replace_tuple = tuple(words_to_replace)
pattern = '|'.join(words_to_replace)
data.loc[data['Q6_1'].str.contains(pattern), 'Q6_1'] = '0'
data.loc[data['Q6_1'] != '0', 'Q6_1'] = 1
data=data.drop(['Q6_2','Q6_3'],axis=1)
data=data.drop(['Q7_1','Q7_2','Q7_3'],axis=1)
enc_nom_1 = (data.groupby('Q1').size())*100 / len(data)
# #print(enc_nom_1)
enc_nom_2 = (data.groupby('Q2').size())*100/ len(data)
data['Q1'] = data['Q1'].apply(lambda x : enc_nom_1[x])
data['Q2'] = data['Q2'].apply(lambda x : enc_nom_2[x])
data=data.drop('Q3A',axis=1)
data = data.replace(to_replace =["Individual (backpacker) – planned trip alone","Group visitor (organized by an agency/ tour-operator)","Business tourism","Moldovan diaspora who traveled in Moldova"], value =[3,2,1,0])
data = data.drop('Q9_2',axis=1)
data['Q13'] = data['Q13'].replace(to_replace =['Health, cure','Transit','Study, educational project, exchange of experience, exchange','Wedding','Documents','Marathon, competition, contest','Other, specify','Charity','Funeral'],value=0)
data['Q13'] = data['Q13'].replace(to_replace =["Leisure tourism","Visiting relatives, friends","Business purposes","DK"], value =[3,2,1,0])
for j in range(1,15):
   data["Q14_"+str(j)].replace(np.nan,0)
data=data.drop(['Q14_99','Q14_14_other'],axis=1)
data['Q15'] = data['Q15'].replace(to_replace =["To a great extent","To some extent","To a small extent","Not at all","DK"],value=[4,3,2,1,0] )
data=data.drop(['Q16B_99','Q16B_11_other'],axis=1)
enc_nom_18 = (data.groupby('Q18').size()) / len(data)
data['Q18'] = data['Q18'].apply(lambda x : enc_nom_18[x])
data['Q19'] = data['Q19'].replace(to_replace =["Yes, I planned","No, I did not plan but visited some tours, excursions, visits","Did not plan, didn’t visit"],value=[2,1,0])
data['Q19']=data['Q19'].replace(np.NaN,0)
data['Q31'] = data['Q31'].map({'I would surely recommend' : 10, 9.0 : 9, 8.0 : 8, 7.0 : 7, 6.0 : 6, 5.0 : 5, 4.0: 4, 3.0 : 3, 2.0 : 2, 'I would not surely recommend' : 1, 'DK' : np.nan})
data['Q38'] = data['Q38'].map({'Master’s degree, Ph. D.' : 9, 'Complete higher education (Bachelor’s degree)' : 8, 'Incomplete higher education' : 7, 'College' : 6, 'High school' : 5, 'Vocational school' : 4, 'General school' : 3, 'Incomplete general school' : 2, 'No completed studies' : 1, 99.0 : np.nan, 'Other (Please specify)' : np.nan})

enc_nom_39 = (data.groupby('Q39').size())*10 / len(data)
#data['Q18'] = data['Q18'].apply(lambda x : enc_nom_18[x])
data['Q39'] = data['Q39'].apply(lambda x : enc_nom_39[x])
data['Q39'] = data['Q39'].replace(8.0,'DK')

df_slice=data.iloc[:,116:200]
df_slice = df_slice.replace(to_replace =['I did not visit','I visited','Very satisfied','Satisfied','Neither satisfied, nor dissatisfied','Dissatisfied','Very dissatisfied'] ,value=[1,0,5,4,3,2,1])
df_slice=df_slice.replace('DK/ NR',np.nan)
df_slice=df_slice.replace(np.NaN,-1)
df_slice=df_slice.replace('DK/ NR',-1)

df_slice_28=data.iloc[:,200:219]

df_slice_28=data.iloc[:,200:219]
df_slice_28=df_slice_28.drop('Q28other',axis=1)
df_slice_28=df_slice_28.replace('DK',-1)
df_slice_28=df_slice_28.replace('Not applicable',-1)
df_slice_28=df_slice_28.replace(to_replace=['Very satisfied','Rather satisfied','Neither satisfied, nor dissatisfied','Rather dissatisfied','Very dissatisfied'],value=[5,4,3,2,1])

df_slice_30=data.iloc[:,219:222]
data['Q30_1'].value_counts()
df_slice_30=df_slice_30.replace(to_replace=['DK','No difficulties',np.nan],value=0)
df_slice_30['Q30_1'] = np.where((df_slice_30['Q30_1'] !=0) , 1, df_slice_30['Q30_1'])
df_slice_30['Q30_2'] = np.where((df_slice_30['Q30_2'] !=0) , 1, df_slice_30['Q30_2'])
df_slice_30['Q30_3'] = np.where((df_slice_30['Q30_3'] !=0) , 1, df_slice_30['Q30_3'])


data['Q21'] = data['Q21'].replace(to_replace =['Very difficult','Difficult','Nor, nor','Easy','Very easy'],value=[0,1,2,3,4]  )

df_slice_25=data.iloc[:,86:116]
#df_slice_25
#df_slice_25=data.iloc[:,78:108]
df_slice_25=df_slice_25.replace(np.nan,0)
cols_to_drop = df_slice_25.columns[df_slice_25.columns.str[-1]== '7']
# drop the selected columns
df_slice_25 = df_slice_25.drop(cols_to_drop, axis=1)

object_data=data.select_dtypes(object)

data['Q33_1']=data['Q33_1'].str.lower()
data['Q33_1']=data['Q33_1'].replace(np.nan,-1)
data['Q33_1'].isnull().sum()

data['Q33_1']=data['Q33_1'].str.lower()
data['Q33_1']=data['Q33_1'].replace(np.nan,'0')
data['Q33_1']=data['Q33_1'].replace('dk','0')
data['Q33_1']=data['Q33_1'].replace('-1','0')

data['Q33_2']=data['Q33_2'].str.lower()
data['Q33_2']=data['Q33_2'].replace(np.nan,'-1')
data['Q33_2']=data['Q33_2'].replace('dk','0')

chis=['restaurant','botanical','city','alley','arch','central','theatre','theater','moldexpo','park','stefan','museum','square','museums','concerts','restaurants','muzeu','chisinau']
vinarii=['kvint','cricova','purcari','cojusna','vartely','mimi','asconi','cetera','milesti','wineries','wine']
churches=['hincu','monasteries','monastery', 'churches','capriana','saharna','tipova','hancu','manastire','orhei']
castele=['tighina','soroca','manuc','mimi','citadel']
landscapes=['vadul','nistru','voda''padurea','domneasca','codrii','pestera','prut','nature','landscapes']
festivals=['international','jazz','festival','wine festival','anniversary','wine day']
agro=['cuisine','butuceni', 'pension','casa','hanul','fata morgana','rural']



data['Q16A']=data['Q16A'].replace(to_replace=[np.nan,'Religion (pilgrimage, missionary travels etc.)','Visit the university','Charity','Sport, fotball game, rugby','To visit Transnistria','DK','Documents','Transit','Health and medical care','Shopping, buying goods with the aim of consumption'],value=0)
data['Q16A']=data['Q16A'].replace(to_replace=['Business or professional activity','Visiting relatives, friends','Leisure: Vacation, rest, recreation','Wine Festival (National Wine Day)','Festivals, celebrations, events','Culture'],value=[1,2,3,4,4,4])
data["Q31"] = data["Q31"].replace(np.nan, -1)
data1=data.iloc[:,0:84]
data_last=data.iloc[:,222:-1]
new_data=pd.concat([data1,df_slice_24A,df_slice_24B,df_slice_25,df_slice,df_slice_28,df_slice_30,data_last,data['Q39']],axis=1)
new_data['Q38']=new_data['Q38'].replace(np.nan,-1)
new_data['Q21']=new_data['Q21'].replace(np.nan,-1)
new_data['Q3']=dataa['Q3A']
cols_to_drop = new_data.columns[new_data.columns.str.contains('Q36')]
# drop the selected columns
new_data = new_data.drop(cols_to_drop, axis=1)
new_data=new_data.drop(['Q34_1','Q34_2','Q34_3','Q35'],axis=1)
new_data=new_data.drop(['Q20_1_other','Q20_3_other','Q20_9_other','Q22_11_other','Q23_6_other'],axis=1)
new_data['Q33_1']=new_data['Q33_1'].replace(to_replace=['0','1','2','3','4','5','6','7'],value=[0,1,2,3,4,5,6,7])
new_data['Q33_1'] = new_data['Q33_1'].apply(lambda x: 0 if isinstance(x, str) else x)
new_data['Q33_2']=new_data['Q33_2'].replace(to_replace=['-1','0','1','2','3','4','5','6','7'],value=[-1,0,1,2,3,4,5,6,7])
new_data['Q33_2'] = new_data['Q33_2'].apply(lambda x: 0 if isinstance(x, str) else x)
new_data['Q33_3']=new_data['Q33_3'].replace(to_replace=['-1','0','1','2','3','4','5','6','7'],value=[-1,0,1,2,3,4,5,6,7])
new_data['Q33_3'] = new_data['Q33_3'].apply(lambda x: 0 if isinstance(x, str) else x)
new_data_1,new_data_2,new_data_3=new_data,new_data,new_data
new_data_1=new_data.drop(['Q33_2','Q33_3'],axis=1)
new_data_2=new_data.drop(['Q33_1','Q33_3'],axis=1)
new_data_3=new_data.drop(['Q33_1','Q33_2'],axis=1)
new_data_1 = new_data_1.rename(columns={'Q33_1': 'target'})
new_data_2 = new_data_2.rename(columns={'Q33_2': 'target'})
new_data_3 = new_data_3.rename(columns={'Q33_3': 'target'})
new_data_1=new_data_1.drop(new_data_1[new_data_1['target'] == -1].index)
new_data_2=new_data_2.drop(new_data_2[new_data_2['target'] == -1].index)
new_data_3=new_data_3.drop(new_data_3[new_data_3['target'] == -1].index)
final_data=pd.concat([new_data_1,new_data_2,new_data_3],axis=0).reset_index(drop=True)
cols_to_drop = final_data.columns[final_data.columns.str.contains('other')]
# drop the selected columns
final_data = final_data.drop(cols_to_drop, axis=1)
final_data['Q16A']=final_data['Q16A'].replace('Other',0)
final_data['Q21']=final_data['Q21'].replace(to_replace=['Don’t know/ Refuze to answer',0],value=-1)
X=final_data.drop('target',axis=1)
y=final_data.target


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_test = X_test[X_train.columns]


def transform_vector(vector):
   if vector[1] == '18-25':
      vector[1] = 5
   elif vector[1] == '26-35':
      vector[1] = 4
   elif vector[1] == '36-45':
      vector[1] = 3
   elif vector[1] == '46-55':
      vector[1] = 2
   elif vector[1] == '56-67':
      vector[1] = 1
   elif vector[1] == '68+':
      vector[1] = 0
   return vector


def choose_row(vector):
   contains_bob = [1, 2, 3, 4, 5, 6]
   contains_bob[0] = final_data['Q2'].isin([enc_nom_1[vector[0]]])
   contains_bob[1] = final_data['Q3'].isin([vector[1]])
   contains_bob[2] = final_data['Q13'].isin([vector[2]])
   contains_bob[3] = final_data['Q16A'].isin([vector[3]])
   contains_bob[4] = final_data['Q39'].isin([enc_nom_39[vector[4]]])
   contains_bob[5] = contains_bob[0].astype(int) + contains_bob[1].astype(int) + contains_bob[2].astype(int) + \
                     contains_bob[3].astype(int) + contains_bob[4].astype(int)
   return contains_bob[5]
import random
def give_response(vector):
   response=choose_row(vector)
   response_maxis=response[response.values==response.max()].index
   response_index=random.choice(response_maxis)
   return final_data['target'].iloc[response_index]
