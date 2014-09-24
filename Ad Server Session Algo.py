import pandas as pd
from pandas import Series, DataFrame
import datetime as dt
import string
import numpy as np
import sys

#pre-process
input=pd.read_csv('yoda_exp_hr_user_askfm.csv',sep=",",encoding='utf-16')        #read file
input['request_time']=input.request_time.str.split('.').str[0];                 #trucnate the microsec
input['request_time']=[dt.datetime.strptime(t,'%Y-%m-%d %H:%M:%S') for t in input.request_time]             #convert request_time from string to datetime format for time based operations
input=input.sort(['normalized_user_id','request_time'],ascending=True)     #sort by user_id & time asc
input['key']=input['normalized_user_id']+input['request_time'].map(str)   #create eky combine for user & request
input['key']=input.key.str.replace(' ','');

#iterate over data
sys.stdout = open('session.csv', 'w')
print 'key delta ucnt scnt'
x=input.request_time[0]; y=input.normalized_user_id[0];ucnt=0;scnt=1;
for index,row in input.iterrows():
    delta=row['request_time']-x
    dlt=delta.seconds
    if row['normalized_user_id'] == y and delta.seconds > 300:  #change minutes defintion for session here
        scnt=scnt+1
        dlt=0
    elif row['normalized_user_id'] != y:
        scnt=1 ;ucnt = ucnt+1; dlt=0;
    else:
        aa=1               
    x=row['request_time']
    y=row['normalized_user_id']
    print row['key'],dlt,ucnt,scnt


input1=pd.read_csv('session.csv',sep=" ",header=0)     #read the file created
sys.stdout = open('output.txt', 'w') #set output content
ds=pd.merge(left=input,right=input1,on='key',how='left')  #merge the two records - now ds is THE thing to play with

#average sesion length
d4=ds['delta'].groupby(ds['scnt'].map(str)+'-'+ds['ucnt'].map(str)).sum().reset_index();
d4['delta'].where(d4['delta'] != 0).mean()  #average session length (excluding 1 request sessions i.e at least 2 req in 5 min time frame) = 244.9s = 4mins
#average requests per session
d4=ds['delta'].groupby(ds['scnt'].map(str)+'-'+ds['ucnt'].map(str)).count().reset_index();
d4['delta'].mean() #average requests per session = 2.99

#chk
d4['delta'].where(d4['delta']!=1).mean() #avg req. per session (min 2 req defines a sess) = 4.72

#average time gap including 0s gaps
req=ds['key'].groupby(ds['delta']).count().reset_index()
req.to_csv('deltainc0.csv',index=False,header=False)
a=ds['delta'].where(ds['delta']!=0).groupby(ds['ucnt']).mean().reset_index(); #get average timegap between request by user
b=a['ucnt'].groupby(np.round(a['delta'],0)).count().reset_index()
a.to_csv('deltareq.csv',index=False)

#(Q1)FINALIZED CHART FORMULAE
dsn0=ds[ds.delta!=0]  #remove 0 timegap requests
tgap=dsn0['delta'].groupby(dsn0['ucnt']).mean().reset_index(); #group by user and get average timegap b/n requests
print "Average Time Gap B/N Requests per User (in a 5min session)",tgap.delta.mean()
tgap1=tgap['ucnt'].groupby(np.round(tgap['delta'],0)) .count().reset_index() #get the distribution of users by average time gap
tgap1.to_csv('tgap.csv',index=False,header=False)  #export to csv
print "Total users with atleast one session",tgap1.ucnt.sum()


#(Q2) AVERAGE SESSION LENGTH / USER (in sec)
ds2=dsn0.groupby(['ucnt','scnt']).sum().reset_index() #use sessions (i.e no single requests) and find total session lenght
print 'Total Sessions',ds2.scnt.count()
dslen=ds2.delta.groupby(ds2['ucnt']).mean().reset_index() #average session len by user 
print "Average(Mean) session length(s) per user",dslen.delta.mean()
print "Average(Median) session length(s) per user",dslen.delta.median()
dslen=dslen['ucnt'].groupby(np.round(dslen['delta'],0)).count().reset_index() #get histogram /distribution
dslen.to_csv('slen.csv',index=False,header=False)  #export to csv

#(Q4) AVERAGE SESSION COUNT / USER (in sec)
ds2=dsn0.groupby(['ucnt','scnt']).sum().reset_index() #use sessions (i.e no single requests) and find total session lenght
noses=ds2.scnt.groupby(ds2['ucnt']).count().reset_index() #total session by user 
print "Average(Mean) # of sessions per user",noses.scnt.mean()
noses=noses['ucnt'].groupby(noses['scnt']).count().reset_index() #get histogram /distribution
noses.to_csv('noses.csv',index=False,header=False)  #export to csv

#(Q3)Average requests/session per user
ds3=dsn0.groupby(['ucnt','scnt']).count().reset_index()
ds3.request_time = ds3.request_time+ 1
dsreq=ds3['request_time'].groupby(ds3['ucnt']).mean().reset_index()
dsreq=dsreq['ucnt'].groupby(np.round(dsreq['request_time'],0)).count().reset_index()
dsreq.to_csv('dsreq.csv',index=False,header=False)  #export to csv


 
