#import all the required libraries
import pandas as pd
import os
import datetime as dt

#set the folder config (where all the log files are stored)
fldr='/home/deepak/iMacros/Downloads/Process'

inp=pd.DataFrame()
#loop through all log files
for fl in os.listdir(fldr):
    temp=pd.read_csv(fldr+"/"+fl,sep=",",engine='python')       #read file
    inp=inp.append(temp)    
    
#remove duplicates
out=inp.drop_duplicates(['Date and Time','IP Address'])

#data processing
out['Date and Time']=[dt.datetime.strptime(t,'%Y-%m-%d %H:%M:%S') for t in out['Date and Time']] 
out['user']=out['IP Address']+out['OS'] #create a USER defn.
output=out[['Date and Time','user','Browser','OS','Country','Region','City','URL','Returning Count']]
output=output.sort(['Date and Time'],ascending=False)
output.to_csv(fldr+"/combined.csv",index=True)








    


    