import os
import numpy as np
import pandas as pd

#Path to poses directory
path='../../PreProcess_poses/'

#frames count in each action sample
fc=[]
sample=[]
subjects=os.listdir(path)
ACTIVITIES=['Activity1','Activity2','Activity3','Activity4','Activity5','Activity6','Activity7','Activity8','Activity9','Activity10','Activity11']
TRIALS=['Trial1','Trial2','Trial3']

for sub in subjects:
	sub_path=path+sub+'/'
	poses=os.listdir(sub_path)
	for pose in poses:
		df=pd.read_csv(sub_path+pose)
		fc.append(len(df))
		sample.append(pose)


#Print min frames
print("Min Frames are ", np.min(fc), " for sample ",sample[np.argmin(fc)])
print("Mean frames are: ",np.mean(fc))
print("Max frames are: ",np.max(fc))
print("Frame size variation: ",np.var(fc))
print("Poses with less than 200 frames: ",np.sum(np.array(fc)>200))
print("Total poses:",len(fc))
#print(fc)
