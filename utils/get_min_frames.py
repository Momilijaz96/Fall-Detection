import os
import numpy as np

#Path to poses directory
path='../poses/'

#frames count in each action sample
fc=[]


subjects=os.listdir(path)
#Get min frames
for sub in subjects:
    sub_path=path+'/'+sub
    if os.path.isdir(sub_path):
        activities=os.listdir(sub_path)
        for act in activities:
            act_path=sub_path+'/'+act
            if os.path.isdir(act_path):
                trials=os.listdir(act_path)
                for trial in trials:
                    trial_path=act_path+'/'+trial
                    if os.path.isdir(trial_path):
                        act_poses=os.listdir(trial_path+'/Camera1/mocap/')
                        fc.append(len(act_poses))
#Print min frames
print("Min Frames are ", np.min(fc), " for sample id-",np.argmin(fc))
print(fc)                            
