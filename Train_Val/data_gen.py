import math
import pandas as pd
import numpy as np
import pickle
import os
import torch

####### LOAD THE 3D POSES FOR UPFALL ACTION SAMPLES #######
path='../poses/' #path to 3d poses for actions
SUBJECTS=['Subject1','Subject2','Subject3','Subject4','Subject5','Subject6']
ACTIVITIES=['Activity1','Activity2','Activity3','Activity4','Activity5','Activity6','Activity7','Activity8','Activity9','Activity10','Activity11']
TRIALS=['Trial1','Trial2','Trial3']
CAMERAS='Camera1' #Using camera1 to compare results with other works

#Function for getting label given activity name
#Activity 1-5: fall (1) Activity 6-11: not fall(0)
def get_actid(act_name):
    act2id={'Activity1':1,'Activity2':1,'Activity3':1,'Activity4':1,'Activity5':1,
        'Activity6':0,'Activity7':0,'Activity8':0,'Activity9':0,'Activity10':0,
        'Activity11':0}
    return act2id[act_name]



#Function to map pose seq for one act to an id and label
#input: path to dir containing poses
#output: 2 dictionaries one for pose2id= {'id-1':[pose1,pose2...posen]}, id2label={'id-1':0,'id-2':1...}
def pose2idlabel(poses_path):
    pose2id=dict()
    id2label=dict()
    i=0
    subjects=os.listdir(poses_path)
    for sub in subjects:
        sub_path=poses_path+'/'+sub
        if os.path.isdir(sub_path):
            activities=os.listdir(sub_path)
            for act in activities:
                act_path=sub_path+'/'+act
                if os.path.isdir(act_path):
                    act_label=get_actid(act)
                    trials=os.listdir(act_path)
                    for trial in trials:
                        trial_path=act_path+'/'+trial
                        if os.path.isdir(trial_path):
                            act_poses=os.listdir(trial_path+'/Camera1/mocap/')
                            pose2id['id-'+str(i)]=[trial_path+'/Camera1/mocap/'+pose for pose in act_poses]
                            id2label['id-'+str(i)]=act_label
                            i+=1
    return pose2id,id2label
 

#Get pose dir to id dict, and id to label dict
pose2id,label=pose2idlabel(path)

#Perform train test split
train_split=70
test_split=30
ids=list(label.keys())
idx=int(np.floor(len(label)*0.7))
partition=dict()
partition['train']=ids[:idx]
partition['test']=ids[idx:]

#print("Partition dict:",partition)

        
#Create pytorch dataset

class Poses3d_Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, pose2id):
            'Initialization'
            self.labels = labels
            self.list_IDs = list_IDs
            self.pose2id = pose2id

    #Function to get poses for F frame, given sample id 
    def get_pose_data(self,id):
        poses=self.pose2id[id]
        data_sample=[]
        for pose in poses:
            if pose.endswith('.pkl'):
                pickle_off=open(pose,"rb")
                pkl_pose=pickle.load(pickle_off)
                joints3d=pkl_pose['pred_output_list'][0]['pred_joints_img'] #3d joints for one frame - 49x3
                data_sample.append(joints3d)
        return np.array(data_sample)

    def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

    def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]
            # Load data and get label
            X = torch.from_numpy(self.get_pose_data(ID))
            y = self.labels[ID]

            return X, y 
            

