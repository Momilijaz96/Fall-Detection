import math
import pandas as pd
import numpy as np
import pickle
import os
import torch
import ast


####### LOAD THE 3D POSES FOR UPFALL ACTION SAMPLES #######
path='poses'#/home/mo926312/Documents/falldet/poses/' #path to 3d poses for actions
SUBJECTS=['Subject1','Subject2','Subject3','Subject4','Subject5','Subject6']
ACTIVITIES=['Activity1','Activity2','Activity3','Activity4','Activity5','Activity6','Activity7','Activity8','Activity9','Activity10','Activity11']
TRIALS=['Trial1','Trial2','Trial3']
CAMERAS='Camera1' #Using camera1 to compare results with other works

#Function for getting label given activity name
#Activity 1-5: fall (1) Activity 6-11: not fall(0)
def get_actid(sample):
    if ('Activity1T' in sample) or ('Activity2T' in sample) or ('Activity3T' in sample) or ('Activity4T' in sample) or ('Activity5T' in sample):
        return 1
    else:
        return 0

#Function to map pose seq for one act to an id and label
#input: path to dir containing poses
#output: 2 dictionaries one for pose2id= {'id-1':'S1A1T1C1.csv','id-2':'....}, id2label={'id-1':0,'id-2':1...}
def pose2idlabel(poses_path):
    pose2id=dict()
    id2label=dict()
    i=0
    subjects=os.listdir(poses_path)
    for sub in subjects:
        sub_path=poses_path+sub+'/'
        if os.path.isdir(sub_path):
            samples=os.listdir(sub_path)
            for sample in samples:
                pose2id['id-'+str(i)]=sample #pose2id['id-1']='S1A1T1C1.csv',...
                id2label['id-'+str(i)]=get_actid(sample) #Get activity label
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

    #Function to get poses for F frames/ one sample, given sample id 
    def get_pose_data(self,id):
        pose=self.pose2id[id] #get path to one action/sample's pose
        data_sample=[]
        if pose.endswith('.csv'):
            df=pd.read_csv(pose)
            for _,row in df.iterrows():
                joints2d = (ast.literal_eval(row['keypoints']))
                joints2d = np.array(joints2d).reshape(17,2) #2d joints for one frame - 17x2
                data_sample.append(joints2d)
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


