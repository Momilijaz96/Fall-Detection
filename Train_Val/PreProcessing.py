import numpy as np
import pandas as pd
import json
import os

#Function to remove scores from keypoints array
def remove_kpscores(keypoints):
	[keypoints.remove(s) for s in keypoints[2::3]]
	return keypoints

#Process pased obj to ake ti compatible for making df
def obj_format(obj):
	obj=obj.to_dict('index')
	key=list(obj.keys())[0]
	return obj[key]

#Function to return list of elements if path is a dir
def ret_dir(path):
	if os.path.isdir(path):
		return os.listdir(path)
	else:
		return None
		
Subjects=['Subject1','Subject2','Subject3','Subject4','Subject5','Subject5','Subject6','Subject7','Subject8','Subject9','Subject10','Subject11','Subject12','Subject13','Subject14','Subject15','Subject16','Subject17']
Activities=['Activity1','Activity2','Activity3','Activity4','Activity5','Activity6','Activity7','Activity8','Activity9','Activity10','Activity11']
Trials=['Trial1','Trial2','Trial3']
Cameras=['Camera1','Camera2']

#Iterate over all actions for one subject
parent_dir='/home/mo926312/Documents/falldet/'
poses_path=parent_dir+'poses/'
output_dir=parent_dir+'PreProcess_poses/'

subjects=os.listdir(poses_path)
temp_path=''
for sub in Subjects[:5]:
	activities=ret_dir(poses_path+sub+'/')
	temp_path=poses_path+sub+'/'
	if activities is not None:
		for act in activities:
			trials=ret_dir(poses_path+sub+'/'+act+'/')
			temp_path=poses_path+sub+'/'+act+'/'
			if trials is not None:
				for trial in trials:
					cams=ret_dir(poses_path+sub+'/'+act+'/'+trial+'/')
					temp_path=poses_path+sub+'/'+act+'/'+trial+'/'
					if cams is not None:
						for cam in cams:
							cam_path=poses_path+sub+'/'+act+'/'+trial+'/'+cam+'/'
							temp_path=poses_path+sub+'/'+act+'/'+trial+'/'+cam+'/'
							if os.path.isdir(cam_path):
								activity_poses=cam_path+'alphapose-results.json'
								#Perform Pre-processing of poses
								#1. Remove extra info columns from apose results.
								#2. Keep one pose per frame with highest score.
								print("PATH: ",activity_poses)
								with open(activity_poses) as f:
									action_sample=json.load(f)
									action_sample_new=[]
									for frame in action_sample:
										frame_obj={'image_id':frame['image_id'],
										'bbox':frame['box'],
										'score':frame['score'],
										'keypoints':remove_kpscores(frame['keypoints'])}
										action_sample_new.append(frame_obj)

									action_df=pd.DataFrame(action_sample_new) #single df with desired cols and keypoints w/o scores
									#print("Action DF: ",action_df['keypoints'][0])

									#Remove Multiple bboxes from one frame
									temp_df=action_df[['image_id','score']].groupby('image_id').max()
									
									unique_action_list=[]
									for image_id,sample in temp_df.iterrows():
										obj=(action_df[(action_df['image_id']==image_id)&(action_df['score']==sample['score'])])
										unique_action_list.append(obj_format(obj))

									unique_action_df=pd.DataFrame(unique_action_list) #df with one pose per frame, with highest score
									#print("Final df:",unique_action_df['keypoints'][0])
									if not os.path.exists(output_dir+sub+'/'):
										os.makedirs(output_dir+sub+'/')	
									file_path=output_dir+sub+'/'+sub+act+trial+cam+'.csv'
									unique_action_df.to_csv(file_path,index=False)
									print("Processed: ",file_path)
									#This data object contains alposes, one per frame for detected persons, in following format
									# image_id - keypoints (17 xy joints) - bbox - score
print("Temp Path: ",temp_path)

