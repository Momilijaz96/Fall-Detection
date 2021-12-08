import numpy as np
from FallModel.pf4fall import FallTransformer
import torch
from sklearn.metrics import precision_recall_fscore_support
from PreProcessing import preprocess_pose
import time
#CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
inf_threshold=0.5
#Input Sample Pose File
pose_file='/home/mo926312/Documents/falldet/YTubeFalls/jump1poses/alphapose-results.json'

#Prprocess pose
Fall_Frames=200
start=time.time()
df=preprocess_pose(pose_file)
poses=list(df['keypoints']) #fx17x2
num_frames = len(poses)
if num_frames<Fall_Frames:
    diff=Fall_Frames-num_frames
    last_pose=poses[-1]
    append_list=[last_pose]*diff
    poses=poses+append_list
else:
    poses=poses[:Fall_Frames]

poses=np.array(poses).reshape(Fall_Frames,17,2)
poses=torch.unsqueeze(torch.from_numpy(poses),0)
posep_time=time.time()-start
#Action labels
labels={1:'Fall',0:'Not Fall'}

#Load pretrained model and criterion
model_path='/home/mo926312/Documents/falldet/Fall-Detection/modelZoo/fall_model.pt'
fall_model=torch.load(model_path)
fall_model=fall_model.to(device)

#Loop over validation split
fall_model.eval()
with torch.no_grad():
    #Transfer to GPU
    poses = poses.to(device)

    start=time.time()
    #Predict fall/no fall activity
    predict_prob=fall_model(poses.float())
    predict_label=predict_prob>=inf_threshold 
    model_inf_time=time.time()-start
print("THE ACTION IS A ",labels[predict_label.item()])
print("Prediction Probability: ",predict_prob.item())
print("Pose Preprocessing Time: ",posep_time)
print("Model Inference Time: ",model_inf_time)




