import numpy as np
from RecogModel.recog_model import RecogTransformer
import torch
from sklearn.metrics import precision_recall_fscore_support
from PreProcessing import preprocess_pose
import time

#CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#Input Sample Pose File
pose_file='/home/mo926312/Documents/falldet/YTubeFalls/walk2poses/alphapose-results.json'

#Prprocess pose
start=time.time()
df=preprocess_pose(pose_file)
poses=list(df['keypoints']) #Fx17x2
num_frames = len(poses)
if num_frames<250:
    diff=250-num_frames
    last_pose=poses[-1]
    append_list=[last_pose]*diff
    poses=poses+append_list
else:
    poses=poses[:250]

poses=np.array(poses).reshape(250,17,2)
poses=torch.unsqueeze(torch.from_numpy(poses),0)
print("Poses shape: ",poses.shape)
posep_time=time.time()-start
#Action labels
labels={0:'Falling forward using hands',
1:'Falling forward using knees',
2:'Falling backwards',
3:'Falling sideward',
4:'Falling sitting in empty chair',
5:'Walking',
6:'Standing',
7:'Sitting',
8:'Picking up an object	',
9:'Jumping',
10:'Laying'}

#Load pretrained model and criterion
model_path='/home/mo926312/Documents/falldet/Fall-Detection/modelZoo/recog_model.pt'
recog_model=torch.load(model_path)
recog_model=recog_model.to(device)

#Loop over validation split
recog_model.eval()
with torch.no_grad():
    #Transfer to GPU
    poses = poses.to(device)
    start=time.time()
    #Predict fall/no fall activity
    predict_probs=recog_model(poses.float())
    predict_label=torch.argmax(predict_probs,dim=1)
    model_inf_time=time.time()-start
print("THE ACTION IS A ",labels[predict_label.item()])

print("Pose Preprocessing Time: ",posep_time)
print("Model Inference Time: ",model_inf_time)




