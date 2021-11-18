import numpy as np
from data_gen import Poses3d_Dataset,label,partition,pose2id
from model import FallTransformer
import torch

#CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Datasets
partition = partition
labels = label

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100
inf_threshold=0.5


#Generator
validation_set = Poses3d_Dataset(partition['test'], labels,pose2id)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

#Load pretrained model and criterion
frames=79
model_path='fall_model.pt'
fall_model=torch.load(model_path)
fall_model=fall_model.to(device)
criterion=torch.nn.BCELoss()

#Loop over validation split
fall_model.eval()
loss=[]
correct=0
for batch_idx,sample in enumerate(validation_generator):
    #Transfer to GPU
    local_batch, local_labels = sample
    local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    
    #Predict fall/no fall activity
    predict_labels=fall_model(local_batch)
    
    #Compute loss
    local_labels=local_labels.view(local_labels.size()[0],1,1)
    local_labels=local_labels.to(torch.float32)
    prediction_loss=criterion(predict_labels,local_labels)
    

    #per epoch loss
    loss.append(prediction_loss.item())

    #Compute number of correctly predicted
    correct += ((predict_labels>=inf_threshold)==local_labels).sum().item()
    
num_samples=(batch_idx+1) * params['batch_size']
train_acc = 100. * correct / num_samples
print(" Loss: ",np.round(np.mean(loss),2)," Accuracy:",np.round(train_acc,2)," for ",correct,"/",num_samples)






