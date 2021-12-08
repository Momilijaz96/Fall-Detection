import numpy as np
from data_gen_fall import Poses2d_Dataset,label,partition,pose2id
from FallModel.pf4fall import FallTransformer
import torch
from sklearn.metrics import precision_recall_fscore_support

#CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Datasets
partition = partition
labels = label
num_frames=200
# Parameters
params = {'batch_size':16,
        'shuffle': True,
        'num_workers': 6}
max_epochs = 100
inf_threshold=0.5


#Generator
validation_set = Poses2d_Dataset(partition['test'], labels,pose2id,num_frames)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

#Load pretrained model and criterion

model_path='/home/mo926312/Documents/falldet/Fall-Detection/modelZoo/fall_model.pt'
fall_model=torch.load(model_path)
fall_model=fall_model.to(device)
criterion=torch.nn.BCELoss()

#Loop over validation split
fall_model.eval()
loss=[]
correct_true=0
pred_true=0
target_true=0
all_correct=0
for batch_idx,sample in enumerate(validation_generator):
    #Transfer to GPU
    local_batch, local_labels = sample
    local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    #Predict fall/no fall activity
    predict_probs=fall_model(local_batch.float())
    predict_labels=predict_probs>=inf_threshold

    #Compute loss
    local_labels=local_labels.view(local_labels.size()[0],1)
    local_labels=local_labels.to(torch.float32)
    prediction_loss=criterion(predict_probs,local_labels)


    #per epoch loss
    loss.append(prediction_loss.item())

    #Compute number of correctly predicted
    all_correct+= torch.sum(predict_labels==local_labels).item()
    target_true += torch.sum(local_labels == 1).float()
    pred_true += torch.sum(predict_labels == 1).float()
    correct_true += torch.sum((predict_labels == local_labels) & (local_labels==1)).item()


num_samples=(batch_idx+1) * params['batch_size']
test_acc = 100. * all_correct / num_samples
print(" Loss: ",np.round(np.mean(loss),2)," Accuracy:",np.round(test_acc,2)," for ",all_correct,"/",num_samples)
precision=correct_true / pred_true
print("Precesion : ",precision)
recall=correct_true / target_true
print("Recall : ",recall)
print("F1 score : ",2 * precision * recall / (precision + recall))







