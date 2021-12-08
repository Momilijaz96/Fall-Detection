import numpy as np
from data_gen_recog import Poses2d_Dataset,label,partition,pose2id
from RecogModel.recog_model import RecogTransformer
import torch
from sklearn.metrics import classification_report
from sklearn import metrics
import seaborn as sns
#CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Datasets
partition = partition
labels = label
num_frames=250
# Parameters
params = {'batch_size':16,
        'shuffle': True,
        'num_workers': 6}
max_epochs = 100


#Generator
validation_set = Poses2d_Dataset(partition['test'], labels,pose2id,num_frames)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

#Load pretrained model and criterion

model_path='/home/mo926312/Documents/falldet/Fall-Detection/modelZoo/recog_model.pt'
model=torch.load(model_path)
model=model.to(device)
criterion=torch.nn.CrossEntropyLoss()

#Loop over validation split
model.eval()
loss=[]
y_true=[]
y_pred=[]
all_correct=0
for batch_idx,sample in enumerate(validation_generator):
    #Transfer to GPU
    local_batch, local_labels = sample
    local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    #Predict fall/no fall activity
    predict_probs=model(local_batch.float())

    #Compute loss
    prediction_loss=criterion(predict_probs,local_labels)

    #per epoch loss
    loss.append(prediction_loss.item())

    #Compute number of correctly predicted
    predict_labels = torch.argmax(predict_probs,dim=1)
    y_true+=(list(local_labels.cpu().detach().numpy()))
    y_pred+=(list(predict_labels.cpu().detach().numpy()))
    all_correct+= torch.sum(predict_labels==local_labels).item()


num_samples=(batch_idx+1) * params['batch_size']
test_acc = 100. * all_correct / num_samples
print(" Loss: ",np.round(np.mean(loss),2)," Accuracy:",np.round(test_acc,2)," for ",all_correct,"/",num_samples)

# Print the confusion matrix
print(metrics.confusion_matrix(y_true, y_pred))

# Print the precision and recall, among other metrics
cf_matrix=metrics.confusion_matrix(y_true, y_pred)
print(metrics.classification_report(y_true, y_pred, digits=3))

#Visualize confusion matrix

cf_plot=sns.heatmap(cf_matrix, annot=True)
fig = cf_plot.get_figure()
fig.savefig('cfmatrix.png') 

