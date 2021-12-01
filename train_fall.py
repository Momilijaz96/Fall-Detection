import torch
import numpy as np
from data_gen import Poses2d_Dataset,label,partition,pose2id
from FallModel.pf4fall import FallTransformer
from utils.visualize import get_plot

#CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100
inf_threshold=0.5


# Datasets
partition = partition
labels = label
num_frames=200
#print("Pose2od: ",pose2id)
# Generators
training_set = Poses2d_Dataset(partition['train'], labels, pose2id, num_frames)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Poses2d_Dataset(partition['test'], labels, pose2id, num_frames)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

#Define model

fall_model=FallTransformer(num_frame=num_frames, num_joints=17, in_chans=2)
fall_model=fall_model.to(device)


#Define loss and optimizer
criterion=torch.nn.BCELoss()
optimizer=torch.optim.SGD(fall_model.parameters(), lr=0.01, momentum=0.9)

#Loop over epochs
print("Begin Training....")

epoch_loss=[]
epoch_acc=[]
for epoch in range(max_epochs):
    correct=0
    loss=[]
    fall_model.train()
    #Training
    for batch_idx,sample in enumerate(training_generator):
        #Transfer to GPU
        local_batch,local_labels=sample; #local_labels=torch.squeeze(local_labels,-1)
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        
        optimizer.zero_grad()

        #Predict fall/no fall activity
        predict_labels=fall_model(local_batch.float())

        #print("Target is", local_labels.size())
        #print("Pred is", predict_labels.size())        
        #Compute loss
        local_labels=local_labels.view(local_labels.size()[0],1)
        local_labels=local_labels.to(torch.float32)
        prediction_loss=criterion(predict_labels,local_labels)
        
        #Compute gradients
        prediction_loss.backward()

        #Update params
        optimizer.step()

        #per epoch loss
        loss.append(prediction_loss.item())

        #Compute number of correctly predicted
        correct += ((predict_labels>=inf_threshold)==local_labels).sum().item()
    
    num_samples=(batch_idx+1) * params['batch_size']
    train_acc = 100 * correct / num_samples
    print("Epoch: ",epoch," Loss: ",np.round(np.mean(loss),2)," Accuracy:",np.round(train_acc,2)," for ",correct,"/",num_samples)
    epoch_loss.append(np.round(np.mean(loss),2))
    epoch_acc.append(np.round(train_acc,2))

print("TRAINING COMPLETED :)")

#Save visualization
get_plot(epoch_loss,'Training_Loss')
get_plot(epoch_acc,'Training_Accuracy',ylabel='Accuracy(%)')

#Save trained model
torch.save(fall_model,"fall_model.pt")
