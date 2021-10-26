from sklearn.ensemble import RandomForestClassifier
from data_gen import Poses3d_Dataset,label,partition,pose2id
import torch

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

# Datasets
partition = partition
labels = label


# Parameters
params_train = {'batch_size': len(partition['train']),
          'shuffle': True,
          'num_workers': 3}
max_epochs = 100
inf_threshold=0.5

# Parameters
params_val = {'batch_size': len(partition['test']),
          'shuffle': True,
          'num_workers': 3}
max_epochs = 100
inf_threshold=0.5

# Generators
training_set = Poses3d_Dataset(partition['train'], labels, pose2id)
training_generator = torch.utils.data.DataLoader(training_set, **params_train)

validation_set = Poses3d_Dataset(partition['test'], labels,pose2id)
validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)


for batch_idx,sample in enumerate(training_generator):
    #Get all trianing data
    X_train, y_train = sample
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    print("batch no: ",batch_idx)



