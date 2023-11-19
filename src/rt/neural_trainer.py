import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
  

# Classification Model used to determine hit or miss
class HitClassifier(nn.Module):
    def __init__(self, input_size=5, output_size=1):
        super(HitClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 120),
            nn.ReLU(),
            nn.Linear(120, 80),
            nn.ReLU(),
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
    def load_model1_weights(self, path):

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded_state_dict = torch.load(path, map_location=device)
        except Exception as e:
            print(f"An error occurred while loading the weights: {e}")
            return
        
        new_state_dict = {}
        
        # Convert keys to format ingestible for model
        for key, val in loaded_state_dict.items():
            new_key = f"model.{key}"
            new_state_dict[new_key] = val
            
        try:
            self.load_state_dict(new_state_dict)
            self.eval()

        except Exception as e:
            print("Error while loading weights:", e)

        

    def predict_hit_or_miss(self, ray_origin, ray_direction):

        ray_data = ray_origin + ray_direction

        # Convert this data to a PyTorch tensor
        tensor_input = torch.tensor(ray_data)

        # Pass this tensor to the forward function
        output = self.forward(tensor_input)

        # Convert the tensor to numpy array after detaching it
        numpy_array = output.detach().numpy()

        # Convert numpy array to python list
        output_list = numpy_array.tolist()

        return output_list
    

class Distance_Az_El_Model(nn.Module):
    def __init__(self, input_size=5, output_size=3):
        super(Distance_Az_El_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 240),
            nn.ReLU(),
            nn.Linear(240, 180),
            nn.ReLU(),
            nn.Linear(180, 120),
            nn.ReLU(),
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.model(x)
    
    def load_model2_weights(self, path):

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded_state_dict = torch.load(path, map_location=device)
        except Exception as e:
            print(f"An error occurred while loading the weights: {e}")
            return
        
        new_state_dict = {}

        # Convert keys to format ingestible for model
        for key, val in loaded_state_dict.items():
            new_key = f"model.{key}"
            new_state_dict[new_key] = val

        try:
            self.load_state_dict(new_state_dict)
            self.eval()
        except Exception as e:
            print("Error while loading weights:", e)
        
    def predict_dist_az_el(self, ray_origin, ray_direction):

        ray_data = ray_origin + ray_direction

        # Convert this data to a PyTorch tensor
        tensor_input = torch.tensor(ray_data)
        
        # Pass this tensor to the forward function
        output = self.forward(tensor_input)

        # Convert the tensor to numpy array after detaching it
        numpy_array = output.detach().numpy()

        # Convert numpy array to python list
        output_list = numpy_array.tolist()
        
        return output_list

def train_model_1(model_path):

    # Parse data from training file
    data = []
    labels = []
    with open("hit_or_miss_file.txt", "r") as file:
        for line in file:
            row = line.strip().split('*')
            newData = []
            label = []
            hit_or_miss = 0
            for idx, item in enumerate(row):
                if(idx == 0 or idx == 1):
                    newData.extend(item.split(','))
                elif (idx == 2):
                    if float(item) != 0:
                        hit_or_miss = 1

            for i in range(len(newData)):
                newData[i] = float(newData[i])

            data.append(newData)
            labels.append(hit_or_miss)

    data = np.array(data)
    labels = np.array(labels)
    X = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

    # Create instance of model
    model1 = nn.Sequential(
            nn.Linear(5, 120),
            nn.ReLU(),
            nn.Linear(120, 80),
            nn.ReLU(),
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model1.parameters(), lr=0.0001)

    n_epochs = 1000   # number of epochs to run
    batch_size = 512  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
    patience = 3

    best_val_loss = float('inf')
    counter = 0  # Counter to track consecutive epochs without improvement

    for epoch in range(n_epochs):
        model1.train()
        for start in range(0, len(X_train), batch_size):
            end = start + batch_size
            X_batch = X_train[start:end].to(device)
            y_batch = y_train[start:end].to(device)
            # Forward pass
            y_pred = model1(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()
       
        model1.eval()
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        total_acc = 0.0
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
          for X_batch_val, y_batch_val in val_loader:
              X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)

              # forward pass
              y_pred = model1(X_batch_val)

              # calculate loss
              loss = loss_fn(y_pred, y_batch_val)
              total_loss += loss.item() * X_batch_val.size(0)

              # calculate accuracy
              acc = (y_pred.round() == y_batch_val).float().mean()
              total_acc += acc.item() * X_batch_val.size(0)
              total_samples += X_batch_val.size(0)
        avg_val_acc = total_acc / total_samples
        avg_val_loss = total_loss / total_samples

         # Implement early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0  # Reset the counter
        else:
            counter += 1
            if counter >= patience or avg_val_acc == 1:
                print(f"Early stopping at epoch {epoch} as validation loss did not improve for {patience} consecutive epochs or validation accuracy is at 100%.")
                break

        print(f"Epoch {epoch}, Validation accuracy: {avg_val_acc:.4f}, Validation loss: {avg_val_loss:.4f}")

    # Save model weights
    folder_name = "model_weights"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    db_model_weights = folder_name + "/" + model_path + "_weights.pth"

    torch.save(model1.state_dict(), db_model_weights)


def train_model_2(model_path):

    # Parse data from training file
    data = []
    labels = []
    with open("hit_file.txt", "r") as file:
        for line in file:
            row = line.strip().split('*')
            newData = []
            label = []

            for idx, item in enumerate(row):
                if(idx == 0 or idx == 1):
                    newData.extend(item.split(','))
                elif (idx == 2):
                    label.append(item.strip())
                else:
                    label.extend(item.split(','))

            for i in range(len(newData)):
                newData[i] = float(newData[i])
            for i in range(len(label)):
                label[i] = float(label[i])


            data.append(newData)
            labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    # Create DataLoader instances
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create instance of model
    model2 = nn.Sequential(
                nn.Linear(5, 240),
                nn.ReLU(),
                nn.Linear(240, 180),
                nn.ReLU(),
                nn.Linear(180, 120),
                nn.ReLU(),
                nn.Linear(120, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )


    # Step 2: Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)

    # Step 3: Train the model
    num_epochs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2.to(device)

    for n in range(num_epochs):
        model2.train()
        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):

            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model2(inputs)

            # Compute loss
            loss = loss_fn(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

        # Print loss after every epoch
        print(f"Epoch {n+1}/{num_epochs}, Loss: {loss.item()}")
        # Timer end for training loop

        model2.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No gradient computation during validation
            val_loss = 0.0
            correct_predictions = 0

            for i, data in enumerate(val_loader, 0):
                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float().to(device), targets.float().to(device)

                # Perform forward pass
                outputs = model2(inputs)

                # Compute loss
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                # Compute predictions (0 or 1 based on threshold)

            accuracy = correct_predictions / len(val_dataset)
            print(f'Epoch: {n}, Validation Loss: {val_loss/len(val_loader)}')
    
    # Evaluate model on test data
    model2.eval()
    test_loss = 0.0
    correct_test_predictions = 0


    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            # Perform forward pass
            outputs = model2(inputs)

            # Compute loss
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()

        print(f'Test Loss: {test_loss/len(test_loader)}')

    # Save model weights

    folder_name = "model_weights"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    #There is a null terminator on the end of model_path that I need to trim off if I cant find the file on the other side
    db_model_weights = folder_name + "/" + model_path + "hits_weights.pth"
    torch.save(model2.state_dict(), db_model_weights)


