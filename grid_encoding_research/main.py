## NOTE: THE COLAB FOR THIS CAN BE FOUND HERE: https://colab.research.google.com/drive/1WtY_IdgLojdbwzfachHtpOIim7n1XPbg?usp=sharing

## Convert the data into a,e,a,e -> d a e

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim

# 2. Construct Your Neural Network
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch

def read_input_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = [line.strip().split('*') for line in lines]
    points = np.array([[float(coord) for coord in point.split(',')] for point, *_ in data])
    rest_of_data = [rest for _, *rest in data]
    return points, rest_of_data

def cartesian_to_spherical(x, y, z, sphere_center):
    x -= sphere_center[0]
    y -= sphere_center[1]
    z -= sphere_center[2]
    r = math.sqrt(x**2 + y**2 + z**2)
    azimuth = math.degrees(math.atan2(y, x))  # Convert radians to degrees
    elevation = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))  # Convert radians to degrees

    return r, azimuth, elevation

def fit_sphere(points):
    A = []
    B = []
    for i in range(1, len(points)):
        A.append([
            2 * (points[i][0] - points[0][0]),
            2 * (points[i][1] - points[0][1]),
            2 * (points[i][2] - points[0][2])
        ])
        B.append(
            (points[i][0] ** 2 - points[0][0] ** 2) +
            (points[i][1] ** 2 - points[0][1] ** 2) +
            (points[i][2] ** 2 - points[0][2] ** 2)
        )
    A = np.array(A)
    B = np.array(B)
    center = np.linalg.lstsq(A, B, rcond=None)[0]
    return center

def save_output_file(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write('*'.join(map(str, line)) + '\n')

# ... (previous functions remain the same)

def main():
    input_filename = 'hit_or_miss_file.txt'
    output_filename = 'output.txt'

    # Read input file
    points, rest_of_data = read_input_file(input_filename)

    # Sample every 10000th point for sphere fitting
    sampled_points = points[::10000]

    # Fit sphere to find the center using the sampled points
    center = fit_sphere(sampled_points) ## THIS IS AN ESTIMATION
    print(center)
    # Convert Cartesian coordinates to Azimuth and Elevation
    ae_data = []
    for point, rest in zip(points, rest_of_data):
        _, azimuth, elevation = cartesian_to_spherical(point[0], point[1], point[2], center)
        ae_string = f"{azimuth},{elevation}"
        ae_data.append([ae_string, *rest])

    # Save to output file
    save_output_file(output_filename, ae_data)
    print(f'Data has been converted and saved to {output_filename}')

if __name__ == '__main__':
    main()

def bilinear_interpolate(grid, normCoords):
    # Calculate the size of the grid
    W, H = grid.size(1) - 1, grid.size(0) - 1

    # Calculate the indices of the top-left corners
    top_left_x = (normCoords[:, 0] * W).long()
    top_left_y = (normCoords[:, 1] * H).long()

    # Calculate the fractional part of the indices
    x_fract = (normCoords[:, 0] * W) % 1
    y_fract = (normCoords[:, 1] * H) % 1

    # Calculate the indices of the corners
    bottom_right_x = torch.min(top_left_x + 1, torch.full_like(top_left_x, W))
    bottom_right_y = torch.min(top_left_y + 1, torch.full_like(top_left_y, H))

    # Gather the vectors at each corner for the entire batch
    tl_vectors = grid[top_left_y, top_left_x]    # Top-left corner vectors
    tr_vectors = grid[top_left_y, bottom_right_x]  # Top-right corner vectors
    bl_vectors = grid[bottom_right_y, top_left_x]  # Bottom-left corner vectors
    br_vectors = grid[bottom_right_y, bottom_right_x]  # Bottom-right corner vectors

    # Calculate the interpolated vectors
    top_interp = (1 - x_fract).unsqueeze(1) * tl_vectors + x_fract.unsqueeze(1) * tr_vectors
    bottom_interp = (1 - x_fract).unsqueeze(1) * bl_vectors + x_fract.unsqueeze(1) * br_vectors

    # Final interpolation between top and bottom
    interpolated_vectors = (1 - y_fract).unsqueeze(1) * top_interp + y_fract.unsqueeze(1) * bottom_interp

    return interpolated_vectors


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 1. Initialize Feature Vectors Grid
        # Grid is of size NxN and each feature vector has a length of 3. Initilized as random inputs from -10 to 10
        self.pos_grid = nn.Parameter(torch.randn(256, 256, 3, requires_grad=True)) * 10  # Initializing with random values
        self.dir_grid = nn.Parameter(torch.randn(256, 256, 3, requires_grad=True)) * 10
        self.fc1 = nn.Linear(6, 64)  # Assuming input feature size is 6
        self.fc2 = nn.Linear(64, 64)  # Second hidden layer with 64 nodes
        self.fc3 = nn.Linear(64, 1)   # Output layer

    def forward(self, pos, dir):
        pos_features = bilinear_interpolate(self.pos_grid, pos).detach()  # Get features from position grid
        dir_features = bilinear_interpolate(self.dir_grid, dir).detach()  # Get features from direction grid

        # Concatenate the features from both grids with the original input features
        x = torch.cat((pos_features, dir_features), dim=1)
        x = F.leaky_relu(self.fc1(x))  # Leaky ReLU after first hidden layer
        x = F.leaky_relu(self.fc2(x))  # Leaky ReLU after second hidden layer
        x = torch.sigmoid(self.fc3(x))  # Sigmoid after output layer to ensure a probability between 0 and 1
        return x
    
def normalize_azimuth(azimuth_degree):
  a = (azimuth_degree + 180) / 360
  return a

def normalize_elevation(elevation_degree, elevation_min=-90, elevation_max=90):
  e = (elevation_degree - elevation_min) / (elevation_max - elevation_min)
  return e

#create data vectors
pos = []
dir = []
labels = []
counter = 0
with open("output.txt", "r") as file:
    for line in file:
        row = line.strip().split('*')
        newPos = []
        newDir = []
        label = []
        hit_or_miss = 0
        for idx, item in enumerate(row):
            if(idx == 0):
                newPos.extend(item.split(','))
            elif(idx == 1):
                newDir.extend(item.split(','))
            elif (idx == 2):
              if float(item) != 0:
                hit_or_miss = 1

        for i in range(len(newPos)):
            newPos[i] = float(newPos[i])
            newDir[i] = float(newDir[i])
        pos.append(newPos)
        dir.append(newDir)
        labels.append(hit_or_miss)



pos = torch.tensor(pos)
print(pos[:5])
normPos = torch.stack([normalize_azimuth(pos[:, 0]),
                                      normalize_elevation(pos[:, 1])], dim=1)
print(normPos[:5])

dir = torch.tensor(dir)
print(dir[:5])
normDir = torch.stack([normalize_azimuth(dir[:, 0]),
                                      normalize_elevation(dir[:, 1])], dim=1)
print(normDir[:5])

labelsTens = torch.tensor(labels)
print(labelsTens[:10])
zero_count = torch.sum(labelsTens == 0).item()
total_elements = labelsTens.numel()
zero_percentage = (zero_count / total_elements) * 100

print(f"Percentage of zeros in the tensor: {zero_percentage}%")


class CustomDataset(Dataset):
    def __init__(self, pos_data, dir_data, labels):
        self.pos_data = pos_data  # Raw position data (azimuth, elevation)
        self.dir_data = dir_data  # Raw direction data (azimuth, elevation)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pos = self.pos_data[idx]
        dir = self.dir_data[idx]
        label = self.labels[idx]
        return pos, dir, label

# Create dataset instance
dataset = CustomDataset(normPos, normDir, labelsTens)



# Define the size of your validation set
val_size = int(len(dataset) * 0.2)  # let's reserve 20% of the dataset for validation
train_size = len(dataset) - val_size  # the rest for training

# Randomly split the dataset into training and validation datasets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create a DataLoader for the training set
train_dataloader = DataLoader(train_dataset, batch_size=pow(2, 11), shuffle=True)

# Create a DataLoader for the validation set
val_dataloader = DataLoader(val_dataset, batch_size=pow(2, 11))

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004) #untested with other values
criterion = nn.BCELoss()

num_epochs = 50



for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for pos_batch, dir_batch, label_batch in train_dataloader:
        # Assuming pos_batch and dir_batch are already normalized
        # Interpolate the features from the grid for the entire batch
        optimizer.zero_grad()

        # Forward pass through the model
        output = model(pos_batch, dir_batch)

        # If label_batch is not already a float tensor, you should convert it
        if label_batch.dtype is not torch.float32:
            label_batch = label_batch.float()

        # Compute loss
        loss = criterion(output.squeeze(), label_batch)
        running_loss += loss.item()

        # Backprop and optimize

        loss.backward()

        optimizer.step()  # Update the weights based on gradient




        # Convert predictions to predicted class labels
        predicted_labels = output.round().squeeze()

        # Calculate the number of correct predictions
        correct_predictions += (predicted_labels == label_batch).sum().item()
        total_predictions += label_batch.size(0)

    model.eval()  # Set model to evaluation mode
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0
    with torch.no_grad():  # No need to track gradients for validation
      for pos_batch, dir_batch, label_batch in val_dataloader:


          # Forward pass through the model
          predictions = model(pos_batch, dir_batch)

          # If label_batch is not already a float tensor, you should convert it
          if label_batch.dtype is not torch.float32:
              label_batch = label_batch.float()

          # Compute loss
          val_loss = criterion(predictions.squeeze(), label_batch)
          val_running_loss += val_loss.item()

          # Convert predictions to predicted class labels
          predicted_labels = predictions.round().squeeze()

          # Calculate the number of correct predictions
          val_correct_predictions += (predicted_labels == label_batch).sum().item()
          val_total_predictions += label_batch.size(0)



    # Calculate training and validation loss and accuracy
    train_loss = running_loss / len(train_dataloader.dataset)
    train_accuracy = correct_predictions / total_predictions * 100
    val_loss = val_running_loss / len(val_dataloader.dataset)
    val_accuracy = val_correct_predictions / val_total_predictions * 100

    # Print out metrics for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}%")

#save weights
torch.save(model.state_dict(), 'model_weights.pth')

model.eval()  # Set the model to evaluation mode
true_points = []

with torch.no_grad():  # No need to track gradients for validation
    for pos_batch, dir_batch, label_batch in val_dataloader:  # Ignore dir_batch
        output = model(pos_batch, dir_batch)  # Assume model.forward() modified to accept only pos_batch
        predicted_labels = output.round().squeeze()  # Get the predicted labels

        # Now, find which ones are true
        mask = predicted_labels.bool()  # Convert to a boolean mask
        true_pos = pos_batch[mask]  # Filter out the true points

        # Assuming pos_batch is structured as [azimuth, elevation]
        true_azimuth = true_pos[:, 0]
        true_elevation = true_pos[:, 1]

        # Store the azimuth and elevation of true points for plotting
        true_points.extend(zip(true_azimuth.tolist(), true_elevation.tolist()))


