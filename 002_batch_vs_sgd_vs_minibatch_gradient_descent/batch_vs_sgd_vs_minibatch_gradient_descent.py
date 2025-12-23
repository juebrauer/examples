import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Configure loguru to write to a file
logger.add("experiment_results.log", mode="w", format="{time} {level} {message}")

# 1. Load data
URL = "http://www.juergenbrauer.org/datasets/0009_real_estates_ames.csv"
df = pd.read_csv(URL)

# 2. Define inputs and outputs
x_df = df[["YearBuilt", "GarageCars", "OverallQual"]]
x_df = pd.get_dummies(x_df, dtype=int)

x = x_df.values
y = df[["SalePrice"]].values

print(f"Input feature dimension: {x.shape[1]}")
print(f"Feature columns: {list(x_df.columns)}\n")

# 3. Define neural network
class HousePricePredictor(nn.Module):
    def __init__(self, input_size):
        super(HousePricePredictor, self).__init__()
        self.hidden1 = nn.Linear(input_size, 10)
        self.output = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.output(x)
        return x


# 4. Function to calculate MAPE
def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    Returns MAPE as a percentage
    """
    y_true = y_true.detach().numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.detach().numpy() if torch.is_tensor(y_pred) else y_pred
    
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


# 5. Training function
def train_model(model, x_train, y_train, dataloader, epochs, method_name):
    """
    Trains a model and returns the loss history
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loss_history = []
    
    for epoch in range(epochs):
        if method_name == "Batch GD":
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            
        elif method_name == "Stochastic GD":
            epoch_loss = 0
            for i in range(len(x_train)):
                optimizer.zero_grad()
                y_pred = model(x_train[i:i+1])
                loss = criterion(y_pred, y_train[i:i+1])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss_history.append(epoch_loss / len(x_train))
            
        elif method_name == "Mini-Batch GD":
            epoch_loss = 0
            batch_count = 0
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            loss_history.append(epoch_loss / batch_count)
    
    return loss_history


# 6. Run experiments with 25 different random seeds
num_experiments = 25
all_mapes = {
    'Batch GD': [],
    'Stochastic GD': [],
    'Mini-Batch GD': []
}
all_losses = {
    'Batch GD': [],
    'Stochastic GD': [],
    'Mini-Batch GD': []
}

logger.info("="*60)
logger.info("STARTING 25 EXPERIMENTS")
logger.info("="*60)

# Create 5x5 subplot grid
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = axes.flatten()

for exp_idx in range(num_experiments):
    seed = 40 + exp_idx  # Use seeds 40, 41, 42, ..., 64
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {exp_idx + 1}/{num_experiments} (seed={seed})")
    print(f"{'='*60}")
    
    logger.info(f"\n--- Experiment {exp_idx + 1} (seed={seed}) ---")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed
    )
    
    # Scale features
    scaler_input = MinMaxScaler()
    scaler_output = MinMaxScaler()
    
    x_train = scaler_input.fit_transform(x_train)
    x_test = scaler_input.transform(x_test)
    
    y_train_scaled = scaler_output.fit_transform(y_train)
    y_test_scaled = scaler_output.transform(y_test)
    
    # Convert to PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    input_size = x_train.shape[1]
    
    # Batch Gradient Descent
    model_batch = HousePricePredictor(input_size)
    loss_batch = train_model(
        model_batch, x_train_tensor, y_train_tensor, 
        None, epochs=100, method_name="Batch GD"
    )
    
    # Stochastic Gradient Descent
    model_sgd = HousePricePredictor(input_size)
    loss_sgd = train_model(
        model_sgd, x_train_tensor, y_train_tensor, 
        None, epochs=100, method_name="Stochastic GD"
    )
    
    # Mini-Batch Gradient Descent
    model_minibatch = HousePricePredictor(input_size)
    batch_size = 32
    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_minibatch = train_model(
        model_minibatch, x_train_tensor, y_train_tensor, 
        dataloader, epochs=100, method_name="Mini-Batch GD"
    )
    
    # Store loss histories
    all_losses['Batch GD'].append(loss_batch)
    all_losses['Stochastic GD'].append(loss_sgd)
    all_losses['Mini-Batch GD'].append(loss_minibatch)
    
    # Evaluate test performance using MAPE
    model_batch.eval()
    model_sgd.eval()
    model_minibatch.eval()
    
    with torch.no_grad():
        pred_batch_scaled = model_batch(x_test_tensor)
        pred_sgd_scaled = model_sgd(x_test_tensor)
        pred_minibatch_scaled = model_minibatch(x_test_tensor)
        
        pred_batch = scaler_output.inverse_transform(pred_batch_scaled.numpy())
        pred_sgd = scaler_output.inverse_transform(pred_sgd_scaled.numpy())
        pred_minibatch = scaler_output.inverse_transform(pred_minibatch_scaled.numpy())
        
        mape_batch = calculate_mape(y_test, pred_batch)
        mape_sgd = calculate_mape(y_test, pred_sgd)
        mape_minibatch = calculate_mape(y_test, pred_minibatch)
    
    # Store MAPEs
    all_mapes['Batch GD'].append(mape_batch)
    all_mapes['Stochastic GD'].append(mape_sgd)
    all_mapes['Mini-Batch GD'].append(mape_minibatch)
    
    # Log results
    logger.info(f"Batch GD:      MAPE = {mape_batch:.2f}%")
    logger.info(f"Stochastic GD: MAPE = {mape_sgd:.2f}%")
    logger.info(f"Mini-Batch GD: MAPE = {mape_minibatch:.2f}%")
    
    print(f"\nExperiment {exp_idx + 1} Results:")
    print(f"  Batch GD:      MAPE = {mape_batch:.2f}%")
    print(f"  Stochastic GD: MAPE = {mape_sgd:.2f}%")
    print(f"  Mini-Batch GD: MAPE = {mape_minibatch:.2f}%")
    
    # Plot in the corresponding subplot
    ax = axes[exp_idx]
    ax.plot(loss_batch, label='Batch GD', linewidth=1.5)
    ax.plot(loss_sgd, label='Stochastic GD', alpha=0.5, linewidth=0.8)
    ax.plot(loss_minibatch, label='Mini-Batch GD', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('Loss (MSE)', fontsize=8)
    ax.set_title(f'Exp {exp_idx + 1} (seed={seed})', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    
    # Only show legend on first subplot
    if exp_idx == 0:
        ax.legend(fontsize=7)

# Adjust layout and save
plt.suptitle('Comparison: Gradient Descent Methods - 25 Experiments', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('gradient_descent_experiments_25runs.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*60}")
print("Plot saved as: gradient_descent_experiments_25runs.png")
print(f"{'='*60}")
plt.show()


# 7. Calculate and log average MAPEs
print(f"\n{'='*60}")
print("SUMMARY: AVERAGE MAPE ACROSS ALL 25 EXPERIMENTS")
print(f"{'='*60}")

logger.info("\n" + "="*60)
logger.info("SUMMARY: AVERAGE MAPE ACROSS ALL 25 EXPERIMENTS")
logger.info("="*60)

avg_mape_batch = np.mean(all_mapes['Batch GD'])
avg_mape_sgd = np.mean(all_mapes['Stochastic GD'])
avg_mape_minibatch = np.mean(all_mapes['Mini-Batch GD'])

std_mape_batch = np.std(all_mapes['Batch GD'])
std_mape_sgd = np.std(all_mapes['Stochastic GD'])
std_mape_minibatch = np.std(all_mapes['Mini-Batch GD'])

print(f"Batch GD:        Average MAPE = {avg_mape_batch:.2f}% (±{std_mape_batch:.2f}%)")
print(f"Stochastic GD:   Average MAPE = {avg_mape_sgd:.2f}% (±{std_mape_sgd:.2f}%)")
print(f"Mini-Batch GD:   Average MAPE = {avg_mape_minibatch:.2f}% (±{std_mape_minibatch:.2f}%)")
print(f"{'='*60}")

logger.info(f"Batch GD:        Average MAPE = {avg_mape_batch:.2f}% (±{std_mape_batch:.2f}%)")
logger.info(f"Stochastic GD:   Average MAPE = {avg_mape_sgd:.2f}% (±{std_mape_sgd:.2f}%)")
logger.info(f"Mini-Batch GD:   Average MAPE = {avg_mape_minibatch:.2f}% (±{std_mape_minibatch:.2f}%)")
logger.info("="*60)

print("\nResults logged to: experiment_results.log")