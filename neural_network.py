import torch
import torch.nn as nn
import numpy as np
from pca_for_training_data import fit_and_save_pca, load_pca, recover_from_pca
import torch.optim as optim

pca_path = "data/cache/pca_model.pkl"
class FullyConnectedNN(nn.Module):
    def __init__(self, d_in, d_out, hidden_dims=[128, 128], activation=nn.ReLU, dropout=0.0):
        super(FullyConnectedNN, self).__init__()
        layers = []
        input_dim = d_in
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_nn_model(model, input_data, targets, n_epochs=1000, batch_size=100, lambda_reg=0, print_every=100, lr=1e-2, weight_given=torch.empty(0), model_given=False, PCA = True):
    assert input_data.shape[0] == targets.shape[0]

    # Sets the model to training mode 
    # important if using dropout or batchnorm
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_history = []


    loss_mask = torch.ones(1, targets.shape[-1], device=targets.device)
    loss_mask[0, 0] = 10.0

    for epoch in range(n_epochs):
        permutation = torch.randperm(input_data.shape[0])
        input_shuffled = input_data[permutation]
        targets_shuffled = targets[permutation]

        for i in range(0, input_data.shape[0], batch_size):
            batch_input = input_shuffled[i:i+batch_size]
            batch_targets = targets_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            predictions = model(batch_input)
            # loss = loss_fn(predictions, batch_targets) 
            loss = (((predictions - batch_targets) ** 2)*loss_mask).mean()
            l2_loss = lambda_reg * sum((p**2).sum() for p in model.parameters())
            rel_loss = (((predictions - batch_targets) ** 2) * (batch_targets.abs() + 1e-6)).mean()
            # print(loss/rel_loss)
            total_loss = loss + l2_loss + 5e5*rel_loss
            total_loss.backward()
            optimizer.step()

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}, MSE Loss: {loss.item():.6f}, L2 Loss: {l2_loss.item():.6f}, rel Loss: {rel_loss.item():.6f}")
            loss_history.append((epoch + 1, loss.item(), l2_loss.item()))

    return model, loss_history

# TODO very bad configuration. make this a full neural network class. s