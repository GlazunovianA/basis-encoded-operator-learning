'''
testing for:
random fourier feature
cosine 
 ( other random feature to be added )
'''

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from projections import GaussianTransform

from pca_for_training_data import fit_and_save_pca, load_pca, recover_from_pca

torch.manual_seed(42)

class RFF(torch.nn.Module):
    """ Classical random Fourier features 
     adaptation source:  https://github.com/joneswack/dp-rfs/blob/main/random_features """
    def __init__(self, d_in, d_features, lengthscale='auto', trainable_kernel=False,
                    complex_weights=False, projection_type='gaussian', device='cpu'):
        super(RFF, self).__init__()

        self.d_in = d_in
        self.d_features = d_features
        if not complex_weights:
            self.d_features = self.d_features // 2
        self.complex_weights = complex_weights
        self.projection_type = projection_type

        if isinstance(lengthscale, str) and lengthscale == 'auto':
            lengthscale = np.sqrt(d_in)
        # we activate ARD
        self.log_lengthscale = torch.nn.Parameter(
            torch.ones(d_in, device=device).float() * np.log(lengthscale), requires_grad=trainable_kernel)

        # if self.projection_type == 'srht':
        #     self.feature_encoder = SRHT(self.d_in, self.d_features,
        #                                 complex_weights=False, shuffle=False, k=3, device=device)
        # else:
        self.feature_encoder = GaussianTransform(self.d_in, self.d_features, complex_weights=False, device=device)
        # TODO put gaussian transform into main code
    def resample(self):
        self.feature_encoder.resample()
    
    def forward(self, x):
        x = x / self.log_lengthscale.exp()

        x = self.feature_encoder.forward(x) # x @ weights

        x = torch.stack([torch.cos(x), torch.sin(x)], dim=-1)
        if self.complex_weights:
            x = torch.view_as_complex(x)
        else:
            x = x.view(len(x), -1)
        
        x = x / np.sqrt(self.d_features)

        return x

    def reference_kernel(self, data_x, data_y=None):
        """
        Reference RBF kernel.
        """
        if data_y is None:
            data_y = data_x.clone()

        data_x = data_x / self.log_lengthscale.exp()
        data_y = data_y / self.log_lengthscale.exp()
        distances = torch.cdist(data_x, data_y, p=2.0)

        return torch.exp(-distances**2 / 2.)


def pad_data_pow_2(data, offset=0):
    """
    Pads the input with zeros s.t. d=2**i - offset
    dimensional matching with feature model dim
    """

    d_new = int(2**np.ceil(np.log2(data.shape[1]+offset)))
    placeholder = torch.zeros(len(data), d_new-offset, device=data.device)
    placeholder[:, :data.shape[1]] = data
    
    return placeholder


class NN_cff(torch.nn.Module):
    '''random cosine feature map '''
    def __init__(self, d_in, d_features, lengthscale='auto', sigma = 1.0, trainable_kernel=False,
                    complex_weights=False, projection_type='Gaussian', device='cpu'):
        super(NN_cff, self).__init__()

        
        self.d_in = d_in
        self.d_features = d_features
        self.sigma = np.array(sigma)
        self.complex_weights = False
        self.device = device
        self.kernel = projection_type
        self.uniform_kernel = False # TODO
        self.trainable_kernel = trainable_kernel
        if isinstance(lengthscale, str) and lengthscale == 'auto':
            lengthscale = np.sqrt(d_in)
        # we activate ARD
        else:
            lengthscale = lengthscale*np.sqrt(d_in)
            # self.log_lengthscale = torch.ones(d_in, device=device).float() * np.log(lengthscale)
        self.lengthscale = torch.nn.Parameter(
                torch.ones(d_in, device=device).float() * lengthscale, requires_grad=trainable_kernel)
        
        self.W = torch.nn.Parameter(None, requires_grad = False)
        self.b = torch.nn.Parameter(None, requires_grad = False)
        
        # self.model = nn.Sequential(nn.Linear(self.d_features, 1))
    
    def resample(self):
        dtype = torch.cfloat if self.complex_weights else torch.float32
        if self.sigma.shape: # when sigma is vector
            stds = torch.tensor(self.sigma, dtype=dtype) # .to(self.device)  # TODO to be tuned
            assert stds.shape == (self.d_in,), ' dimension of sigma must match model dimensions'
            # stds = torch.sqrt(variances)  # std devs per dimension
            print(stds)

            if self.kernel == 'Gaussian':
                # Sample W ~ N(0, W) where W is diagonal covariance
                normal = torch.randn(self.d_in, self.d_features, dtype=dtype, device=self.device)
                print('stds, normal', stds.shape, normal.shape)
            elif self.kernel == 'Cauchy':
                laplace = torch.distributions.Laplace(
                    torch.tensor(0.0, dtype=dtype, device=self.device),
                    torch.tensor(1.0, dtype=dtype, device=self.device)
                )
                normal = laplace.sample((self.d_in, self.d_features))
            else:
                raise ValueError(f"Unknown projection type: {self.kernel}")
            self.W.data = stds.unsqueeze(1) * normal  # scale each row i by std[i]
            print(self.W.shape, self.W.dtype, self.W.device)
            # Sample b ~ Uniform[0, 2pi]
            self.b.data = torch.rand(self.d_features, dtype=dtype, device=self.device) * 2 * torch.pi
        else:           
            self.W.data = torch.randn(self.d_in, self.d_features, dtype=dtype, device=self.device)*self.sigma
            self.b.data = torch.randn(self.d_features, dtype=dtype, device=self.device)* 2 * torch.pi 
    def forward(self,x):
        print(x.shape, self.lengthscale.shape)
        x = x / self.lengthscale
        scale = torch.sqrt(torch.tensor(2.0 / self.d_features))
        print(x.dtype, self.W.dtype, self.b.dtype)
        x = torch.cos(x @ self.W + self.b) * scale
        print('x dim', x.shape) # (num_sample, features)
        # x = x / np.sqrt(self.d_features)
        return x
    
    def reference_kernel(self, data_x, data_y=None):
        """
        Reference RBF kernel.
        """
        if data_y is None:
            data_y = data_x.clone()

        data_x = data_x / self.lengthscale
        data_y = data_y / self.lengthscale
        distances = torch.cdist(data_x, data_y, p=2.0)

        return torch.exp(-distances**2 / 2.)
    
def adjusted_loss(loss, alpha=5.0):
    return loss + alpha * torch.exp(-loss)

# class NN_cff_vec(torch.nn.Module):
#     '''random cosine feature map '''
#     def __init__(self, d_in, d_features, d_out, lengthscale='auto', sigma_in = 1.0, sigma_out = 1.0, trainable_kernel=False,
#                     complex_weights=False, projection_type='Gaussian', device='cpu'):
#         super(NN_cff_vec, self).__init__()

        
#         self.d_in = d_in
#         self.d_features = d_features
#         self.d_out = d_out # TODO assuming d_in = d_out
#         self.sigma_in = np.array(sigma_in)
#         self.sigma_out = np.array(sigma_out)
#         self.complex_weights = False
#         self.device = device
#         self.kernel = projection_type
#         self.uniform_kernel = False # TODO
#         self.trainable_kernel = trainable_kernel
#         if isinstance(lengthscale, str) and lengthscale == 'auto':
#             lengthscale = np.sqrt(d_in)
#         else:
#             lengthscale = lengthscale*np.sqrt(d_in)
#             # self.log_lengthscale = torch.ones(d_in, device=device).float() * np.log(lengthscale)
#         self.lengthscale = torch.nn.Parameter(
#                 torch.ones(d_in, device=device).float() * lengthscale, requires_grad=trainable_kernel)
        
#         self.W = torch.nn.Parameter(None, requires_grad = False)
#         self.b = torch.nn.Parameter(None, requires_grad = False)
        
#         # self.model = nn.Sequential(nn.Linear(self.d_features, 1))
    
#     def resample(self):
#         dtype = torch.cfloat if self.complex_weights else torch.float32
#         # if self.sigma.shape : # when sigma is vector
#             # stds = torch.tensor(self.sigma, dtype=dtype) # .to(self.device)  # TODO to be tuned
#             # assert stds.shape == (self.d_in,), ' dimension of sigma must match model dimensions'
#             # std = torch.zeros((self.d_in, self.d_in))
#             # # TODO separate scaling for std_i and std_j (input and output)!!!
#             # for i in range (self.d_in):
#             #     for j in range (self.d_in):
#             #         std[i, j] = stds[i]*stds[j]
            
#             # stds = torch.sqrt(variances)  # std devs per dimension
#             # print(std)
#         # sigma_in -> shape (d_in,)
#         if np.array(self.sigma_in).shape:
#             sigma_in = torch.tensor(self.sigma_in, device=self.device, dtype=dtype)
#             assert sigma_in.shape == (self.d_in,)
#         else:
#             sigma_in = torch.full((self.d_in,), float(self.sigma_in), device=self.device, dtype=dtype)

#         # sigma_out -> shape (d_out,)
#         if np.array(self.sigma_out).shape:
#             sigma_out = torch.tensor(self.sigma_out, device=self.device, dtype=dtype)
#             assert sigma_out.shape == (self.d_out,)
#         else:
#             sigma_out = torch.full((self.d_out,), float(self.sigma_out), device=self.device, dtype=dtype)

#         if self.kernel == 'Gaussian':
#             # Sample W ~ N(0, W) where W is diagonal covariance
#             normal = torch.randn(self.d_in, self.d_features, self.d_out, dtype=dtype, device=self.device)
#             # print('std, normal', normal.shape)
#         elif self.kernel == 'Cauchy':
#             laplace = torch.distributions.Laplace(
#                 torch.tensor(0.0, dtype=dtype, device=self.device),
#                 torch.tensor(1.0, dtype=dtype, device=self.device)
#             )
#             normal = laplace.sample((self.d_in, self.d_features, self.d_out), dtype=dtype)
#         else:
#             raise ValueError(f"Unknown projection type: {self.kernel}")
#         self.W.data = normal * sigma_in[:, None, None] * sigma_out[None, None, :]  # scale each row i by std[i]
#         # print(self.W.shape, self.W.dtype, self.W.device)
#         # Sample b ~ Uniform[0, 2pi]
#         self.b.data = torch.rand(self.d_features, self.d_out, dtype=dtype, device=self.device) * 2 * torch.pi
#         # else:  
#         #     # TODO         
#         #     self.W.data = torch.randn(self.d_in, self.d_features, dtype=dtype, device=self.device)*self.sigma
#         #     self.b.data = torch.randn(self.d_features, dtype=dtype, device=self.device)* 2 * torch.pi 
#     def forward(self,x):
#         print(x.shape, self.lengthscale.shape)
#         x = x / self.lengthscale
#         scale = torch.sqrt(torch.tensor(2.0 / self.d_features))
#         # print('shape check')
#         # print(x.shape, self.W.shape, self.b.shape)
#         # torch.Size([900, 9]) torch.Size([9, 512, 9]) torch.Size([512, 9])
#         xw = torch.einsum('ni,ifd->nfd', x, self.W)

#         # print('xw', xw.shape)
#         # torch.Size([900, 512, 9])
#         x = torch.cos(xw + self.b) * scale
#         # print('x dim', x.shape) # (num_sample, d_out, features)
#         # x = x / np.sqrt(self.d_features)
#         return x
    

class NN_cff_vec(torch.nn.Module):
    """Random cosine feature map producing features of shape (n, d_features, d_out)."""

    def __init__(
        self,
        d_in: int,
        d_features: int,
        d_out: int,
        lengthscale="auto",
        sigma_in=1.0,
        sigma_out=1.0,
        trainable_kernel: bool = False,
        complex_weights: bool = False,
        projection_type: str = "Gaussian",
        device: str = "cpu",
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_features = int(d_features)
        self.d_out = int(d_out)

        self.sigma_in = np.array(sigma_in)
        self.sigma_out = np.array(sigma_out)

        self.complex_weights = bool(complex_weights)
        self.device = device
        self.kernel = projection_type
        self.trainable_kernel = bool(trainable_kernel)

        # lengthscale vector (broadcasted per input dim)
        if isinstance(lengthscale, str) and lengthscale == "auto":
            ls = np.sqrt(self.d_in)
        else:
            ls = float(lengthscale) * np.sqrt(self.d_in)  # your current convention
        self.lengthscale = torch.nn.Parameter(
            torch.ones(self.d_in, device=device, dtype=torch.float32) * ls,
            requires_grad=trainable_kernel,
        )

        # W and b are fixed random tensors (no gradients): store as buffers
        dtype = torch.cfloat if self.complex_weights else torch.float32
        self.register_buffer("W", torch.empty(self.d_in, self.d_features, self.d_out, dtype=dtype, device=device))
        self.register_buffer("b", torch.empty(self.d_features, self.d_out, dtype=dtype, device=device))

        self.resample()

    def resample(self):
        dtype = self.W.dtype

        # sigma_in -> (d_in,)
        if np.array(self.sigma_in).shape:
            sigma_in = torch.tensor(self.sigma_in, device=self.device, dtype=dtype)
            assert sigma_in.shape == (self.d_in,)
        else:
            sigma_in = torch.full((self.d_in,), float(self.sigma_in), device=self.device, dtype=dtype)

        # sigma_out -> (d_out,)
        if np.array(self.sigma_out).shape:
            sigma_out = torch.tensor(self.sigma_out, device=self.device, dtype=dtype)
            assert sigma_out.shape == (self.d_out,)
        else:
            sigma_out = torch.full((self.d_out,), float(self.sigma_out), device=self.device, dtype=dtype)

        if self.kernel == "Gaussian":
            base = torch.randn(self.d_in, self.d_features, self.d_out, dtype=dtype, device=self.device)
        elif self.kernel == "Cauchy":
            laplace = torch.distributions.Laplace(
                torch.tensor(0.0, dtype=dtype, device=self.device),
                torch.tensor(1.0, dtype=dtype, device=self.device),
            )
            base = laplace.sample((self.d_in, self.d_features, self.d_out))
        else:
            raise ValueError(f"Unknown projection type: {self.kernel}")

        # scale per input dim and per output dim
        self.W.copy_(base * sigma_in[:, None, None] * sigma_out[None, None, :])

        # b ~ Uniform[0,2pi]
        self.b.copy_(torch.rand(self.d_features, self.d_out, dtype=dtype, device=self.device) * (2.0 * torch.pi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, d_in)
        x = x / self.lengthscale  # broadcast across batch
        scale = torch.sqrt(torch.tensor(2.0 / self.d_features, device=x.device, dtype=x.dtype))
        xw = torch.einsum("ni,ifd->nfd", x, self.W)
        return torch.cos(xw + self.b) * scale  # (n, d_features, d_out)

def train_feature_model(model, input_data, targets, n_epochs=1000, batch_size=100, print_every=100, lr=1e-2, lambda_reg = 1.0, weight_given = torch.empty(0), model_given = False, PCA = True, vectorize = False):
    '''
    input_data, targets: tensor with shape [N, m*m].
     
      this is only needed (outside the ridge regression model) when the problem is not convex. for eg when we use trainable kernel lengthscale etc. '''
    #TODO test trainable kernel lengthscale and its gradient behavior.
    if model_given:
        model = model
    else:
        model.resample()
    assert input_data.shape[0] == targets.shape[0]
    # print('input', input_data.shape, 'target', targets.shape)
    # assert len(input_data.shape) == 2
    # assert len(targets.shape) == 2

    features = model.forward(input_data)
    # print('features.shape', features.shape) # [N, d_features, d_in] 
    d_features = features.shape[1]
    if vectorize:
        weights_dim2 = 1
    else:
        weights_dim2 = targets.shape[-1]

    if weight_given.numel() > 0:
        if weight_given.shape == (d_features, weights_dim2):
            weights = weight_given
            print(f'using given weight with shape{(d_features, weights_dim2)}.')
        else:
            raise ValueError(f"Given weight shape {weight_given.shape} does not match expected shape {(d_features, weights_dim2)}.")
    else:
        weights = torch.randn(d_features, weights_dim2, requires_grad=True)
    optimizer = optim.SGD([weights], lr=lr)
    loss_history = []

    # loss_mask = torch.ones(1, targets.shape[-1], device=targets.device)
    # loss_mask[0, 0] = 36.0  # x weight on the first entry
    # loss_mask[0, torch.int(torch.sqrt(input_data.shape[-1])) + 1] = 10.0
    
    ##### test

    print("Mean of target[0]:", targets[:, 0].mean().item())
    print("Std of target[0]:", targets[:, 0].std().item())
    # print('first entry:', targets[0, 0])
    # print("Compared to other entries:", targets[:, 1].mean().item(), targets[:, 1].std().item())
    # print("Compared to other entries:", targets[:, 7].mean().item(), targets[:, 7].std().item())
    
    #########
    for epoch in range(n_epochs):
        permutation = torch.randperm(features.shape[0]) # sample dimension shuffling
        features_shuffled = features[permutation]
        targets_shuffled = targets[permutation]
        # print(permutation.shape)
        # print(features_shuffled.shape)
        # print(targets_shuffled.shape)

        for i in range(0, features.shape[0], batch_size):
            batch_features = features_shuffled[i:i+batch_size]
            batch_targets = targets_shuffled[i:i+batch_size]
            # print('batch', batch_features.shape)
            optimizer.zero_grad()
            if vectorize:

                predictions = (batch_features * weights.view(1, d_features, 1)).sum(dim = 1)# (torch.concat([weights]*targets.shape[-1], dim = 1))
                # print('weight shape', (torch.concat([weights]*targets.shape[-1], dim = 1)).shape)
                # print('prediction shape', predictions.shape)
            else:
                predictions = torch.einsum('bfd,fd->bd', batch_features, weights)

                # predictions = (batch_features * weights.unsqueeze(0)).sum(dim=1)  # shape: [batch_size, d_out]
                # print('predictions.shape', predictions.shape)

            # print(predictions.shape, batch_targets.shape)
            
            # mse_loss = (((predictions - batch_targets) ** 2) * loss_mask).mean()
            # sum_loss = torch.sum(((predictions - batch_targets) ** 2) * loss_mask, 1).mean()
            # amp_loss = adjusted_loss(mse_loss, alpha=5.0)
            mse_loss = (((predictions - batch_targets) ** 2) ).mean()
            l2_loss = (weights ** 2).mean() # lambda_reg * 
            reg = lambda_reg
            loss = mse_loss + reg*l2_loss
            loss.backward(retain_graph=model.trainable_kernel)
            # torch.nn.utils.clip_grad_norm_([weights], max_norm=10.0)
            optimizer.step()

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}, MSE Loss: {mse_loss.item():.4f}, L2 Loss: {l2_loss.item():.4f}")
            loss_history.append((epoch + 1, mse_loss.item(), l2_loss.item()))
            # print('amp', ((predictions - batch_targets) ** 2)[:,0] * 10)# loss_mask)
            # print('ori', ((predictions - batch_targets) ** 2)[:,0])
            
    return model, weights, loss_history







# TODO if training for lengthscale is desired, features need to be resampled to fit lengthscale every update. 


# if __name__ == "__main__":

#     ''' 
#     very bad example, should use softmax instead.
#     TODO: use softmax for classification
#     ##
#     weights = torch.randn(d_features, 10, requires_grad=True)  # One column per class
#     ...
#     predictions = batch_features @ weights  # shape: [batch_size, 10]
#     loss = nn.CrossEntropyLoss()(predictions, batch_targets.long())
#     ##
#     '''
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#     train_data = train_dataset.data.float().view(-1, 28*28)
#     train_labels = train_dataset.targets.float()
#     test_data = test_dataset.data.float().view(-1, 28*28)
#     test_labels = test_dataset.targets.float()

#     data = train_data[torch.randperm(len(train_data))][:1000]
#     data = pad_data_pow_2(data)
#     lengthscale = torch.cdist(data, data, p=2.0).median()

#     for d_features in [1024, 2048, 4096, 8192]:
#         model = NN_cff(d_in=data.shape[1], d_features=d_features, lengthscale=lengthscale)
#         trained_model, weights, loss_progress = train_feature_model(model, data, train_labels[:len(data)], n_epochs=2000)
#         test_data_pad = pad_data_pow_2(test_data)
#         features_test = trained_model.forward(test_data_pad)
#         predictions = features_test @ weights
#         # print(predictions.squeeze().shape, test_labels.shape)
#         test_mse = ((predictions.squeeze() - test_labels) ** 2).mean().item()
#         print(f"[d_features={d_features}] Test MSE: {test_mse:.4f}")

#     print("Done!")



# if __name__ == "__main__":
#     # write training as separate function
#     complex_weights = False
#     projection_type = 'gaussian'

#     # Define a transform to convert images to tensors and normalize them
#     transform = transforms.Compose([
#         transforms.ToTensor(),  # Convert images to PyTorch tensors
#         transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
#     ])

#     # Download and load the MNIST dataset
#     train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#     # Extract data and labels
#     train_data = train_dataset.data.float()  # Shape: (60000, 28, 28)
#     train_labels = train_dataset.targets  # Shape: (60000)
#     test_data = test_dataset.data.float()  # Shape: (10000, 28, 28)
#     test_labels = test_dataset.targets  # Shape: (10000)

#     # Flatten the images into vectors (28x28 -> 784)
#     train_data = train_data.view(train_data.size(0), -1)  # Shape: (60000, 784)
#     test_data = test_data.view(test_data.size(0), -1)  # Shape: (10000, 784)

#     # Combine into a tuple to match the expected format
#     data = ('MNIST', train_data, test_data, train_labels, test_labels)

#     data_name, train_data, test_data, train_labels, test_labels = data

#     data = train_data[torch.randperm(len(train_data))][:1000]
#     data = pad_data_pow_2(data) # 1024
#     lengthscale = torch.cdist(data, data, p=2.0).median()
#     lambda_reg = 1.0  # Regularization strength

#     batch_size = 64  # Mini-batch size
#     n_epochs = 2000  # Number of epochs
#     for d_features in [1024, 2048, 4096, 8192]:
#         ts = NN_cff(
#             data.shape[1], d_features,
#             lengthscale=lengthscale, trainable_kernel=False, #dtype=torch.FloatTensor, 
#             complex_weights=complex_weights, projection_type=projection_type)
#         ts.resample() # initialize self.weights with torch.randn()
#         features = ts.forward(data)
#         y = train_labels  # Shape: (n_train, 1)

#         d_features = features.shape[1]
#         a = torch.randn(d_features, 1, requires_grad=True)  # Initialize coefficients
        
#         learning_rate = 1e-3
#         optimizer = optim.SGD([a], lr=learning_rate)
#         ref_kernel = ts.reference_kernel(data)
#         approx_kernel = features @ features.conj().t()
#         if complex_weights:
#             approx_kernel = approx_kernel.real
#         # score = torch.abs(approx_kernel - ref_kernel) / torch.abs(ref_kernel)
#         mse = (ref_kernel - approx_kernel).pow(2).mean().item()
#         print(d_features, mse)
#         # correct approximation for kernel matrix validated 
        

#         for epoch in range(n_epochs):
#             # Shuffle the data
#             permutation = torch.randperm(features.shape[0])
#             features_shuffled = features[permutation]
#             y_shuffled = y[permutation]

#             # Mini-batch SGD
#             for i in range(0, features.shape[0], batch_size):
#                 # Get mini-batch
#                 batch_features = features_shuffled[i:i+batch_size]
#                 batch_y = y_shuffled[i:i+batch_size]

#                 # Zero gradients
#                 optimizer.zero_grad()
#                 # print(batch_features)
#                 # Compute predictions
#                 y_pred = batch_features @ a
#                 # NOTE y_pred&batch_y shape 
#                 # print((y_pred.t()-batch_y))
#                 # Compute loss (MSE + L2 regularization)
#                 mse_loss = ((y_pred-batch_y.t())**2).mean()
#                 l2_loss = lambda_reg * (a ** 2).sum()
#                 loss = mse_loss + l2_loss

#                 # Backpropagate
#                 loss.backward()

#                 # Update coefficients
#                 optimizer.step()

#             # Print loss every 100 epochs
#             if (epoch + 1) % 100 == 0:
#                 print(f"Epoch {epoch + 1}, MSE Loss: {mse_loss.item()}, L2 Loss: {l2_loss.item()}, ")
#         test_data_pad = pad_data_pow_2(test_data)
#         features_test = ts.forward(test_data_pad)  # Shape: (n_test, d_features)
#         y_pred = features_test @ a  # Shape: (n_test, 1)

#         # Evaluate the model
#         mse = ((y_pred - test_labels) ** 2).mean().item()
#         print("Test MSE:", mse)

#     print('Done!')


# TODO redo all experiments with uniform dist for all input. 