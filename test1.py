# import tensorflow as tf;
# print(tf.__version__)
# print(tf.config.list_physical_devices('GPU'))
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# if not gpus:
#     print("No GPU detected.")
# else:
#     print(f"Available GPUs: {gpus}")


# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print(tf.config.list_physical_devices('GPU'))

import torch
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
# import tensorflow as tf

# List available devices
# print("Available devices:", tf.config.list_physical_devices('GPU'))
# print("Is built with CUDA:", tf.test.is_built_with_cuda())
# print("Is GPU available:", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


# import torch
# print("cuDNN available:", torch.backends.cudnn.is_available())
# print("cuDNN version:", torch.backends.cudnn.version())

# import keras
# print("Keras version:", keras.__version__)

# x=torch.rand(5,3)
# print(x)

# import torch.nn as nn
# import torch

# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.layer = nn.Linear(10, 1)  # Example layer
    
#     def forward(self, x):
#         return self.layer(x)

# model = SimpleModel()

# loss_fn = nn.MSELoss()  # Mean Squared Error for regression
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# # Example input and output
# inputs = torch.randn(5, 10)
# targets = torch.randn(5, 1)

# for epoch in range(100):  # Training loop
#     optimizer.zero_grad()  # Clear gradients
#     outputs = model(inputs)
#     loss = loss_fn(outputs, targets)
#     loss.backward()  # Backpropagation
#     optimizer.step()  # Update weights

#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item()}')

