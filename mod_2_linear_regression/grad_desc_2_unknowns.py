import torch
import matplotlib.pyplot as plt
import numpy as np

w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size())

def forward(x):
    return w*x+b # yhat = wx


def criterion(yhat, y):
    return torch.mean((y- yhat)**2)

lr = 0.2
epochs = 10
COST = []

for epoch in range(epochs):
# =============================================================================
    Yhat = forward(X)  # apply function with current w value
# =============================================================================
    loss = criterion(Yhat, Y)  # mean sqaured error
    print(f'\nepoch {epoch}')
    print('loss:',round(loss.item(), 2))
    print('w:', round(w.item(), 2))
    print('b:', round(b.item(), 2))
# =============================================================================
    loss.backward()  # derivative our our w value
# =============================================================================
    w.data = w.data-(lr*w.grad.data)  # update w based on gradient and lr
    print(w.grad)
    # Assuming X, Y, and Yhat are PyTorch tensors
    # dw = (-2 / len(X)) * torch.matmul(X.T, (Y - Yhat))
    # i manually calculated the derivative to get this
    # print(torch.mean(dw))
    
    
    b.data = b.data-(lr*b.grad.data) 
    # print('updated w:', round(w.item(), 2))
    # print('updated b:', round(b.item(), 2))
# =============================================================================

    w.grad.data.zero_()  # set gradient to 0 before next iteration
    b.grad.data.zero_()
    
    COST.append(loss.item())

    
    # plt.grid(True)
    
    # Plot actual data points
    # plt.figure(figsize=(10, 6))
    # plt.scatter(X.detach().numpy(), Y.detach().numpy(), label="Actual Data", color="blue")
    
    # # Plot predicted values
    # plt.plot(X.detach().numpy(), Yhat.detach().numpy(), label="Predicted Data", color="red")

# # Add labels and title
# plt.title("Actual vs Predicted Values")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.grid(True)
# plt.show()

plt.plot(COST, marker='o')
plt.title("Cost vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cost (Loss)")
