# 2.4
# taking this course
# https://www.coursera.org/learn/deep-neural-networks-with-pytorch
# print('\ni be learning deep\n')
print('\nthis is the "hard way" of doing gradient descent\n')
# =============================================================================
import torch  # the only import you ever need to be awesome
import matplotlib.pyplot as plt
# =============================================================================
# =============================================================================
#
# =============================================================================
# setup stuff
# =============================================================================
w = torch.tensor(-10.0, requires_grad=True)
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3*X
# plt.plot(X.numpy(), f.numpy())
Y = f+0.1*torch.randn(X.size())
# plt.plot(X.numpy(), Y.numpy(), 'ro')
# =============================================================================
#
# =============================================================================
# calculate yhat
# =============================================================================
def forward(x):
    return w*x  # yhat = wx
# =============================================================================
# =============================================================================
#
# =============================================================================
# mean squared error
# =============================================================================
def criterion(yhat, y):
    return torch.mean((y- yhat)**2)
# =============================================================================
# =============================================================================
#
# =============================================================================
# GRADIENT DESCENT
# =============================================================================
lr = 0.1
epochs = 5

COST = []

for epoch in range(epochs):
# =============================================================================
    Yhat = forward(X)  # apply function with current w value
# =============================================================================
    loss = criterion(Yhat, Y)  # mean sqaured error
# =============================================================================
    loss.backward()  # derivative our our w value
# =============================================================================
    w.data = w.data-(lr*w.grad.data)  # update w based on gradient and lr
# =============================================================================

    w.grad.data.zero_()  # set gradient to 0 before next iteration
    print(f'\nepoch {epoch}')
    print('loss:',round(loss.item(), 2))
    print('w:', round(w.item(), 2))
    
    COST.append(loss.item())

plt.plot(COST, marker='o')
plt.title("Cost vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cost (Loss)")
plt.grid(True)

    
    