import torch

def compute_kernel(x, y):

    x_size = x.shape[0]
    y_size = y.shape[0]
    x_dim = x.shape[1]
    tiled_x = x.unsqueeze(1).repeat(1, y_size, 1)
    tiled_y = y.unsqueeze(0).repeat(x_size, 1, 1)

    return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2) / x_dim)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)

    debug_x = torch.sum(x_kernel)
    debug_y = torch.sum(y_kernel)
    debug_xy = torch.sum(xy_kernel)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)