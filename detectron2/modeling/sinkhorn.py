import torch
import torch.nn.functional as F
import torch.distributed as dist
import detectron2.utils.comm as comm

def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = torch.tensor(Q.shape[1]).cuda() #* comm.get_world_size() # number of samples to assign
    
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


def distributed_greenkhorn(out, sinkhorn_iterations=100, epsilon=0.05):
    L = torch.exp(out / epsilon).t()
    K = L.shape[0]
    B = L.shape[1]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    r = torch.ones((K,), dtype=L.dtype).to(L.device) / K
    c = torch.ones((B,), dtype=L.dtype).to(L.device) / B

    r_sum = torch.sum(L, axis=1)
    c_sum = torch.sum(L, axis=0)

    r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
    c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)

    for _ in range(sinkhorn_iterations):
        i = torch.argmax(r_gain)
        j = torch.argmax(c_gain)
        r_gain_max = r_gain[i]
        c_gain_max = c_gain[j]

        if r_gain_max > c_gain_max:
            scaling = r[i] / r_sum[i]
            old_row = L[i, :]
            new_row = old_row * scaling
            L[i, :] = new_row

            L = L / torch.sum(L)
            r_sum = torch.sum(L, axis=1)
            c_sum = torch.sum(L, axis=0)

            r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
            c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)
        else:
            scaling = c[j] / c_sum[j]
            old_col = L[:, j]
            new_col = old_col * scaling
            L[:, j] = new_col

            L = L / torch.sum(L)
            r_sum = torch.sum(L, axis=1)
            c_sum = torch.sum(L, axis=0)

            r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
            c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)

    L = L.t()

    indexs = torch.argmax(L, dim=1)
    G = F.gumbel_softmax(L, tau=0.5, hard=True)

    return L, indexs