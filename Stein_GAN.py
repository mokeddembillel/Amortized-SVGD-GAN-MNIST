import numpy as np

import torch as T
import torch.autograd as autograd

print(T.__version__)

device = T.device('cuda' if T.cuda.is_available() else 'cpu')


def rbf_kernel(input_1, input_2,  h_min=1e-3):
    
    k_fix, out_dim1 = input_1.size()[-2:]
    k_upd, out_dim2 = input_2.size()[-2:]
    
    assert out_dim1 == out_dim2
    
    # Compute the pairwise distances of left and right particles.
    diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)  
    #print(diff)

    dist_sq = diff.pow(2).sum(-1)
    #print(dist_sq)
    dist_sq = dist_sq.unsqueeze(-1)
    #print(dist_sq)
    
    # Get median.
    median_sq = T.median(dist_sq, dim=1)[0]
    #print(median_sq)
    median_sq = median_sq.unsqueeze(1)
    #print(median_sq)
    
    h = median_sq / np.log(k_fix + 1.)
    #print(h)

    kappa = T.exp(- dist_sq / (h + 1e-8))
    #print(kappa)

    # Construct the gradient
    #kappa_grad = -2. * diff / (h * kappa + 1e-8)
    kappa_grad = -autograd.grad(kappa.sum(), input_1, retain_graph=True)[0]

    #print(kappa_grad)
    return kappa, kappa_grad


def learn_G(P, g_net, d_net, x_obs, batch_size = 10, alpha=1.):
    # Draw zeta random samples
    zeta = T.FloatTensor(batch_size, 2).uniform_(0, 1)
    #zeta.requires_grad_(True)
    # Forward the noise through the network
    f_x = g_net.forward(zeta)
    #print(f_x)
    ### Equation 7 (Compute Delta xi)
    # Get the energy using the discriminator
    score = d_net.forward(f_x)
    #print(score)
    # Get the Gradients of the energy with respect to x and y
    grad_score = autograd.grad(-score.sum(), f_x)[0].squeeze(-1)
    #print(grad_score)
    grad_score = autograd.grad(P.log_prob(f_x).sum(), f_x)[0].squeeze(-1)
    grad_score = P.log_prob(f_x)

    #print(grad_score)
    # Compute the similarity using the RBF kernel 
    kappa, grad_kappa = rbf_kernel(f_x, f_x) # <<<<<<<<<<<<<<<<<<<<<<<<
    #print(kappa)

    # svgd_x = T.mean(grad_score_x * kappa - 1 * grad_kappa, dim=1).detach()
    #svgd_y = T.mean(grad_score_y * kappa - 1 * grad_kappa, dim=1).detach()
    
    #svgd = T.mean(T.matmul(kappa.squeeze(-1), grad_score) - 1 * grad_kappa, dim=1).detach()
    svgd = (T.matmul(kappa.squeeze(-1), grad_score) - 1 * grad_kappa) / f_x.size(0)

    # print(svgd)
    #T.matmul(grad_score.T, kappa)
    # Equation 8
    f_x_prime = T.clone(zeta)
    for i in range(batch_size):
        f_x_prime[i, 0] = f_x_prime[i, 0] + g_net.lr * svgd[i, 0]
        f_x_prime[i, 1] = f_x_prime[i, 1] + g_net.lr * svgd[i, 0]
        
    # Computing the loss
    
    loss = T.abs(f_x - f_x_prime)
    loss = T.sum(loss, dim=0)
    
    loss = loss.sum(0)
    g_net.optimizer.zero_grad()
    loss.backward()
    g_net.optimizer.step()
  
    # g_net.optimizer.zero_grad()
    # autograd.backward(-f_x, grad_tensors=svgd)
    # g_net.optimizer.step()
    
def learn_D(g_net,d_net, x_obs, batch_size = 10, epsilon = 0.001):
    # Draw zeta random samples
    zeta = T.FloatTensor(batch_size, 2).uniform_(0, 1)
    # Forward the noise through the network
    f_x = g_net.forward(zeta)
    # Get the energy of the observed data using the discriminator
    data_score = d_net.forward(x_obs)
    # Get the energy of the generated data using the discriminator
    gen_score = d_net.forward(f_x)
    
    loss = - d_net.lr * data_score.mean() + d_net.lr * ( 1- 0.7) * gen_score.mean()
    
    
    d_optim.zero_grad()
    autograd.backward(-loss)
    d_optim.step()
    
    # data_grad_score = []
    # for i in range(batch_size):
    #     grad = list(autograd.grad(data_score[i], (d_net.parameters()), retain_graph=True, allow_unused=True, create_graph=True))
    #     for j in range(len(grad)):
    #         grad[j] = grad[j].unsqueeze(0)
    #     data_grad_score.append(grad)
    
    # gen_grad_score = []
    # for i in range(batch_size):
    #     grad = list(autograd.grad(gen_score[i], (d_net.parameters()), retain_graph=True, allow_unused=True, create_graph=True))
    #     for j in range(len(grad)):
    #         grad[j] = grad[j].unsqueeze(0)
    #     gen_grad_score.append(grad)
        
        
    # data_grad_score_mean = []
    # for i in range(len(data_grad_score[0])):
    #     tmp = []
    #     for t in data_grad_score:
    #         tmp.append(t[i])
    #     data_grad_score_mean.append(T.cat(tuple(tmp), dim=0).mean(0))
        
        
    # gen_grad_score_mean = []
    # for i in range(len(gen_grad_score[0])):
    #     tmp = []
    #     for t in gen_grad_score:
    #         tmp.append(t[i])
    #     gen_grad_score_mean.append(T.cat(tuple(tmp), dim=0).mean(0))
    
    # gradients = -1 * data_grad_score_mean + gen_grad_score_mean
    
    # with T.no_grad():
    #     for i, p in enumerate(d_net.parameters()):
    #         p.copy_(p + d_net.lr * gradients[i])
    #         # print(p)
     
   