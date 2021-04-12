import numpy as np
import torch as T
import matplotlib.pyplot as plt

from networks import Generator, Discriminator
from Stein_GAN import rbf_kernel, learn_G, learn_D

TRAIN_PARTICLES = 10
NUM_PARTICLES = 100
ITER_NUM = int(3e5)
BATCH_SIZE = 16
IMAGE_SHOW = 5e+2
  

#x = X[:, 0].reshape(-1, 1)
#y = X[:, 1].reshape(-1, 1)

x = []
y = []

def train(alpha=1.0):
    for i in range(ITER_NUM):
        # sample minibatch
        index = np.random.choice(range(len(x)), size=BATCH_SIZE, replace=False)
        mini_x = x[index]
        mini_y = y[index]
        
        x_obs = []
        for j in range(len(mini_x)):
            x_obs.append([mini_x[j], mini_y[j]])
    
        x_obs = T.from_numpy(np.array(x_obs)).float()
        
        # eval_svgd
        learn_G(g_net, d_net, x_obs, batch_size=BATCH_SIZE)
        
        # sample minibatch
        index = np.random.choice(range(len(x)), size=BATCH_SIZE, replace=False)
        mini_x = x[index]
        mini_y = y[index]
        
        x_obs = []
        for j in range(len(mini_x)):
            x_obs.append([mini_x[j], mini_y[j]])
    
        x_obs = T.from_numpy(np.array(x_obs)).float()
        
        # learn discriminator
        learn_D(g_net, d_net, x_obs, batch_size=BATCH_SIZE)
        #print(i)
        if (i+1)%IMAGE_SHOW == 0:
            plt.rcParams["figure.figsize"] = (20,10)
            fig, ax = plt.subplots()
            plt.xlim(-30,30)
            plt.ylim(-20,20)
            
            zeta = T.FloatTensor(3000, 2).uniform_(-10, 10)

            ax.scatter(x, y,s=3,color='blue')
            
            predict = g_net.forward(zeta.cpu()).detach().cpu().squeeze(-1)
            
            # print(zeta)                        
            print(predict)

            
            ax.scatter(predict[:, 0].numpy(), predict[:, 1].numpy(),s=1, color='red')
            # ax.scatter(list(np.linspace(-10,10, 300)), list(np.linspace(-10,10, 300)),s=1, alpha=0.4, color='red')
            # ax.scatter(predict[:, 0], predict[:, 1],s=1, alpha=0.4, color='red')

            
            
            ax.set_title('Iter:'+str(i+1)+' alpha:'+str(alpha))
            plt.show()

g_net = Generator().cpu()
d_net = Discriminator().cpu()
train(1.)
