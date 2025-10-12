"""
Coordination Game with Bayesian Network Policy (Tabular Exact Policy Gradient)

WORKFLOW SUMMARY:
================
This module implements a coordination game environment with Bayesian network-based policies
for multi-agent reinforcement learning using exact policy gradient methods.

Main Components:
1. Net: Neural network policy model
   - Supports tabular and Bayesian network policies
   - Defines parent-child relationships based on graph structure (all_ones, all_zeros, line)
   - Computes action probabilities conditioned on parent actions

2. CoordinationGame: Main game environment and training loop
   - Initialization: Sets up game parameters, rewards, transition probabilities
   - Policy Update Workflow (update_policy method):
     a. Initialize state transition probabilities (first call only)
     b. Compute best achievable value (optimal policy, first call only)
     c. Get action probabilities from Bayesian policy
     d. Compute Q-values using state-action rewards and transition dynamics
     e. Compute state visitation distribution (d)
     f. Calculate Nash Equilibrium gap for each agent
     g. Compute policy gradient loss
     h. Backpropagate and update policy parameters
     i. Return current value, Price of Anarchy (PoA), and NE gap

Key Methods:
- get_Q(): Solves for Q-values using linear system (Bellman equation)
- get_V(): Computes state values from Q-values and action probabilities
- get_d(): Computes stationary state distribution under current policy
- get_NE_Gap_i(): Calculates Nash Equilibrium gap for agent i (best response improvement)
- value_iteration(): Iteratively computes optimal value function
- get_states_probs(): Computes state transition probabilities from action space

Typical Usage:
--------------
# Initialize game
game = CoordinationGame(n=3, epsilon=0.1, gamma=0.95, mu=initial_dist, 
                        eta=0.01, policy='tabular_baysian', G_type='all_ones')

# Training loop
for iteration in range(num_iterations):
    value, poa, ne_gap = game.update_policy()
    # value: current policy value
    # poa: Price of Anarchy (ratio to optimal value)
    # ne_gap: Nash Equilibrium gap (improvement potential)
"""

import torch
import numpy as np
import torch.nn.functional as F
import copy
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

class Net(nn.Module):
    """
    Bayesian Network Policy Model
    
    Implements a policy network with parent-child dependencies defined by a graph structure.
    Each agent's action probability depends on its parents' actions in the Bayesian network.
    
    Args:
        n: Number of agents
        policy: Policy type ('tabular' or 'tabular_baysian')
        G_type: Graph topology ('all_ones', 'all_zeros', or 'line')
        device: Computing device ('cpu' or 'cuda')
    """
    def __init__(self, n, policy = 'tabular', G_type = 'all_zeros', device = 'cpu'):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.policy = policy
        self.n = n
        self.device = device
        if G_type == 'all_ones':
            self.G = torch.triu(torch.ones(self.n, self.n), diagonal=1).int()
        elif G_type == 'all_zeros':
            self.G = torch.zeros(self.n, self.n).int()
        elif G_type == 'line':
            self.G = self.get_line(n).int()
        else:
            raise NotImplementedError("Not implemented")
        self.size_s = 2**n
        self.size_a = 2**n
        self.parents = {}
        if self.policy == 'tabular':
            self.m = nn.Parameter(torch.empty(2**n, 2).normal_(mean=0,std=1), requires_grad=True)
        elif self.policy == 'tabular_baysian':
            parameter_lists = []
            for i in range(self.n):
                parents = set()
                for j in range(self.n):
                    if self.G[:,i][j]==1:
                        parents.add(j)
                self.parents[i] = parents
                num_parents = len(parents)
                parameter_lists.append(nn.Parameter(torch.empty(2**n, 2**num_parents, 2).normal_(mean=0,std=1), requires_grad=True))
            self.m = nn.ParameterList(parameter_lists)
        else:
            raise NotImplementedError("Not implemented")
    
    def helper(self, i):
        """Computes action probabilities for agent i conditioned on parent agents"""
        action_probs_i = self.softmax(self.m[i].view(-1,2)) #s * num_parents * 2
        action_probs_i = action_probs_i.view(self.size_s, 2**(len(self.parents[i])), 2)
        shape = [self.size_s]
        repeat_amounts = [1]
        for j in range(self.n):
            if j in self.parents[i]:
                shape.append(2)
                repeat_amounts.append(1)
            elif j !=i:
                shape.append(1)
                repeat_amounts.append(2)
            else:
                continue
        shape.append(2)
        repeat_amounts.append(1)
        action_probs_i = action_probs_i.view(shape)
        action_probs_i = action_probs_i.repeat(repeat_amounts)
        action_probs_i = action_probs_i.view(self.size_s, 2**i, 2**(self.n-i-1), 2)
        action_probs_i = action_probs_i.permute((0,1,3,2))
        action_probs_i = action_probs_i.reshape(self.size_s, self.size_a)
        return action_probs_i
        
    

    def get_line(self, n):
        line = torch.zeros((n,n))
        for i in range(n-2):
            line[i][i+1] = 1
        return line.int()

    def init(self):
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias)
        nn.init.normal_(self.fc3.weight)
        nn.init.normal_(self.fc3.bias)

    def softmax(self, x):
        z = x - torch.max(x,dim=1).values.view(len(x),1).repeat(1,len(x[0]))
        numerator = torch.exp(z)
        denominator = torch.sum(numerator,dim=1).view(len(x),1).repeat(1,len(x[0]))
        softmax = numerator/denominator
        return softmax

    def forward(self, x):
        """Forward pass: computes joint action probabilities for all agents"""
        if self.policy == 'tabular':
            return self.m
        elif self.policy == 'tabular_baysian':
            action_probs = torch.ones(self.size_s, self.size_a).to(self.device)
            for i in range(self.n):
                action_probs_i = self.softmax(self.m[i].view(-1,2)) #s * num_parents * 2
                action_probs_i = action_probs_i.view(self.size_s, 2**(len(self.parents[i])), 2)
                shape = [self.size_s]
                repeat_amounts = [1]
                for j in range(self.n):
                    if j in self.parents[i]:
                        shape.append(2)
                        repeat_amounts.append(1)
                    elif j !=i:
                        shape.append(1)
                        repeat_amounts.append(2)
                    else:
                        continue
                shape.append(2)
                repeat_amounts.append(1)
                action_probs_i = action_probs_i.view(shape)
                action_probs_i = action_probs_i.repeat(repeat_amounts)
                action_probs_i = action_probs_i.view(self.size_s, 2**i, 2**(self.n-i-1), 2)
                action_probs_i = action_probs_i.permute((0,1,3,2))
                action_probs_i = action_probs_i.reshape(self.size_s, self.size_a)
                action_probs = action_probs * action_probs_i
            return action_probs
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

class CoordinationGame():
    """
    Multi-Agent Coordination Game with Bayesian Network Policies
    
    Implements a tabular coordination game where agents learn policies using
    exact policy gradient methods. The game features:
    - Binary state space per agent (2^n total states)
    - Binary action space per agent (2^n joint actions)
    - Reward based on state balance (number of 0s vs 1s)
    - Stochastic state transitions with epsilon noise
    
    Args:
        n: Number of agents
        epsilon: Transition noise probability
        gamma: Discount factor
        mu: Initial state distribution
        eta: Learning rate
        policy: Policy type (default 'tabular_baysian')
        k: Unused parameter (kept for compatibility)
        lbda: Log barrier coefficient
        optimizer_type: Optimizer ('SGD' or 'Adam')
        device: Computing device
        G_type: Bayesian network graph structure
    """
    def __init__(self, n, eplison,gamma,mu,eta,policy = 'tabular', k = 1, lbda = 10, optimizer_type = 'SGD', device = 'cpu', G_type = 'all_ones'):
        self.n = n
        self.eta = eta
        if policy == 'tabular':
            self.clip_amount = 1e3
        else:
            self.clip_amount = 1e3
        self.device = device
        self.policy = policy
        self.lbda = lbda
        self.optimizer_type = optimizer_type
        self.G_type = G_type

        if self.policy == 'tabular_baysian':
            self.baysian_policy = Net(n, self.policy, G_type = self.G_type, device = self.device).to(self.device)
        else:
            raise NotImplementedError("Not implemented")
        self.parameters = []

        if self.policy == 'tabular_baysian':
            self.parameters = list(self.baysian_policy.parameters())
        else:
            raise NotImplementedError("Not implemented")

        self.optimizer = self.get_optimizer(self.optimizer_type, self.parameters)
        self.k = k
                
        self.eplison = eplison
        self.gamma=gamma
        self.mu = mu.to(self.device)
        self.prob=[[1-eplison,eplison],[eplison,1-eplison]] #p(s|a)
        self.trans_prob={(0,0):1-eplison,(0,1):eplison,(1,0):eplison,(1,1):1-eplison} #p(s|a)
        self.states_probs = None
        self.best_value = None
        self.size_s = 2**n
        self.size_a = 2**n
        self.l={2:1,3:1,5:2,10:3}
        self.reward_table = self.get_reward(self.l[self.n]) #2**n

    def get_reward(self,l):
        """
        Generates reward table based on state balance
        States with balanced 0s and 1s (within threshold l) get rewards 0 or 1
        Imbalanced states get higher rewards (2 or 3)
        """
        reward = torch.zeros(self.size_a)
        for i in range(self.size_a):
            state = np.binary_repr(i, self.n)
            if abs(state.count('0') - state.count('1'))<=l:
                if state.count('0') < state.count('1'):
                    reward[i] = 1
                else:
                    reward[i] = 0
            elif state.count('0') > state.count('1'):
                reward[i] = 3
            else:
                reward[i] = 2
        return reward.to(self.device)
    
    def get_optimizer(self, optimizer_type, parameters):
        if optimizer_type == 'SGD':
            optimizer = optim.SGD(parameters, lr=self.eta)
        elif optimizer_type == 'Adam':
            optimizer = optim.Adam(parameters, lr=self.eta)
        else:
            raise NotImplementedError("Not implemented")
        return optimizer

    
    def softmax(self, x):
        z = x - torch.max(x,dim=1).values.view(len(x),1).repeat(1,len(x[0]))
        numerator = torch.exp(z)
        denominator = torch.sum(numerator,dim=1).view(len(x),1).repeat(1,len(x[0]))
        softmax = numerator/denominator
        return softmax
    
    def array_rearange(self, A, size):
        ret = []
        m,n = A.shape
        l = int(m/size[0])
        p = int(n/size[1])
        for i in range(1,l+1):
            for j in range(1,p+1):
                x = A[(i-1)*size[0]:i*size[0],(j-1)*size[1]:j*size[1]].tolist()
                ret+= A[(i-1)*size[0]:i*size[0],(j-1)*size[1]:j*size[1]].tolist()
        return torch.tensor(ret).to(self.device)

    
    def get_p_pi(self, actions_probs):
        actions_probs = torch.unsqueeze(actions_probs, dim=0).repeat(self.size_a,1,1)
        states_probs = torch.unsqueeze(self.states_probs, dim=-1).repeat(1,1,self.size_a)
        p_pi = states_probs * actions_probs
        p_pi = torch.unsqueeze(p_pi, dim=0).repeat(self.size_s,1,1,1)
        p_pi = p_pi.view(self.size_s * self.size_a, self.size_s * self.size_a)
        return p_pi
    
    def get_p(self, actions_probs, states_probs):
        p = actions_probs @ states_probs
        return p
    
    def get_Q(self, actions_probs):
        """
        Computes Q-values by solving the Bellman equation as a linear system
        Q = (I - γP_π)^(-1) * r
        """
        r = self.reward_table.view(-1,1).repeat(1,self.size_a)
        r = r.view(-1,1)
        p_pi = self.get_p_pi(actions_probs)
        Q = torch.linalg.solve(torch.eye(self.size_s * self.size_a).to(self.device)-self.gamma*p_pi,r)
        return Q.view(self.size_s,self.size_a)
    
    def get_d(self, actions_probs):
        """Computes stationary state distribution under current policy"""
        p = self.get_p(actions_probs, self.states_probs)
        temp1 = torch.linalg.solve((torch.eye(2**self.n).to(self.device)-self.gamma*p).T,self.mu).T
        d = (1-self.gamma)*(temp1).view(-1)
        return d
    
    def get_V(self,Q, actions_probs):
        """Computes state values V(s) = Σ_a π(a|s) Q(s,a)"""
        return (Q * actions_probs).sum(dim=1)
    
    def get_Q_i(self, i, Q, actions_probs): #s*2
        Q_i = 0
        Q_temp = torch.zeros(self.size_s, int(self.size_a/2),2).to(self.device)
        a_temp = torch.zeros(self.size_a, int(self.size_a/2),2).to(self.device)
        actions_except_i_probs = actions_probs.view(self.size_s, 2**i,2,2**(self.n-i-1))
        actions_except_i_probs = actions_except_i_probs.permute((0,2,1,3))
        actions_except_i_probs = torch.sum(actions_except_i_probs,dim=1)
        actions_except_i_probs = actions_except_i_probs.view(self.size_s, 2**(self.n-1))
        for s in range(self.size_s):
            for a_m_i in range(int(self.size_a/2)):
                for a_i in range(2):
                    a_minus_i = np.binary_repr(a_m_i, self.n-1)
                    a = a_minus_i[0:i] + str(a_i) + a_minus_i[i:]
                    Q[s,int(a,2)]
                    Q_temp[s,a_m_i,a_i] = Q[s,int(a,2)]
                    a_temp[s,a_m_i,a_i] = actions_except_i_probs[s,a_m_i]
        Q_i = Q_temp * a_temp
        Q_i = Q_i.sum(dim=1)
        return Q_i, actions_except_i_probs
                                 
    def get_actions_probs(self, theta): #(s*(i,j,k...)*joint action)
        actions_probs = []
        probs_per_states = self.softmax(theta.view(-1,2)).view(self.size_s,-1,2)
        for probs in probs_per_states:
            probs = [prob.view(2,1) for prob in probs]
            actions_probs.append(self.get_joint_probs(probs).view(-1))
        actions_probs = torch.stack(actions_probs)
        return actions_probs, probs_per_states
    
    def get_states_probs(self):
        """
        Computes joint state transition probabilities P(s'|s,a)
        Returns: Tensor of shape (size_a, size_s) representing transitions
        """
        probs = torch.stack([torch.tensor([1-self.eplison,self.eplison,self.eplison,1-self.eplison]).view(4,1) for i in range(self.n)]).to(self.device)
        states_probs = self.get_joint_probs(probs,rearrange=True).view(self.size_a,self.size_s)
        return states_probs.to(self.device)
    
    def get_states_probs_i(self):
        """
        Computes state transition probabilities for each agent i (excluding agent i)
        Used for computing Nash Equilibrium gaps
        """
        states_probs_i = {}
        for i in range(self.n):
            probs = torch.stack([torch.tensor([1-self.eplison,self.eplison,self.eplison,1-self.eplison]).view(4,1) for k in range(self.n) if k!=i]).to(self.device)
            states_probs = self.get_joint_probs(probs,rearrange=True).view(2**(self.n-1), 2**(self.n-1))
            states_probs_i[i] = states_probs.to(self.device)
        return states_probs_i

    def value_iteration(self,P,r,eps = 1e-4):
        """
        Performs value iteration to find optimal value function
        Used for computing best responses in NE gap calculation
        """
        Q = torch.zeros(len(r)).to(self.device)
        next_Q = None
        V_Q = 0
        prev_V_Q = 0
        while True:
            prev_V_Q = V_Q
            V_Q = torch.max(Q.view(self.size_s,-1), dim=1).values
            V_Q = V_Q.view(-1,1)
            next_Q = r + self.gamma * P @ V_Q
            if torch.norm(next_Q - Q) <= eps:
                break
            Q = next_Q
        Q = Q.view(self.size_s, -1)
        x = torch.max(Q,dim=1)
        V_max = x.values
        return V_max
    
    def value_iteration_global(self,P,r,eps = 1e-4):
        """
        Performs global value iteration for computing optimal policy value
        Used to calculate Price of Anarchy (PoA)
        """
        Q = torch.zeros(len(r)).to(self.device)
        next_Q = None
        V_Q = 0
        prev_V_Q = 0
        while True:
            prev_V_Q = V_Q
            V_Q = torch.max(Q.view(self.size_s,-1), dim=1).values
            V_Q = V_Q.view(-1,1)
            PV_Q= P @ V_Q
            PV_Q=PV_Q.view(1,-1,1).repeat(self.size_s,1,1)
            PV_Q = PV_Q.view(-1,1)
            next_Q = r + self.gamma * PV_Q
            if torch.norm(next_Q - Q) <= eps:
                break
            Q = next_Q
        Q = Q.view(self.size_s, -1)
        x = torch.max(Q,dim=1)
        V_max = x.values
        return V_max
            
    def get_NE_Gap_i(self, i, actions_probs):
        """
        Computes Nash Equilibrium gap for agent i
        
        The NE gap measures how much agent i can improve by best responding
        while other agents maintain their current policies.
        
        Returns: Maximum improvement in value agent i can achieve
        """
        
        size = 2**(self.n-1)              #s*a
        #for P(s,a_i)
        action_probs_i = self.baysian_policy.helper(i)+1e-20

        actions_probs = actions_probs/action_probs_i
        actions_probs = actions_probs.view(self.size_s, 2**i,2,2**(self.n-i-1))
        actions_probs = actions_probs.permute((0,2,1,3))
        P = actions_probs.reshape(self.size_s, 2, -1)

        actions_probs = actions_probs.reshape(self.size_s, 2, -1,1)
        actions_probs = actions_probs.repeat(1,1,1,size)
        states_probs_i = self.states_probs_i[i]
        states_probs_i = states_probs_i.view(1,1,size,size).repeat(self.size_s,2,1,1)
        P = (states_probs_i * actions_probs).sum(dim=2)
        P = P.view(self.size_s,2,2**i,1,2**(self.n-i-1))
        P = P.repeat(1,1,1,2,1)

        p_ai = torch.tensor([1-self.eplison,self.eplison,self.eplison,1-self.eplison]).view(1,2,1,2,1)
        p_ai = p_ai.repeat(self.size_s, 1, 2**i, 1,2**(self.n-i-1)).to(self.device)
        P = P*p_ai
        P = P.view(self.size_s,2,self.size_s)
        P = P.view(-1,self.size_s)
        
        r = self.reward_table.view(self.size_s,1).repeat(1,2).view(-1,1) #r(s,a_i) ???
        
        V_max = self.value_iteration(P,r)
        NE_Gap_i = V_max.T@self.mu
        return NE_Gap_i

    def get_joint_probs(self, probs, rearrange = False):
        joint_probs = probs[0]
        for i, prob in enumerate(probs[1:]):
            joint_probs = joint_probs @ prob.T
            if rearrange:
                joint_probs = self.array_rearange(joint_probs, (2**(i+1),2))
            joint_probs = joint_probs.view(-1,1)
        return joint_probs

    def get_log_barrier_loss(self, probs_per_states):
        log_barrier_loss = self.lbda * torch.log(1e-10+probs_per_states)
        log_barrier_loss = log_barrier_loss.view(-1).sum()
        return log_barrier_loss/self.size_s/2
  
    def update_policy(self):
        """
        Main policy update method using exact policy gradient
        
        Workflow:
        1. Initialize state transition probabilities (first call only)
        2. Compute optimal value for Price of Anarchy (first call only)
        3. Get current action probabilities from Bayesian policy
        4. Compute Q-values using Bellman equation
        5. Compute state values V(s)
        6. Compute objective J = μ^T V
        7. Compute stationary state distribution d
        8. Calculate Nash Equilibrium gap for each agent
        9. Compute policy gradient loss: -1/(1-γ) * d * Q * π
        10. Backpropagate and update parameters
        
        Returns:
            J: Current policy value
            PoA: Price of Anarchy (ratio of current to optimal value)
            NE_Gap: Nash Equilibrium gap (maximum improvement potential)
        """
        if self.states_probs == None:
            self.states_probs = self.get_states_probs()
            self.states_probs_i = self.get_states_probs_i()
            
        if self.best_value == None:
            P = self.states_probs.view(self.size_a, self.size_s)
            r = self.reward_table.view(-1,1).repeat(1,self.size_a)
            r = r.view(-1,1)
            self.best_value = (self.value_iteration_global(P,r)).T @ self.mu
        
        self.optimizer.zero_grad()
        
        actions_probs = self.baysian_policy([None])
        Q = self.get_Q(actions_probs)
        V = self.get_V(Q, actions_probs)
        
        J = V.T@self.mu
        
        d = self.get_d(actions_probs)
        
        NE_Gap = -float('inf')
        for i in range(self.n):
            NE_Gap = max(NE_Gap, self.get_NE_Gap_i(i, actions_probs.clone()))
        

        d = d.view(self.size_s, 1).repeat(1, self.size_a).detach()
        Q = Q.detach()
        
        loss = -1/(1-self.gamma)*d*Q*actions_probs
        loss = torch.sum(loss)

        loss.backward()
        unclipped_norm = clip_grad_norm_(self.parameters, self.clip_amount)
        self.optimizer.step()
       
        NE_Gap -= J
        return J.detach(), (J/self.best_value).detach(), NE_Gap.detach()
