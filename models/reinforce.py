import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import gym
import numpy as np
from torch.distributions import Categorical
from torch.distributions import Bernoulli
import random
# 定义一个策略网络，使用 ResNet 作为特征提取器
class PolicyNetwork(nn.Module):
    def __init__(self, num_decisions=23):
        super(PolicyNetwork, self).__init__()
        # Use pretrained ResNet18 model
        resnet = models.resnet18(pretrained=True)
        # Remove the last layer of ResNet18
        self.embed = nn.Conv2d(27,64, kernel_size=7,stride=2,padding=3,bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[1:-1])
        # Fully connected layer to map ResNet output to decision probabilities
        self.fc = nn.Linear(resnet.fc.in_features, num_decisions)

    def forward(self, x):
        # Extract features using ResNet
        # with torch.no_grad():
        x = self.feature_extractor(self.embed(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        x = torch.sigmoid(x)  # Use sigmoid to output probabilities for binary decisions
        return x


# 定义REINFORCE算法
class REINFORCE(object):
    def __init__(self, action_size, policy, hidden_size=128, lr=1e-3, gamma=0.99):
        self.policy = policy
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.action_size = action_size

    def select_actions(self, state, eps=5e-2, add_noise=True):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state)
        if add_noise:
            if random.random() < eps:
                noise = torch.normal(mean=0.0, std=1, size=probs.size(),device=probs.device)
                # print(noise)
                noisy_probs = probs + noise  # 加噪声
                probs = torch.clamp(noisy_probs, 0.0, 1.0)
        m = Bernoulli(probs)
        actions = m.sample()
        return actions.int(), m.log_prob(actions)


#     # 计算奖励的折扣因子
# def discount_rewards(rewards, gamma):
#     discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
#     running_sum = 0
#     for t in reversed(range(len(rewards))):
#         running_sum = rewards[t] + gamma * running_sum
#         discounted_rewards[t] = running_sum
#     return discounted_rewards

    # def update_policy(self, rewards, log_probs):
    #     discounted_rewards = []
    #     R = 0
    #     for r in rewards[::-1]: R = r + self.gamma * R
    #     discounted_rewards.insert(0, R)
    #     discounted_rewards = torch.tensor(discounted_rewards)
    #     discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
    #     policy_loss = []
    #     for log_prob, reward in zip(log_probs, discounted_rewards): policy_loss.append(-log_prob * reward)
    #     self.optimizer.zero_grad()
    #     policy_loss = torch.cat(policy_loss).sum()
    #     policy_loss.backward()
    #     self.optimizer.step()





# REINFORCE 算法的训练过程
# def reinforce(policy_net, optimizer, env, num_episodes, gamma):
#     for episode in range(num_episodes):
#         state = env.reset()
#         state = np.transpose(state, (2, 0, 1))  # 从 HWC 到 CHW
#         state = np.expand_dims(state, 0)  # 增加批次维度
#         done = False
#         log_probs = []
#         rewards = []
#
#         while not done:
#             state_tensor = torch.tensor(state, dtype=torch.float32)
#             probs = policy_net(state_tensor)
#             dist = torch.distributions.Categorical(probs)
#             action = dist.sample()
#             log_prob = dist.log_prob(action)
#
#             next_state, reward, done, _ = env.step(action.item())
#             next_state = np.transpose(next_state, (2, 0, 1))
#             next_state = np.expand_dims(next_state, 0)
#
#             log_probs.append(log_prob)
#             rewards.append(reward)
#
#             state = next_state
#
#         # 计算折扣奖励
#         discounted_rewards = discount_rewards(rewards, gamma)
#         discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
#
#         # 计算策略梯度
#         policy_loss = []
#         for log_prob, reward in zip(log_probs, discounted_rewards):
#             policy_loss.append(-log_prob * reward)
#         policy_loss = torch.stack(policy_loss).sum()
#
#         # 更新策略网络
#         optimizer.zero_grad()
#         policy_loss.backward()
#         optimizer.step()


# 示例使用
if __name__ == "__main__":
    import gym
    from torchvision import transforms

    # 设置环境和参数
    env = gym.make('CartPole-v1')
    num_actions = env.action_space.n
    gamma = 0.99
    num_episodes = 1000

    # 初始化策略网络和优化器
    policy_net = PolicyNetwork(num_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    # 训练策略网络
    # reinforce(policy_net, optimizer, env, num_episodes, gamma)

# import gym
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical
# from torchvision import transforms
# import numpy as np
#
#
# # 定义策略网络，使用卷积神经网络处理图像输入
# class PolicyNetwork(nn.Module):
#
#
#     def __init__(self, action_size):
#
#
#     super(PolicyNetwork, self).__init__()
# self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
# self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
# self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
# self.fc1 = nn.Linear(64 * 7 * 7, 512)
# self.fc2 = nn.Linear(512, action_size)
#
#
# def forward(self, x):
#
#
#     x = torch.relu(self.conv1(x))
# x = torch.relu(self.conv2(x))
# x = torch.relu(self.conv3(x))
# x = x.view(x.size(0), -1)
# x = torch.relu(self.fc1(x))
# x = torch.softmax(self.fc2(x), dim=-1)
# return x


