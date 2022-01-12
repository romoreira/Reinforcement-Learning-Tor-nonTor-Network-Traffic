"""
DQN in PyTorch
https://gym.openai.com/evaluations/eval_onwKGm96QkO9tJwdX7L0Gw/

python3 teste.py --gamma 0.99 --env "gym_basic:basic-v0" --n-episode 200 --batch-size 64 --hidden-dim 12 --capacity 50000 --max-episode 50 --min-eps 0.01

"""
import argparse
import torch
import torch.nn
import numpy as np
import random
import gym
from collections import namedtuple
from collections import deque
from typing import List, Tuple
import matplotlib
import matplotlib.pyplot as plt
from csv import writer

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
episode_durations = []

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gamma",
                    type=float,
                    default=0.99,
                    help="Discount rate for Q_target")
parser.add_argument("--env",
                    type=str,
                    default="CartPole-v0",
                    help="Gym environment name")
parser.add_argument("--n-episode",
                    type=int,
                    default=1000,
                    help="Number of epsidoes to run")
parser.add_argument("--batch-size",
                    type=int,
                    default=64,
                    help="Mini-batch size")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=12,
                    help="Hidden dimension")
parser.add_argument("--capacity",
                    type=int,
                    default=50000,
                    help="Replay memory capacity")
parser.add_argument("--max-episode",
                    type=int,
                    default=50,
                    help="e-Greedy target episode (eps will be the lowest at this episode)")
parser.add_argument("--min-eps",
                    type=float,
                    default=0.01,
                    help="Min epsilon")
FLAGS = parser.parse_args()


class DQN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        super(DQN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)

        return x


Transition = namedtuple("Transition",
                        field_names=["state", "action", "reward", "next_state", "done"])


class ReplayMemory(object):

    def __init__(self, capacity: int) -> None:
        """Replay memory class
        Args:
            capacity (int): Max size of this memory
        """
        self.capacity = capacity
        self.cursor = 0
        self.memory = []

    def push(self,
             state: np.ndarray,
             action: int,
             reward: int,
             next_state: np.ndarray,
             done: bool) -> None:
        """Creates `Transition` and insert
        Args:
            state (np.ndarray): 1-D tensor of shape (input_dim,)
            action (int): action index (0 <= action < output_dim)
            reward (int): reward value
            next_state (np.ndarray): 1-D tensor of shape (input_dim,)
            done (bool): whether this state was last step
        """
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = Transition(state,
                                              action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size: int) -> List[Transition]:
        """Returns a minibatch of `Transition` randomly
        Args:
            batch_size (int): Size of mini-bach
        Returns:
            List[Transition]: Minibatch of `Transition`
        """
        return random.sample(self.memory, batch_size)
    
    def gather_all(self) -> List[Transition]:
        """Returns all the items in buffer."""
        return self.memory

    def __len__(self) -> int:
        """Returns the length """
        return len(self.memory)


class Agent(object):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """Agent class that choose action and train
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
        """
        self.dqn = DQN(input_dim, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters())

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))

    def get_action(self, states: np.ndarray, eps: float) -> int:
        """Returns an action
        Args:
            states (np.ndarray): 2-D tensor of shape (n, input_dim)
            eps (float): 𝜺-greedy for exploration
        Returns:
            int: action index
        """
        if np.random.rand() < eps:
            #print("Retornando np.random.rand()")
            #print(str("np.random.rand() < eps: "+str(np.random.choice(self.output_dim))))
            return np.random.choice(self.output_dim)
        else:
            print("Retornando argmax.numpy()")
            self.dqn.train(mode=False)
            #print("States: "+str(np.array([states])))
            scores = self.get_Q(np.array([states]))
            _, argmax = torch.max(scores.data, 1)
            #print("Action returned by get_action - get_Q "+str(argmax.numpy()))
            return int(argmax.numpy())

    def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q-value`
        Args:
            states (np.ndarray): 2-D Tensor of shape (n, input_dim)
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        #print("States into get_Q: "+str(states))
        #states = self._to_variable(states.reshape(-1, self.input_dim))
        states = self._to_variable(states.reshape(-1, self.input_dim))
        self.dqn.train(mode=False)
        return self.dqn(states)

    def train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        """Computes `loss` and backpropagation
        Args:
            Q_pred (torch.FloatTensor): Predicted value by the network,
                2-D Tensor of shape(n, output_dim)
            Q_true (torch.FloatTensor): Target value obtained from the game,
                2-D Tensor of shape(n, output_dim)
        Returns:
            float: loss value
        """
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()

        return loss


def train_helper(agent: Agent, minibatch: List[Transition], gamma: float) -> float:
    """Prepare minibatch and train them
    Args:
        agent (Agent): Agent has `train(Q_pred, Q_true)` method
        minibatch (List[Transition]): Minibatch of `Transition`
        gamma (float): Discount rate of Q_target
    Returns:
        float: Loss value
    """
    states = np.vstack([x.state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    next_states = np.vstack([x.next_state for x in minibatch])
    done = np.array([x.done for x in minibatch])

    Q_predict = agent.get_Q(states)
    Q_target = Q_predict.clone().data.numpy()
    Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * np.max(agent.get_Q(next_states).data.numpy(), axis=1) * ~done
    Q_target = agent._to_variable(Q_target)
    #print("q_predict: "+str(Q_predict)+" Q_target: "+str(Q_target))
    return agent.train(Q_predict, Q_target)


def play_episode(env: gym.Env,
                 agent: Agent,
                 replay_memory: ReplayMemory,
                 eps: float,
                 batch_size: int) -> int:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        replay_memory (ReplayMemory): trajectory is saved here
        eps (float): 𝜺-greedy for exploration
        batch_size (int): batch size
    Returns:
        int: reward earned in this episode
    """
    s = env.reset()
    done = False
    total_reward = 0

    while not done:

        a = agent.get_action(s, eps)
        #print("Play Episode retorno de agent.get_action(): "+str(a))
        s2, r, done, info = env.step(a)

        total_reward += r

        if done:
            r = -1
        replay_memory.push(s, a, r, s2, done)

        #print("Cheking if replay_memory > batch_size: "+str(len(replay_memory))+ " > "+str(batch_size))
        if len(replay_memory) > batch_size:
            #print("train_helper calling")
            minibatch = replay_memory.pop(batch_size)
            loss = train_helper(agent, minibatch, FLAGS.gamma)
            #print("LOSS: "+str(loss))

        s = s2

    return total_reward, loss


def get_env_dim(env: gym.Env) -> Tuple[int, int]:
    """Returns input_dim & output_dim
    Args:
        env (gym.Env): gym Environment (CartPole-v0)
    Returns:
        int: input_dim
        int: output_dim
    """
    print("Observation_space: "+str(env.observation_space.shape[0]))
    print("Action_space: "+str(env.action_space.n))
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    return input_dim, output_dim


def epsilon_annealing(epsiode: int, max_episode: int, min_eps: float) -> float:
    """Returns 𝜺-greedy
    1.0---|\
          | \
          |  \
    min_e +---+------->
              |
              max_episode
    Args:
        epsiode (int): Current episode (0<= episode)
        max_episode (int): After max episode, 𝜺 will be `min_eps`
        min_eps (float): 𝜺 will never go below this value
    Returns:
        float: 𝜺 value
    """

    slope = (min_eps - 1.0) / max_episode
    return max(slope * epsiode + 1.0, min_eps)

def csv_writter(register):
    with open('adaptative_pooling_results.csv', 'a') as f:
        writer_object = writer(f)
        writer_object.writerow(register)
        f.close()

def csv_creator():
    List_Exp = ['episode_number', 'train_reward', 'train_loss', 'rewards_avg']
    with open('adaptative_pooling_results.csv', 'w') as f:
        writer_object = writer(f)
        writer_object.writerow(List_Exp)
        f.close()

def main():
    """Main
    """
    csv_creator()

    try:
        env = gym.make(FLAGS.env)
#        env = gym.wrappers.Monitor(env, directory="monitors", force=True)
        rewards = deque(maxlen=10)
        input_dim, output_dim = get_env_dim(env)
        agent = Agent(input_dim, output_dim, FLAGS.hidden_dim)
        replay_memory = ReplayMemory(FLAGS.capacity)

        loss_history = []
        episodes = []
        registro_csv = []
        recompensas = []
        
        for i in range(FLAGS.n_episode):
            eps = epsilon_annealing(i, FLAGS.max_episode, FLAGS.min_eps)
            r, loss = play_episode(env, agent, replay_memory, eps, FLAGS.batch_size)
            print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f}".format(i + 1, r, eps))

            rewards.append(r)
            loss_history.append(loss.item())
            episodes.append(i)
            recompensas.append(r)

            registro_csv.append(str(i))
            registro_csv.append(r)
            registro_csv.append(loss.item())
            registro_csv.append(np.mean(rewards))
            csv_writter(registro_csv)
            registro_csv = []

            print("Media Rewards: "+str(np.mean(rewards)))
            if len(rewards) == rewards.maxlen:
                if np.mean(rewards) >= 4.9:
                    print("Game cleared in {} games with {}".format(i + 1, np.mean(rewards)))
                    break
        
        print("Rplay_memory: "+str(max(replay_memory.pop(1))))
    finally:
        env.close()
   
    print("Episodes number: "+str(len(episodes)))
    print("Full loss List: "+str(len(loss_history)))
    print("Rewards: "+str(len(rewards)))

    transitions = replay_memory.gather_all()
    print("Replay Memory: "+str(transitions[-100:]))

    plt.plot(episodes, loss_history, 'r', label='Training Loss')
    plt.title('Training Loss and Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.savefig('train_loss.pdf')

    plt.clf()

    plt.plot(episodes, recompensas, 'b', label='Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('train_rewards.pdf')

if __name__ == '__main__':
    main()
