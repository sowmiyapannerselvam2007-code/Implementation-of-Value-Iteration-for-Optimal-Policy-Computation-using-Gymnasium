# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement

To develop a Value Iteration agent using the Gymnasium FrozenLake environment to determine the optimal policy. The objective is to
maximize the expected cumulative reward by reaching the goal while avoiding holes in the grid




## Software Requirements
Python 3.x
Gymnasium
NumPy
Matplotlib
Jupyter Notebook / Google Colab / VS Code

## Environment Description
The FrozenLake environment is a grid-based problem where the agent starts at the Start (S) state and aims to reach the Goal (G) while
avoiding Holes (H). Safe tiles are represented by F. The environment is stochastic (is_slippery=True), meaning the agent may not always
move in the intended direction.

## MDP Representation

States (S): 16 states in a 4×4 grid.

Actions (A): Left, Down, Right, Up.

Rewards (R): +1 for reaching the goal and 0 for other transitions.

Transition Probability (P): Probability of moving from one state to another.

Discount Factor (γ): 0.99.
## Theory


Value Iteration is a Dynamic Programming algorithm used to solve Markov Decision Processes (MDP). It repeatedly updates the value of
each state using the Bellman Optimality Equation until the values converge.
The algorithm calculates the maximum expected future reward for each state by considering all possible actions. After convergence, the
optimal policy is obtained by selecting the action with the highest expected value for each state.
Bellman Optimality Equation:
V(s) = maxₐ Σ P(s'|s,a) [R(s,a,s') + γV(s')]

## Algorithm


1. Initialize all state values to zero.
2. Set the discount factor γ = 0.99.
3. Calculate the expected value of all possible actions for each state.
4. Update each state value using the maximum action value.
5. Repeat the process until the maximum change in state values is less than the threshold.
6. Extract the optimal policy by selecting the action with the highest expected value.
7. Display the number of iterations, optimal state-value function and optimal policy.


## Python Program
```
# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs value iteration and returns the optimal value function.
    """

def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs value iteration and returns the optimal value function.
    """
    P = env.unwrapped.P
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    iteration = 0

    while True:
        delta = 0

        for s in range(n_states):
            old_value = V[s]
            action_values = []

            for a in range(n_actions):
                value = 0

                for prob, next_state, reward, terminated in P[s][a]:
                    value += prob * (
                        reward + gamma * V[next_state]
                    )

                action_values.append(value)

            V[s] = max(action_values)
            delta = max(delta, abs(old_value - V[s]))

        iteration += 1

        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        action_values = []

        for a in range(n_actions):
            value = 0

            for prob, next_state, reward, terminated in P[s][a]:
                value += prob * (
                    reward + gamma * V[next_state]
                )

            action_values.append(value)

        policy[s] = np.argmax(action_values)

    return V, policy, iteration
```


## Output

<img width="335" height="287" alt="Screenshot 2026-11-18 174122" src="https://github.com/user-attachments/assets/1ba2e49c-ff6e-4b6c-bdee-b733eb24019b" />
<img width="348" height="347" alt="Screenshot 2026-11-18 174128" src="https://github.com/user-attachments/assets/19df4129-c9d8-4224-ac04-262512a31f19" />


## Result
Thus, the Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake environment. The optimal state-value
function and optimal policy were computed using the Bellman Optimality Equation to determine the best actions for reaching the goal
while avoiding hole

## Inference
The Value Iteration algorithm repeatedly updates the state values until convergence. The optimal policy is obtained by selecting the action
with the maximum expected value for each state. The algorithm helps the agent identify the best possible path to reach the goal while
avoiding holes in the FrozenLake environment.




