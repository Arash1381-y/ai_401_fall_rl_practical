# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tabular Q-Learner example on Tic Tac Toe.

Two Q-Learning agents are trained by playing against each other. Then, the game
can be played against the agents from the command line.

After about 10**5 training episodes, the agents reach a good policy: win rate
against random opponents is around 99% for player 0 and 92% for player 1.
"""

import logging
import sys
import os
from absl import app
from absl import flags
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import random_agent

from tabular_qlearner import QLearner

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(6e4), "Number of train episodes.")
flags.DEFINE_boolean(
    "interactive_play",
    True,
    "Whether to run an interactive play with the agent after training.",
)

reward_mask = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 0, 0]])

flat_reward_mask = np.reshape(reward_mask, (9,))


def likeable_pattern(board):
    """Returns a reward if the board is likeable and 0 otherwise."""
    for i in range(9):
        if board[i] != flat_reward_mask[i]:
            return 0

    return 1000


def pretty_board(time_step):
    """Returns the board in `time_step` in a human-readable format."""
    info_state = time_step.observations["info_state"][0]
    x_locations = np.nonzero(info_state[9:18])[0]
    o_locations = np.nonzero(info_state[18:])[0]
    board = np.full(3 * 3, ".")
    board[x_locations] = "X"
    board[o_locations] = "0"
    board = np.reshape(board, (3, 3))
    return board


def command_line_action(time_step):
    """Gets a valid action from the user on the command line."""
    current_player = time_step.observations["current_player"]
    legal_actions = time_step.observations["legal_actions"][current_player]
    action = -1
    while action not in legal_actions:
        print("Choose an action from {}:".format(np.array(legal_actions) + 1))
        sys.stdout.flush()
        action_str = input()
        try:
            action = int(action_str) - 1
        except ValueError:
            continue
    return action


def eval_against_random_bots(
        env, trained_agents, random_agents, num_episodes, show_non_wins=False, top1=False
):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = np.zeros(2)
    losses = np.zeros(2)
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [trained_agents[0], random_agents[1]]
        else:
            cur_agents = [random_agents[0], trained_agents[1]]

        for _ in range(num_episodes):
            time_steps = []
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                opts = dict()
                if player_id == player_pos:
                    opts["top1"] = top1
                agent_output = cur_agents[player_id].step(
                    time_step, is_evaluation=True, **opts
                )
                time_step = env.step([agent_output.action])
                time_steps.append(time_step)

            reward = time_step.rewards[player_pos]
            if reward > 0:
                wins[player_pos] += 1
            elif reward < 0:
                losses[player_pos] += 1
                if show_non_wins and losses[player_pos] <= 4:  #: shows the first four losses
                    logging.info(f"\nnot won: {reward}")
                    for time_step in time_steps:
                        print(
                            f"\nstate:\n{pretty_board(time_step)}\n",
                            file=sys.stderr,
                            # flush=True,
                        )

    print(f"wins: {wins}, losses: {losses}")
    return wins / num_episodes, losses / num_episodes


def main(_):
    game = "tic_tac_toe"
    num_players = 2

    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]

    # create a lambda function which get board as input and if its equal to mask reward return 1000

    agents = [
        QLearner(
            player_id=idx,
            num_actions=num_actions,
            epsilon_schedule=rl_tools.ConstantSchedule(
                0.2,
            ),
            discount_factor=0.6,
            rules=[likeable_pattern]
        )
        for idx in range(num_players)
    ]

    # random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    against_agents_dict = dict(
        random_agents=random_agents,
        qlearning_agents=agents,
    )

    # 1. Train the agents
    training_episodes = FLAGS.num_episodes
    for cur_episode in range(training_episodes):
        last_episode_p = cur_episode == (training_episodes - 1)
        if cur_episode % int(1e4) == 0 or last_episode_p:
            for against_agents_key in against_agents_dict.keys():
                against_agents = against_agents_dict[against_agents_key]
                win_rates, lose_rates = eval_against_random_bots(
                    env,
                    agents,
                    against_agents,
                    1000,
                    show_non_wins=last_episode_p,
                    top1=last_episode_p,
                )
                logging.info(
                    "Starting episode %s, win_rates %s, lose_rates %s against %s", cur_episode, win_rates, lose_rates,
                    against_agents_key,
                )
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

    if not FLAGS.interactive_play:
        return

    # 2. Play from the command line against the trained agent.
    human_player = 1
    while True:
        logging.info("You are playing as %s", "O" if human_player else "X")
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            if player_id == human_player:
                agent_out = agents[human_player].step(time_step, is_evaluation=True)
                logging.info(
                    "\nagent suggests these actions with these probabilities:\n%s",
                    agent_out.probs.reshape((3, 3)),
                )
                logging.info("\n%s", pretty_board(time_step))
                action = command_line_action(time_step)
            else:
                agent_out = agents[1 - human_player].step(time_step, is_evaluation=True)
                action = agent_out.action
            time_step = env.step([action])

        logging.info("\n%s", pretty_board(time_step))

        logging.info("End of game!")
        if time_step.rewards[human_player] > 0:
            logging.info("You win")
        elif time_step.rewards[human_player] < 0:
            logging.info("You lose")
        else:
            logging.info("Draw")
        # Switch order of players
        human_player = 1 - human_player


if __name__ == "__main__":
    app.run(main)
