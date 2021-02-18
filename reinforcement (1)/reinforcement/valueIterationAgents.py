# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Loop through the number of iterations
        for x in range(0,self.iterations):
            # loop through every state
            Tvals = self.values.copy()
            for state in mdp.getStates():
                # get all possible actions of a state
                listnodes = []
                # Loop through the set of all possible actions from state
                for action in mdp.getPossibleActions(state):
                    wt_avg = 0
                    # Loop through all the successor states
                    for nextState in mdp.getTransitionStatesAndProbs(state, action):
                        t = Tvals[nextState[0]]
                        # calculate the weighted average
                        sample_update = (nextState[1] * ((t * discount) + mdp.getReward(state, action, nextState[0])))
                        wt_avg = wt_avg + sample_update
                    # add the weighted average to the list
                    listnodes.append(wt_avg)
                # check if length of nodes is greater than 0
                if len(listnodes) > 0:
                    # find the maximum in the list
                    self.values[state] = max(listnodes)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q_val = 0
        for nextState in self.mdp.getTransitionStatesAndProbs(state, action):
            disc = self.discount
            t = self.values[nextState[0]]
            sample_update = (nextState[1] * ((t * disc) + self.mdp.getReward(state, action, nextState[0])))
            Q_val = Q_val + sample_update
        return Q_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_val = float("-inf")
        best_action = ""
        if self.mdp.isTerminal(state):
            return None
        # Loop through all the actions in the list of possible actions from the state
        for action in self.mdp.getPossibleActions(state):
            # get the Q-values of the state and action
            wt_avg = self.computeQValueFromValues(state, action)
            # update the best action and max value
            if wt_avg >= max_val:
                max_val = wt_avg
                # update the best action
                best_action = action
            # update the best action and max value
            elif action == "" and max_val == 0.0:
                max_val = wt_avg
                # update the best action
                best_action = action
        # return the best action in the state
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
