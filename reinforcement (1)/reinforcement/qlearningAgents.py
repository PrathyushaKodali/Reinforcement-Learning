# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()    # set the counter

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        Q_node_val = self.values[(state, action)]
        return Q_node_val


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # if no legal actions, then it is a terminal state
        if len(self.getLegalActions(state)) == 0:
            return 0.0      # return 0.0 if it is a terminal state
        # Set the initial max Q value to -infinity
        max_Q_val = float("-inf")
        # loop through every action in legal actions
        for action in self.getLegalActions(state):
            # get the Q value for state action pair
            curr_Q_val = self.getQValue(state, action)
            # if max Q Value is -infinity
            if max_Q_val == float("-inf"):
                max_Q_val = curr_Q_val
            # else if current Q value is greater than max, then update max val
            elif curr_Q_val > max_Q_val:
                max_Q_val = curr_Q_val
        return max_Q_val

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # if no legal actions, then it is a terminal state
        if len(self.getLegalActions(state)) == 0:
            return None             # return None if it is a terminal state
        # Set the initial max Q value to -infinity
        max_Q_val = float("-inf")
        best_action = ""
        # loop through every action in legal actions
        for action in self.getLegalActions(state):
            # get the Q value for state action pair
            curr_Q_val = self.getQValue(state, action)
            # if max Q Value is -infinity, update the best action
            if max_Q_val == float("-inf"):
                best_action = action #update best action
                max_Q_val = curr_Q_val  #update Q val
            # else if current Q value is greater than max, then update max val
            elif curr_Q_val > max_Q_val:
                best_action = action  # update best action
                max_Q_val = curr_Q_val  #update Q val
        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # if there are no legal actions, it indicates terminal state
        if len(legalActions) == 0:      #terminal state
            return None                 # choose None action
        # If probability self.epsilon
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)    # choose random action
            return action
        else:   # else if prob is not self epsilon
            # choose the best policy
            action = self.computeActionFromQValues(state)
            return action


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        Computed_Val = self.computeValueFromQValues(nextState)
        dicount_factor = self.discount
        alpha = self.alpha
        Q_val = self.getQValue(state, action)
        #  update the Q-value
        update_1 = Q_val * (1 - alpha)
        update_2 = alpha * (reward + Computed_Val * dicount_factor)
        self.values[(state, action)] = update_1 + update_2

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        q_val = 0
        # Loop through all the features extracted by feature extracted
        for feat in self.featExtractor.getFeatures(state, action):
            features_extracted = self.featExtractor.getFeatures(state, action)
            # update using featureVector * w
            updated_q_val = features_extracted[feat] * self.weights[feat]
            # update the q value
            q_val = q_val + updated_q_val
        # return Q(state,action)
        return q_val

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        alpha = self.alpha
        discount_factor = self.discount
        val = self.getValue(nextState)
        q_val = self.getQValue(state, action)
        # Loop through all the features extracted by feature extracted
        for feat in self.featExtractor.getFeatures(state, action):
            features_extracted = self.featExtractor.getFeatures(state, action)
            # calculate the update in weights
            update_in_weights = alpha * features_extracted[feat] * (((discount_factor * val) + reward) - q_val)
            # Update the weights of features
            self.weights[feat] += update_in_weights

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
