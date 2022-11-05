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

import random
import util
import math


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

        "*** CS5368 YOUR CODE HERE ***"

        self.values = util.Counter()  # A Counter is a dict with default 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** CS5368 YOUR CODE HERE ***"

        '''if no state is found in values return 0 else return the value'''
        if (state, action) not in self.values:
            return 0.0
        else:
            return self.values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** CS5368 YOUR CODE HERE ***"

        '''get list of all legal actions'''
        actions = self.getLegalActions(state)

        '''if not legal actions found return 0.0
        which is the case for terminal state
        '''
        if not actions:
            return 0.0

        '''maxValue to keep track of max q value among all legal states'''
        maxValue = -math.inf

        '''for each legal actions compute q value, check if it maximum'''
        for action in actions:
            value = self.getQValue(state, action)
            maxValue = max(value, maxValue)
        return maxValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** CS5368 YOUR CODE HERE ***"

        '''get list of all legal actions'''
        actions = self.getLegalActions(state)

        '''if not legal actions found return None'''
        if not actions:
            return None

        '''initialize variables to keep track of max value and best action'''
        bestAction = None
        maxValue = -math.inf

        '''for each legal actions compute q value, check if it maximum
        if maximum set that action as best action
        '''
        for action in actions:
            value = self.getQValue(state, action)
            if value > maxValue:
                maxValue = value
                bestAction = action

        return bestAction

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
        "*** CS5368 YOUR CODE HERE ***"

        '''if not legalAction is found return None'''
        if not legalActions:
            return None

        '''use flipcoin function for randomness'''
        '''for question 6'''
        prob = util.flipCoin(self.epsilon)

        '''if true, make a random choice of action
        else return best action from the state'''

        if (prob):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** CS5368 YOUR CODE HERE ***"

        '''get old q value'''
        oldValue = self.getQValue(state, action)

        '''compute max q value for next state'''
        newQValue = self.computeValueFromQValues(nextState)

        '''compute sample'''
        sample = reward + self.discount * newQValue

        '''update the value using the Q based on sample formula'''
        self.values[(state, action)] = (1 - self.alpha) * \
            oldValue + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
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
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
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
        "*** CS5368 YOUR CODE HERE ***"

        '''get all features'''
        features = self.featExtractor.getFeatures(state, action)

        '''initalize qValue to 0.0'''
        qValue = 0.0

        '''for each feature compute cumulative qValue and return'''
        for f in features:
            qValue = qValue + features[f] * self.weights[f]

        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** CS5368 YOUR CODE HERE ***"

        '''get old value'''
        oldValue = self.getQValue(state, action)

        '''get all features'''
        features = self.featExtractor.getFeatures(state, action)

        '''calculate maxQvalue'''
        maxQvalue = self.computeValueFromQValues(nextState)

        '''calculate newValue'''
        newValue = reward + self.discount * maxQvalue

        '''calculate difference'''
        difference = newValue - oldValue

        '''go through each feature and update weight'''
        for f in features:
            self.weights[f] += self.alpha * difference * features[f]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** CS5368 YOUR CODE HERE ***"
            pass
