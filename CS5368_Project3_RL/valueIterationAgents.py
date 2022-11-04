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


import mdp
import util
import math

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0

        states = mdp.getStates()

        # Write value iteration code here
        "*** CS5368 YOUR CODE HERE ***"

        '''run the for loop till provided number of iterations'''
        for i in range(self.iterations):

            '''temporary values for each iteration'''
            new_values = util.Counter()

            '''for all states in mdp states'''
            for state in states:

                '''get best action'''
                action = self.computeActionFromValues(state)

                '''if best action is found i.e. if the state is not terminal state'''
                if action:
                    '''calculate qvalue and keep it in temp variable'''
                    qValue = self.computeQValueFromValues(state, action)
                    new_values[state] = qValue

            '''assign temp values to values'''
            self.values = new_values

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
        "*** CS5368 YOUR CODE HERE ***"

        '''initialize qvalue'''
        qValue = 0.0

        '''get transtition states and probability'''
        for trans, prob in self.mdp.getTransitionStatesAndProbs(state, action):

            '''for each transistions states calculate qvalue and add up'''

            '''calcualte reward and discount'''
            reward = self.mdp.getReward(state, action, trans)
            dis = self.discount * self.getValue(trans)

            '''calculate total qvalue'''
            qValue = qValue + prob * (reward + dis)

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** CS5368 YOUR CODE HERE ***"

        '''initialize bestaction and maxvalue'''
        bestAction = None
        maxValue = -math.inf

        '''for each action in possible actions of the provide state'''
        for action in self.mdp.getPossibleActions(state):

            '''calculate qvalue'''
            qValue = self.computeQValueFromValues(state, action)

            '''if qvalue of current state is better than of previous states
            set new bestaction and maxvalue
            '''
            if (qValue > maxValue):
                bestAction = action
                maxValue = qValue

        '''return the bestaction based of max qvalue'''
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
