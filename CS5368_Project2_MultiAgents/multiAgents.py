# multiAgents.py
# --------------
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


from turtle import pen
from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** CS5368 YOUR CODE HERE ***"
        "Decribe your function:"

        """
        The evalucation fucntion takes into account of three factors. 
        1) position of ghost in succesor states
        2) no of current food items remaining
        3) minimum food distance

        At first, the function checks if the pacman is near the ghost in next state.
        If the distance between pacman and ghost is less then 2 meaning that they are 
        next to each other, there is possibility that they will meet in another state
        and the pacman will die.Hence to prevent that from happening, function returns
        negative infinity.

        Second it checks if the pacman eats some food or not. It can be done by comparing
        the number of remaining food items in the current state and successors states. It
        is good thing that food is being eaten hence return postive infinity.

        There may come situation when there is no ghost nearby and no of remaning food does
        not decrease in next states. In such case the pacman will never make progress.
        To prevent this, the function calculates the minimum distance to closet food and 
        returns the reciprocal of it as hinted in the assignment pdf. 

        """

        # check if succesor state has ghost nearby
        for ghostState in newGhostStates:
            ghostDistance = manhattanDistance(ghostState.getPosition(), newPos)
            if ghostDistance < 2:
                return -math.inf

        # check if it can eat food or not
        currentNoOfRemFoods = len(currentGameState.getFood().asList())
        succesorNoOfRemFoods = len(successorGameState.getFood().asList())
        if succesorNoOfRemFoods < currentNoOfRemFoods:
            return math.inf

        # min distance between food and pacman postion
        minDistance = math.inf
        for food in newFood.asList():
            distance = manhattanDistance(food, newPos)
            minDistance = min(minDistance, distance)
        return 1.0 / minDistance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** CS5368 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"

        """
        initializing values:    
            agentIndex = 0 means agent is pacman
        """
        agentIndex = 0
        bestScore = -math.inf
        bestMove = Directions.STOP

        """
        for each actions in next game state
        """
        for action in gameState.getLegalActions(agentIndex):

            """
            get next state detail
            """
            nextState = gameState.generateSuccessor(agentIndex, action)

            """
            get minimum evaluation score since currently it's pacman turn to chose and in 
            next state it is turn of ghost which minimizes the value. here the modulus of 
            agent index is taken to reset the index of agent (ghost in this case)
            """

            score = self.min_value(
                nextState, self.depth, ((agentIndex + 1) % gameState.getNumAgents())
            )

            """
            if it is the best score, set the varibales and return best move
            """

            if score > bestScore:
                bestMove = action
                bestScore = score
        return bestMove

    """
    function for minimum value i.e in case of ghosts
    we need extra parameter here agentIndex cause there could be
    multiple number of ghost and we have to address that while 
    selecting the next state
    """

    def min_value(self, gameState, depth, agentIndex):

        """
        check the state of the game and depth
        if the game is won or lost by the pacman and if the depth
        for searching is 0 return the evaluation value
        """

        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        """
        initialize score to infinity
        """
        score = math.inf

        """
        first check if the agent is the last ghost or not
        """

        if agentIndex == (gameState.getNumAgents() - 1):

            """
            if it is the last ghost, then, in next state it is turn of pacman to
            select value i.e select maximum value
            """

            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                score = min(score, self.max_value(nextState, depth - 1))

        else:

            """
            if it is not the last ghost, then, in next state it is turn of another ghost to
            select value i.e select minimum value within same depth for agentIndex + 1 ghost
            """

            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                score = min(
                    score,
                    self.min_value(
                        nextState,
                        depth,
                        ((agentIndex + 1) % gameState.getNumAgents()),
                    ),
                )

        return score

    """
    function for maximum value i.e when pacman is selecting the state
    """

    def max_value(self, gameState, depth):

        """
        check the state of the game and depth
        if the game is won or lost by the pacman and if the depth
        for searching is 0 return the evaluation value
        """

        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        """
        initialize score to negative infinity since we are trying to maximize value
        """

        score = -math.inf

        """
        since only pacman tries to maximize the output no need to check the type of agent
        just return the maximum score from succesors state
        """
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            score = max(score, self.min_value(nextState, depth, 1))

        return score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** CS5368 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"

        """
        Alphabeta agent will be similar to minmax agent witha dditional feature of alpha and beta and purning
        """

        """
        initializing values:    
            agentIndex = 0 means agent is pacman

            alpha and beta are also initialized here
        """
        agentIndex = 0
        bestScore = -math.inf
        bestMove = Directions.STOP
        alpha = -math.inf
        beta = math.inf

        """
        for each actions in next game state
        """
        for action in gameState.getLegalActions(agentIndex):

            """
            get next state detail
            """
            nextState = gameState.generateSuccessor(agentIndex, action)

            """
            get minimum evaluation score since currently it's pacman turn to chose and in 
            next state it is turn of ghost which minimizes the value. here the modulus of 
            agent index is taken to reset the index of agent (ghost in this case)
            """

            score = self.min_value(
                nextState,
                self.depth,
                ((agentIndex + 1) % gameState.getNumAgents()),
                alpha,
                beta,
            )

            """
            if it is the best score, set the varibales and return best move
            """

            if score > bestScore:
                bestMove = action
                bestScore = score

            """
            check for alpha and beta variables here
            since it is maximizing node, set alpha value
            """

            if score > beta:
                return score

            alpha = max(score, alpha)

        return bestMove

    """
    function for minimum value i.e in case of ghosts
    we need extra parameter here agentIndex cause there could be
    multiple number of ghost and we have to address that while 
    selecting the next state

    extra parameters aplha and beta are added
    """

    def min_value(self, gameState, depth, agentIndex, alpha, beta):

        """
        check the state of the game and depth
        if the game is won or lost by the pacman and if the depth
        for searching is 0 return the evaluation value
        """

        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        """
        initialize score to infinity
        """
        score = math.inf

        """
        first check if the agent is the last ghost or not
        """

        if agentIndex == (gameState.getNumAgents() - 1):

            """
            if it is the last ghost, then, in next state it is turn of pacman to
            select value i.e select maximum value
            """

            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                score = min(score, self.max_value(nextState, depth - 1, alpha, beta))

                """
                check for alpha and beta variables here
                """

                if alpha > score:
                    return score

                beta = min(score, beta)

        else:

            """
            if it is not the last ghost, then, in next state it is turn of another ghost to
            select value i.e select minimum value within same depth for agentIndex + 1 ghost
            """

            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.generateSuccessor(agentIndex, action)
                score = min(
                    score,
                    self.min_value(
                        nextState,
                        depth,
                        ((agentIndex + 1) % gameState.getNumAgents()),
                        alpha,
                        beta,
                    ),
                )

                """
                check for alpha and beta variables here
                since it is minimizing node, set beta value
                """

                if alpha > score:
                    return score

                beta = min(score, beta)

        return score

    """
    function for maximum value i.e when pacman is selecting the state
    """

    def max_value(self, gameState, depth, alpha, beta):

        """
        check the state of the game and depth
        if the game is won or lost by the pacman and if the depth
        for searching is 0 return the evaluation value
        """

        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        """
        initialize score to negative infinity since we are trying to maximize value
        """

        score = -math.inf

        """
        since only pacman tries to maximize the output no need to check the type of agent
        just return the maximum score from succesors state
        """
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            score = max(score, self.min_value(nextState, depth, 1, alpha, beta))

            """
            check for alpha and beta variables here
            since it is maximizing node, set alpha value
            """

            if beta < score:
                return score

            alpha = max(score, alpha)

        return score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** CS5368 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** CS5368 YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
