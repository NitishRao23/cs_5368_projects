from game import Directions, Actions
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], "point1 is a wall: " + str(point1)
    assert not walls[x2][y2], "point2 is a wall: " + str(point2)
    prob = PositionSearchProblem(
        gameState, start=point1, goal=point2, warn=False, visualize=False
    )
    return len(breadthFirstSearch(prob))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    bfs=BFS()
    return (bfs.breadthFirstSearch(problem))

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(
        self,
        gameState,
        costFn=lambda x: 1,
        goal=(1, 1),
        start=None,
        warn=True,
        visualize=True,
    ):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None:
            self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print("Warning: this does not look like a regular search maze")

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__

            if "_display" in dir(__main__):
                if "drawExpandedCells" in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(
                        self._visitedlist
                    )  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [
            Directions.NORTH,
            Directions.SOUTH,
            Directions.EAST,
            Directions.WEST,
        ]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None:
            return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += self.costFn((x, y))
        return cost

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

class BFS(object):
    def breadthFirstSearch(self, problem):
        "*** TTU CS5368 YOUR CODE HERE ***"

        startNode = problem.getStartState()

        # use queue to keep track of nodes
        nodeQueue = Queue()

        # use queue to keep track of actions as per nodes
        actionsQueue = Queue()

        # add initial start node to the nodeQueue and add empty action
        nodeQueue.push(startNode)
        actionsQueue.push([])

        # List to keep track of visited nodes
        visitedNodes = []

        # loop till list nodequeue is not empty
        while len(nodeQueue.list) != 0:
            currentNode = nodeQueue.pop()
            currentActions = actionsQueue.pop()

            # check for goal node and return steps
            if problem.isGoalState(currentNode):
                return currentActions

            # if currentNode is not visited,
            # find its succesors and add them to queue
            # and add it to visitedlist

            if currentNode not in visitedNodes:

                currentNodeSuccesors = problem.getSuccessors(currentNode)
                for node in currentNodeSuccesors:
                    if node[0] not in visitedNodes:

                        nodePosition = node[0]
                        nodeActions = node[1]

                        nodeQueue.push(nodePosition)

                        newActions = currentActions + [nodeActions]
                        actionsQueue.push(newActions)

                visitedNodes.append(currentNode)

        return []


