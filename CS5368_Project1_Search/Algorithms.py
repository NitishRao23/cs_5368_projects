import util
import time
from util import Stack, Queue, PriorityQueue


class DFS(object):
    def depthFirstSearch(self, problem):
        """
        Search the deepest nodes in the search tree first
        [2nd Edition: p 75, 3rd Edition: p 87]

        Your search algorithm needs to return a list of actions that reaches
        the goal.  Make sure to implement a graph search algorithm
        [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:

        print "Start:", problem.getStartState()
        print "Is the start a goal?", problem.isGoalState(problem.getStartState())
        print "Start's successors:", problem.getSuccessors(problem.getStartState())
        """
        "*** TTU CS5368 YOUR CODE HERE ***"

        startNode = problem.getStartState()

        # use stack to keep track of nodes
        nodesStack = Stack()

        # use stack to keep track of actions as per nodes
        actionsStack = Stack()

        # add initial start node to the stack and add empty action
        nodesStack.push(startNode)
        actionsStack.push([])

        # List to keep track of visited nodes
        visitedNodes = []

        # loop till list nodestack is not empty
        while len(nodesStack.list) != 0:
            currentNode = nodesStack.pop()
            currentActions = actionsStack.pop()

            # check for goal node and return steps
            if problem.isGoalState(currentNode):
                return currentActions

            # if currentNode is not visited,
            # find its succesors and add them to stack
            # and add it to visitedlist

            if currentNode not in visitedNodes:

                currentNodeSuccesors = problem.getSuccessors(currentNode)
                for node in currentNodeSuccesors:
                    if node[0] not in visitedNodes:

                        nodePosition = node[0]
                        nodeActions = node[1]

                        nodesStack.push(nodePosition)

                        newActions = currentActions + [nodeActions]
                        actionsStack.push(newActions)

                visitedNodes.append(currentNode)

        return []


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


class UCS(object):
    def uniformCostSearch(self, problem):
        "*** TTU CS5368 YOUR CODE HERE ***"

        startNode = problem.getStartState()

        # use priorityqueue to keep track of nodes
        nodeQueue = PriorityQueue()

        # use priorityqueue to keep track of actions as per nodes
        actionsQueue = PriorityQueue()

        # use priorityqueue to keep track of cost till the node
        costQueue = PriorityQueue()

        # add initial start node to the priorityqueue
        # and add empty action adn add initial cost 0
        # to cost queue
        nodeQueue.push(startNode, 0)
        actionsQueue.push([], 0)
        costQueue.push(0, 0)

        # List to keep track of visited nodes
        visitedNodes = []

        # loop till list nodequeue is not empty
        while len(nodeQueue.heap) != 0:
            currentNode = nodeQueue.pop()
            currentActions = actionsQueue.pop()
            currentCost = costQueue.pop()

            # check for goal node and return steps
            if problem.isGoalState(currentNode):
                return currentActions

            # if currentNode is not visited,
            # find its succesors and add them to queue
            # and add it to visitedlist
            # likewise keep track of priority since it's
            # a priority queue

            if currentNode not in visitedNodes:

                visitedNodes.append(currentNode)

                currentNodeSuccesors = problem.getSuccessors(currentNode)
                for node in currentNodeSuccesors:
                    if node[0] not in visitedNodes:

                        nodePosition = node[0]
                        nodeActions = node[1]
                        nodeCost = node[2]

                        priority = currentCost + nodeCost
                        nodeQueue.push(nodePosition, priority)

                        newActions = currentActions + [nodeActions]
                        actionsQueue.push(newActions, priority)
                        costQueue.push(priority, priority)

        return []


class aSearch(object):
    def nullHeuristic(state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0

    def aStarSearch(self, problem, heuristic=nullHeuristic):
        "Search the node that has the lowest combined cost and heuristic first."
        "*** TTU CS5368 YOUR CODE HERE ***"

        startNode = problem.getStartState()

        # use priorityqueue to keep track of nodes
        nodeQueue = PriorityQueue()

        # use priorityqueue to keep track of actions as per nodes
        actionsQueue = PriorityQueue()

        # use priorityqueue to keep track of cost till the node
        costQueue = PriorityQueue()

        # add initial start node to the priorityqueue
        # and add empty action adn add initial cost 0
        # to cost queue
        nodeQueue.push(startNode, 0)
        actionsQueue.push([], 0)
        costQueue.push(0, 0)

        # List to keep track of visited nodes
        visitedNodes = []

        # loop till list nodequeue is not empty
        while len(nodeQueue.heap) != 0:
            currentNode = nodeQueue.pop()
            currentActions = actionsQueue.pop()
            currentCost = costQueue.pop()

            # check for goal node and return steps
            if problem.isGoalState(currentNode):
                return currentActions

            # if currentNode is not visited,
            # find its succesors and add them to queue
            # and add it to visitedlist
            # likewise keep track of priority since it's
            # a priority queue

            if currentNode not in visitedNodes:

                visitedNodes.append(currentNode)

                currentNodeSuccesors = problem.getSuccessors(currentNode)
                for node in currentNodeSuccesors:
                    if node[0] not in visitedNodes:

                        nodePosition = node[0]
                        nodeActions = node[1]
                        nodeCost = node[2]

                        priority = currentCost + nodeCost + heuristic(nodePosition, problem)
                        nodeQueue.push(nodePosition, priority)

                        newActions = currentActions + [nodeActions]
                        actionsQueue.push(newActions, priority)
                        costQueue.push(currentCost + nodeCost, priority)

        return []
