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


from util import manhattanDistance
from game import Directions
import random, util

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
				some Directions.X for some X in the set {North, South, West, East, Stop}
				"""
				# Collect legal moves and successor states
				legalMoves = gameState.getLegalActions()

				# Choose one of the best actions
				scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
				bestScore = max(scores)
				bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
				chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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

				"*** YOUR CODE HERE ***"

				if successorGameState.isWin():
						return float("inf")

				for ghostState in newGhostStates:
						if util.manhattanDistance(ghostState.getPosition(), newPos) < 2:
								return float("-inf")

				foodDist = [manhattanDistance(foodPos, newPos) for foodPos in newFood.asList()]

				foodSuccessor = 0
				if (currentGameState.getNumFood() > successorGameState.getNumFood()):
						foodSuccessor = 300

				return successorGameState.getScore() - 5 * min(foodDist) + foodSuccessor

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

		def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
				self.index = 0 # Pacman is always agent index 0
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
				"""
				"*** YOUR CODE HERE ***"
				_, action = self.minimaxSearch(gameState, agentIndex = 0, depth = self.depth)
				return action
		
		def minimaxSearch(self, gameState, agentIndex, depth):
				if depth == 0 or gameState.isLose() or gameState.isWin():
						return (self.evaluationFunction(gameState), Directions.STOP)
				else:
						if self.firstAgent(gameState, agentIndex):
								return self.chooseBranch(gameState, agentIndex, depth, lambda nextScore, curScore: nextScore > curScore) 
						else:
								return self.chooseBranch(gameState, agentIndex, depth, lambda nextScore, curScore: nextScore < curScore)
		
		def chooseBranch(self, gameState, agentIndex, depth, option):
				legalActions = gameState.getLegalActions(agentIndex)
				if self.lastAgent(gameState, agentIndex):
						nextAgent, nextDepth = 0, depth - 1
				else:
						nextAgent, nextDepth = agentIndex + 1, depth
				
				if option(1, 2):
						bestScore = float('inf')
				else:
						bestScore = float('-inf')
				action = Directions.STOP

				for legalAction in legalActions:
						successorGameState = gameState.generateSuccessor(agentIndex, legalAction)
						newScore, _ = self.minimaxSearch(successorGameState, nextAgent, nextDepth)
						if option(newScore, bestScore):
								bestScore, action = newScore, legalAction
											
				return (bestScore, action)
		
		def lastAgent(self, gameState, agentIndex):
				return agentIndex == gameState.getNumAgents() - 1
		
		def firstAgent(self, gameState, agentIndex):
				return agentIndex == 0

class AlphaBetaAgent(MultiAgentSearchAgent):
		"""
			Your minimax agent with alpha-beta pruning (question 3)
		"""

		def getAction(self, gameState):
				"""
					Returns the minimax action using self.depth and self.evaluationFunction
				"""
				"*** YOUR CODE HERE ***"
				_, action = self.ABPruning(gameState, agentIndex = 0, depth = self.depth)
				return action
		
		def ABPruning(self, gameState, agentIndex, depth, alpha = float('-inf'), beta = float('inf')):
				if depth == 0 or gameState.isLose() or gameState.isWin():
						return (self.evaluationFunction(gameState), Directions.STOP)
				else:
						if self.firstAgent(gameState, agentIndex):
								return self.maxChoice(gameState, 
																		agentIndex, 
																		depth, 
																		alpha, beta)
						else:
								return self.minChoice(gameState, 
																		agentIndex, 
																		depth, 
																		alpha, beta)

		def firstAgent(self, gameState, agentIndex):
				return agentIndex == 0
		
		def lastAgent(self, gameState, agentIndex):
				return agentIndex == gameState.getNumAgents() - 1
		
		def chooseBranch(self, gameState, agentIndex, depth, alpha, beta, option):
				legalActions = gameState.getLegalActions(agentIndex)

				if self.lastAgent(gameState, agentIndex):
						nextAgent, nextDepth = 0, depth - 1
				else:
						nextAgent, nextDepth = agentIndex + 1, depth
				
				if option(1, 2):
						bestScore = float('inf')
				else:
						bestScore = float('-inf')
				action = Directions.STOP

				for legalAction in legalActions:
						successorGameState = gameState.generateSuccessor(agentIndex, legalAction)
						nextScore, nextAction = self.ABPruning(successorGameState, nextAgent, nextDepth, alpha, beta)
						
						if option(nextScore, bestScore):
								bestScore, action = nextScore, legalAction
						
						if option(nextScore, beta):
								return (nextScore, nextAction)
						
						if option(1, 2):
								beta = min(beta, nextScore)
						else:
								alpha = max(alpha, nextScore)
				
				return (bestScore, action)
		
		def maxChoice(self, gameState, agentIndex, depth, alpha, beta):
				legalActions = gameState.getLegalActions(agentIndex)
				
				if self.lastAgent(gameState, agentIndex):
						nextAgent, nextDepth = 0, depth - 1
				else:
						nextAgent, nextDepth = agentIndex + 1, depth
				
				bestScore, action = float('-inf'), Directions.STOP

				for legalAction in legalActions:
						successorGameState = gameState.generateSuccessor(agentIndex, legalAction)
						nextScore, nextAction = self.ABPruning(successorGameState, nextAgent, nextDepth, alpha, beta)
						
						if nextScore > bestScore:
								bestScore, action = nextScore, legalAction
						
						if nextScore > beta:
								return (nextScore, nextAction)
						
						alpha = max(alpha, nextScore)
				
				return (bestScore, action)

		def minChoice(self, gameState, agentIndex, depth, alpha, beta):
				legalActions = gameState.getLegalActions(agentIndex)
		
				if self.lastAgent(gameState, agentIndex):
						nextAgent, nextDepth = 0, depth - 1
				else:
						nextAgent, nextDepth = agentIndex + 1, depth
				
				bestScore, action = float('inf'), Directions.STOP

				for legalAction in legalActions:
						successorGameState = gameState.generateSuccessor(agentIndex, legalAction)
						nextScore, nextAction = self.ABPruning(successorGameState, nextAgent, nextDepth, alpha, beta)
						
						if nextScore < bestScore:
								bestScore, action = nextScore, legalAction
						
						if nextScore < alpha:
								return (nextScore, nextAction)
						
						beta = min(beta, nextScore)
				
				return (bestScore, action)

						

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
				"*** YOUR CODE HERE ***"
				_, action = self.expectimaxSearch(gameState, agentIndex = 0, depth = self.depth)
				return action

		def expectimaxSearch(self, gameState, agentIndex, depth):
				if depth == 0 or gameState.isLose() or gameState.isWin():
						return (self.evaluationFunction(gameState), Directions.STOP)
				else:
						if self.firstAgent(gameState, agentIndex):
								return self.maxChoice(gameState, agentIndex, depth)
						else:
								return self.expectChoice(gameState, agentIndex, depth)
		
		def firstAgent(self, gameState, agentIndex):
				return agentIndex == 0
		
		def lastAgent(self, gameState, agentIndex):
				return agentIndex == gameState.getNumAgents() - 1
		
		def maxChoice(self, gameState, agentIndex, depth):
				legalActions = gameState.getLegalActions(agentIndex)
				successorGameStates = [gameState.generateSuccessor(agentIndex, legalAction) for legalAction in legalActions] 
				if self.lastAgent(gameState, agentIndex):
						nextAgent, nextDepth = 0, depth - 1
				else:
						nextAgent, nextDepth = agentIndex + 1, depth

				scores = [self.expectimaxSearch(successorGameState, nextAgent, nextDepth)[0] for successorGameState in successorGameStates]
				bestScore = max(scores)
				bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
				chosenIndex = random.choice(bestIndices)
				return (bestScore, legalActions[chosenIndex])
		
		def expectChoice(self, gameState, agentIndex, depth):
				legalActions = gameState.getLegalActions(agentIndex)
				successorGameStates = [gameState.generateSuccessor(agentIndex, legalAction) for legalAction in legalActions]

				if self.lastAgent(gameState, agentIndex):
						nextAgent, nextDepth = 0, depth - 1
				else:
						nextAgent, nextDepth = agentIndex + 1, depth
				
				scores = [self.expectimaxSearch(successorGameState, nextAgent, nextDepth)[0] for successorGameState in successorGameStates]
				averageScore = sum(scores) / float(len(scores))
				return (averageScore, Directions.STOP)

def betterEvaluationFunction(currentGameState):
		"""
			Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
			evaluation function (question 5).

			DESCRIPTION: <write something here so we know what you did>
		"""
		"*** YOUR CODE HERE ***"

		newPos = currentGameState.getPacmanPosition()
		newFood = currentGameState.getFood()
		newGhostStates = currentGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


		if currentGameState.isWin():
				return float('inf')
		if currentGameState.isLose():
				return float('-inf')

		ghostDist = [manhattanDistance(ghostState.getPosition(), newPos) for ghostState in newGhostStates]
		if min(ghostDist) < 2:
				return float('-inf')

		foodDist = [manhattanDistance(foodPos, newPos) for foodPos in newFood.asList()]
		averageFoodDist = sum(foodDist) / float(len(foodDist))
		numFood = len(foodDist)


		score = currentGameState.getScore()
		foodEvaluation = -(0.9 * min(foodDist) + 0.1 * averageFoodDist) / 2.0 - numFood
		ghostEvaluation =  -5.0 / (min(ghostDist) + 1.0)
		capsuleEvaluation = - 50.0 * len(currentGameState.getCapsules()) + 0.1* sum(newScaredTimes)

		return score + foodEvaluation + ghostEvaluation + capsuleEvaluation

# Abbreviation
better = betterEvaluationFunction

