from MCTS import MCTS
from tqdm import tqdm
import numpy as np
from pickle import Pickler, Unpickler
from random import shuffle
from Arena import Arena
import os

from collections import deque

import logging
import coloredlogs

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

class Manager():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.trainHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainHistory)
        f.closed


    def executeEpisode(self):
        trainSample = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        #Cur Player
        episodeStep = 0

        while True:
            episodeStep += 1
            playerBoard = self.game.getPlayerBoard(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            pi = self.mcts.getActionProb(playerBoard, temp=temp)
            sym = self.game.getSymmetries(playerBoard, pi)
            for b, p in sym:
                trainSample.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            next_board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(next_board, self.curPlayer)

            if r != 0:
                # print("Prev>>", board, action)
                # print("Next>>", next_board)
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainSample]
            board = next_board
    
    def learn(self):
        for i in range(self.args['numIters']):
            log.info("Start Iteration %d" % i)

            trainSample = deque([], maxlen=self.args["maxlenOfQueue"])

            for _ in tqdm(range(self.args["numEps"]), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                trainSample += self.executeEpisode()
            self.trainHistory.append(trainSample)
            if len(self.trainHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainHistory) = {len(self.trainHistory)}")
                self.trainHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            trainExamples = []
            for e in self.trainHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='test.pth.tar')
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='test.pth.tar')
            # self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # pmcts = MCTS(self.game, self.pnet, self.args)
            # nmcts = MCTS(self.game, self.nnet, self.args)
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'
            # save the iteration examples to the history 
            # self.trainExamplesHistory.append(trainSample)

