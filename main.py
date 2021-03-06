import logging
import coloredlogs

from Manager import Manager
from oware.owareGame import owareGame as Game
from oware.NNet import NNetWrapper as NNet

from utils import *

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

def main():
    log.info('Make Game')
    game = Game()
    log.info('Make NN')
    nnet = NNet(game)
    log.info('[TBD] Load NN checkpoint')
    log.info('[TBD] Load Manager')
    manager = Manager(game, nnet, args)
    log.info('[TBD] Load Manager NN checkpoint')
    log.info('[TBD] Start Learning')
    manager.learn()


if __name__ == "__main__":
    main()