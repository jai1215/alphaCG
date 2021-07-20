import numpy as np
import math

EPS = 1e-8

class MCTS():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
 
        self.Qsa = {} # Q value for s,a
        self.Nsa = {} # N value for s,a(how many access)
        self.Ns  = {} # N value for s
        self.Ps  = {} # Policy?
        
        self.Es = {} # ended?
        self.Vs = {} # Valid move

    def getActionProb(self, playerBoard, temp=1):
        for i in range(self.args["numMCTSSims"]):
            self.search(playerBoard)

        s = self.game.stringRepresentation(playerBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(6)]
        # counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs


    def search(self, playerBoard):
        s = self.game.stringRepresentation(playerBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(playerBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(playerBoard.getNNInput())
            valids = self.game.getValidMoves(playerBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                print("state", playerBoard)
                print(self.Es[s])
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # # pick the action with the highest upper confidence bound
        # for a in range(self.game.getActionSize()):
        for a in range(6):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        # print(playerBoard, "select", a)
        next_s, next_player = self.game.getNextState(playerBoard, 1, a)
        next_s = self.game.getPlayerBoard(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v