
import numpy as np

class owareGame():
    def __init__(self):
        pass

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board()
        return b

    def getPlayerBoard(self, board, player):
        if player == -1:
            return Board(board.board1, board.board0, [board.point[1], board.point[0]], board.turn)
        return board

    def stringRepresentation(self, board):
        return board.toString()

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        #If Turn == 200: Compare Point
        #If I have more than 25 Point return +1
        #If Y have more than 25 Point return -1
        valid = self.getValidMoves(board, player)
        if (board.turn == 200) | (valid.sum() == 0):
            if board.point[0] >= board.point[1]:
                return +1
            else:
                return -1
        if board.point[0] >= 25:
            return +1
        if board.point[1] >= 25:
            return -1
        return 0
    
    def getSymmetries(self, board, pi):
        l = [[board, pi]]
        #no Symmetries
        return l

    def getValidMoves(self, board, player):
        valid = [0]*6
        b0 = board.board0 if player == 1 else board.board1
        b1 = board.board1 if player == 1 else board.board0
        b1_zeros = sum(b1) == 0
        b1_0_oneseed = (sum(b1[1:]) == 0) & (b1[0] == 1)
        for i in range(len(valid)):
            #if b0 has seed
            cur_seed = b0[i]
            if cur_seed != 0:
                valid[i] = 1
                #if board1 does not have seed, then b0 must give seed
                if b1_zeros & ((i+cur_seed) < 6):
                    valid[i] = 0
                #if board1[0] has 1 seed and b0 give only one seed
                if b1_0_oneseed & ((i+cur_seed) == 6):
                    valid[i] = 0
        return np.array(valid)

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        # print(">> Action [%d] Player[%d] " % (action, player), board)
        nb = board.copy()
        seeds = nb.board0[action] if player==1 else nb.board1[action]
        nb.board0[action] = 0 if player == 1 else nb.board0[action]
        nb.board1[action] = 0 if player == -1 else nb.board1[action]

        assert(seeds > 0)
        assert(nb.turn <= 200)

        destRaw = 1+action
        #Calculate Board
        while(seeds>0):
            owner = 1 if ((destRaw%12)<6) else -1
            dest  = destRaw%6
            destSeed = nb.board0[dest] if owner==player else nb.board1[dest]
            if(destSeed >= 12):
                pass#Skip
            elif(owner==player)&(dest==action):
                pass#Skip
            else:
                if(owner==player):
                    nb.board0[dest] += 1
                else:
                    nb.board1[dest] += 1
                seeds -= 1
            destRaw += 1
        #Calculate Point
        owner = 1 if ((destRaw%12)<6) else -1
        dest  = destRaw%6
        destSeed = nb.board0[dest] if owner==player else nb.board1[dest]
        if (owner == -1) & (destSeed in [2, 3]):
            if(player == 1):
                nb.board1[dest] = 0
                nb.point[0] += destSeed
            else:
                nb.board0[dest] = 0
                nb.point[1] += destSeed
        #Calculate Turn
        nb.turn += 1
        # print("<< Action [%d] Player[%d] " % (action, player), nb)
        assert((sum(nb.board0)+sum(nb.board1)+sum(nb.point))==48)
        # nb.swapBoard()
        return (nb, -player)
            
class Board():
    def __init__(self, board0=None, board1=None, point=None, turn=None):
        self.board0 = board0 if board0 else [4]*6
        self.board1 = board1 if board1 else [4]*6
        self.point  = point if point else [0,0]
        self.turn   = turn  if turn else 1
        self.prv = None
    
    def toString(self):
        ret = ""
        for b in self.board0:
            ret += "%x" % b
        ret += "_"
        for b in self.board1:
            ret += "%x" % b
        ret += "_"
        ret += "%02d_%02d_%03d" % (self.point[0], self.point[1], self.turn)
        return ret

    def getNNInput(self):
        # return np.array(self.board0 + self.board1 + self.point + [self.turn])
        return np.array(self.board0 + self.board1)
        
    def copy(self):
        return Board(list(self.board0),list(self.board1),list(self.point),self.turn)
    
    def swapBoard(self):
        tmp = self.board0
        self.board0 = self.board1
        self.board1 = tmp
    
    def __str__(self) -> str:
        return "T[%3d] (%2d : %2d) " % (self.turn, self.point[0], self.point[1])+str(self.board0) + str(self.board1)
    
    def __repr__(self):
        return self.__str__()