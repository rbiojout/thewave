from ..tdagent import TDAgent
import numpy as np

class UBAH(TDAgent):

    def __init__(self, b = None, nb_tickers = 10):
        super(UBAH, self).__init__()
        self.b = b
        self.nb_tickers = nb_tickers

    def decide_by_history(self, x, last_b):
        '''return new portfolio vector
        :param x: input matrix
        :param last_b: last portfolio weight vector
        '''
        if self.b is None:
            vec_size = self.nb_tickers + 1
            self.b = np.ones(vec_size) / vec_size
        else:
            self.b = last_b
        return self.b
