"""Matrix factorization utilities.
"""
import os
from typing import Optional
import numpy as np
from logzero import logger

# matrix support
from scipy.sparse import coo_matrix
import implicit
from implicit.evaluation import train_test_split, precision_at_k

# internal
from dataset import CSVData


class MatrixWrapper:
    def _to_coo(self, data):
        cat1 = data[self.col1].astype('category')
        cat2 = data[self.col2].astype('category')
        coo = coo_matrix((np.ones(data.shape[0]), (cat2.cat.codes.copy(), cat1.cat.codes.copy())))
        return coo, cat1, cat2

    def __init__(self, dataset: CSVData, col1, col2):
        """

        Args:
            dataset: dataset object.
            col1: object column name.
            col2: feature column name.
        """
        self.data = dataset.data
        self.cols = dataset.COLs
        assert len(self.data) > 0
        assert col1 in self.cols
        assert col2 in self.cols
        self.col1 = col1
        self.col2 = col2

        # build coo for col1, 2
        self.coo, self.cat1, self.cat2 = self._to_coo(self.data)
        logger.info(repr(self.coo))

        # build category mapping data
        self.m1 = dict(enumerate(self.cat1.cat.categories))
        self.m1_inv = {r: i for i, r in self.m1.items()}
        self.m2 = dict(enumerate(self.cat2.cat.categories))
        self.m2_inv = {r: i for i, r in self.m2.items()}

    def get_train(self, train_config, report_test: Optional[bool] = True, test_df=None):
        assert train_config, 'train configuration has to be provided.'

        if report_test:
            logger.info('-- Performing MM sanity check on {} {}'.format(self.col1, self.col2))
            if test_df is None:
                if train_config.random_state:
                    np.random.seed(train_config.random_state)
                train_csr, test_csr = train_test_split(self.coo, train_percentage=train_config.train_percentage)
            else:
                assert len(test_df) > 0
                train_csr = self.coo
                test_csr = self._to_coo(test_df)
            _als = implicit.als.AlternatingLeastSquares(factors=train_config.factor,
                                                        regularization=train_config.regularization,
                                                        iterations=train_config.iterations)
            _als.fit(train_csr * train_config.conf_scale)
            prec = precision_at_k(_als, train_csr.T, test_csr.T, K=train_config.top_n)
            logger.warning('ACCURACY REPORT at top {}: {:.5f}%'.format(train_config.top_n, prec*100))
            if train_config.safe_pass:
                assert prec > train_config.safe_pass

        # training on complete matrix
        logger.info('Training on complete matrix')
        _als = implicit.als.AlternatingLeastSquares(factors=train_config.factor,
                                                    regularization=train_config.regularization,
                                                    iterations=train_config.iterations)
        _als.fit(self.coo * train_config.conf_scale)
        return _als


# sample usage
# from dataset import HLAData
# from config import HLA_Training_Config
#
# mw = MatrixWrapper(HLAData(), col1='char_id', col2='feature')
# mw.get_train(HLA_Training_Config)
