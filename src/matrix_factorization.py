"""Matrix factorization utilities.
"""
from typing import Optional
import numpy as np
from logzero import logger
from collections import Counter
import math

# matrix support
from scipy.sparse import coo_matrix
import implicit
from implicit.evaluation import train_test_split, precision_at_k

# internal
from dataset import CSVData, HLAData
from config import MatrixTrainingConfig, ClusterConfig


class _TrainedModelWrapper:
    """Wrapper for trained model. Provides utilities support for inference data from trained model.
    """

    def __init__(self, als):
        assert als, 'pass in als can not be empty'
        self.als = als

    def inspect_feature(self, feature=None):
        if feature is None:  # inspect all feature, pass on to outside
            return None


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
        self.dataset = dataset
        assert len(self.dataset.data) > 0
        assert col1 in self.dataset.COLs
        assert col2 in self.dataset.COLs
        self.col1 = col1
        self.col2 = col2

        # build coo for col1, 2
        self.coo, self.cat1, self.cat2 = self._to_coo(self.dataset.data)
        logger.info(repr(self.coo))
        self.model = None

        # build category mapping data
        # m1 is for object, m2 is for feature
        # maps is built from both directions: forward direction cat_id: value.
        self.m1 = dict(enumerate(self.cat1.cat.categories))
        self.m1_inv = {r: i for i, r in self.m1.items()}
        self.m2 = dict(enumerate(self.cat2.cat.categories))
        self.m2_inv = {r: i for i, r in self.m2.items()}
        self.m1_count = len(self.m1)
        self.m2_count = len(self.m2)
        # all key and value should be unique (map to category id)
        assert self.m1_count == len(self.m1_inv)
        assert self.m2_count == len(self.m2_inv)

    def get_train(self, train_config: MatrixTrainingConfig,
                  report_test: Optional[bool] = True, test_df=None,
                  overwrite=False):
        if self.model and not overwrite:
            raise Exception('Already trained and does not allow overwrite (consider access via model instance).')

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
            _model = implicit.als.AlternatingLeastSquares(factors=train_config.factor,
                                                          regularization=train_config.regularization,
                                                          iterations=train_config.iterations)
            _model.fit(train_csr * train_config.conf_scale)
            prec = precision_at_k(_model, train_csr.T, test_csr.T, K=train_config.top_n)
            logger.warning('ACCURACY REPORT at top {}: {:.5f}%'.format(train_config.top_n, prec * 100))
            if train_config.safe_pass:
                assert prec > train_config.safe_pass

        # training on complete matrix
        logger.info('Training on complete matrix')
        _model = implicit.als.AlternatingLeastSquares(factors=train_config.factor,
                                                      regularization=train_config.regularization,
                                                      iterations=train_config.iterations)
        _model.fit(self.coo * train_config.conf_scale)
        self.model = _model

    def convert(self, value, to_category=True, feature=False):
        _map = None
        if to_category:  # id -> category
            if feature:  # get feature
                _map = self.m2_inv
            else:
                _map = self.m1_inv
        else:  # category -> id
            if feature:
                _map = self.m2
            else:
                _map = self.m1
        # get value
        try:
            res = _map[value]
        except KeyError:
            logger.warn('DEBUG MAP: {}'.format(_map))
            raise KeyError('Map does not exists for value: {}'.format(value))
        return res

    def get_similar(self, id_, top_n: Optional[int] = None, feature=False, convert_back=True):
        """Get top ranking for similar objects or features.

        Args:
            id_: the id value of the object / feature.
            top_n: bool, default none. top n ranking items will be returned.
                If not provided, all items will be returned.
            feature: bool, default false. If false, object similarity will be used, otherwise feature will be used.
            convert_back: bool, default true. If true, the id of the items rather than category will be returned.
                This is a useful wrapper if you do not want to concern with category id.
                However, this would slow down operations if you have additional pipelines attached using categories.

        Returns:
            List of items, each item has format (category_id, confidence) or (item_id, confidence) depend on
                convert_back variable.
        """
        assert type(feature) is bool, 'feature indicates if feature or object is used, must be boolean.'
        if not self.model:
            raise Exception('Model has not been trained yet (call get_train first)')
        cat = self.convert(id_, feature=feature, to_category=True)
        if feature:
            res = self.model.similar_items(cat, N=top_n)
        else:
            res = self.model.similar_users(cat, N=top_n)
        if convert_back:
            return [(self.convert(e[0], to_category=False, feature=feature), e[1]) for e in res]
        return res

    def inspect(self, id_, top_n=20, feature=False, readable_fn=None):
        if feature:
            logger.info('-- INSPECTION: Feature close {} --'.format(id_))
        else:
            logger.info('-- INSPECTION: Object close to {} --'
                        .format(id_ if readable_fn is None else readable_fn(id_)))
        for fid, conf in self.get_similar(id_, top_n=top_n, feature=feature):
            logger.info('{}: {}'.format(fid if readable_fn is None else readable_fn(fid), conf))
        logger.info('__ END INSPECTION __')


# sample usage
# from config import HLA_TRAIN_CONFIG  # default hla training configuration
# hla_d = HLAData()
# mw = MatrixWrapper(hla_d, col1='char_id', col2='feature')
# mw.get_train(HLA_TRAIN_CONFIG, report_test=False)
# mw.inspect('AbusiveParents', feature=True)  # inspect model understanding to "abusive parent"
# mw.inspect('TheHeart', feature=True)  # inspect model understanding to "the heart"
# mw.inspect('l4390', feature=False, readable_fn=hla_d.char_note)  # inspect model top 20 ranking close to Sheldon


class CharCluster:
    """Target character cluster manager.

    The manager build character cluster based on one cluster provided.
    """
    # TODO: add experimental cleaned hla data support
    def __init__(self, target,
                 matrix_wrapper: MatrixWrapper):
        self.target = target
        self.mw = matrix_wrapper
        logger.info('CLUSTER: Lock target {}'.format(target))

    def _expand(self, l1, l2, aco, weighted, log_scale):
        """Communities expand from target as center by two levels. For details, refer to paper and docs.

        Returns:
            positive characters: set of character id for positives.
            negative characters: set of character id for negatives.
        """
        # level 2 holder. Key: candidate category id; value: (freq, score)
        level1 = self.mw.get_similar(self.target, top_n=l1, feature=False)

        # expand level 1 and further level 2
        counter = Counter()
        logger.info('Level 1 total {}'.format(len(level1)))
        for char1, _ in level1:
            for char2, _ in self.mw.get_similar(char1, feature=False, top_n=l2):
                counter[char2] += 1

        # build positive / negative character set
        _pos, _neg = [], []
        print(counter)
        logger.info('Level 2 total {}'.format(len(counter)))
        for char_id, freq in counter.most_common():
            score = 1
            if weighted:
                if log_scale:
                    score = math.log(freq)
                else:
                    score = freq
            if freq >= aco:
                _pos.append((char_id, freq, score,))
            else:
                _neg.append((char_id, freq, score,))
        return _pos, _neg

    def retrieve(self, config: ClusterConfig):
        l1 = config.l1 if config.l1 is not None else self.mw.m1_count
        l1 = min(int(float(self.mw.m1_count) * config.perc_cutoff / 100), l1)
        l2 = config.l2 if config.l2 is not None else self.mw.m1_count
        logger.info('Considering {} out of L1: {}, L2 {} top ranked characters.'.format(l1, l2, self.mw.m1_count))

        positives, negatives = self._expand(l1, l2, config.aco, config.weighted, config.log_scale)
        return positives, negatives


# sample usage
# from config import HLA_CLUSTER_CONFIG, HLA_TRAIN_CONFIG
# hla_d = HLAData()
# mw = MatrixWrapper(hla_d, col1='char_id', col2='feature')
# mw.get_train(HLA_TRAIN_CONFIG, report_test=False)
# cc = CharCluster('l4390', matrix_wrapper=mw)
# pos, neg = cc.retrieve(config=HLA_CLUSTER_CONFIG)
# print([hla_d.char_note(char_id) for char_id, _, _ in pos[:40]])
# print([hla_d.char_note(char_id) for char_id, _, _ in neg[:40]])
