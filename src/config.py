# config files for easy modification
from typing import Optional


# -- Storage classes --
class FileInfo:
    def __init__(self, name, check_sum, reproduce_length=None):
        self.file_name = name
        self.md5 = check_sum
        self.reproduce_length = reproduce_length  # length after clean


class MatrixTrainingConfig:
    """Matrix factor training configuration.
    """
    def __init__(self, top_n: int,
                 conf_scale: int, factor: int, regularization: float, iterations: int,
                 random_state: Optional[int] = None, safe_pass: Optional[float] = None,
                 train_percentage: Optional[float] = 0.7):
        """Init Matrix config

        Args:
            top_n: testing evaluation of top n retrieval.
            conf_scale: scale up the matrix with a confidence integer.
            factor:
            regularization: matrix regularization.
            iterations: perform a fixed amount of iterations before testing.
            random_state: Optional. To fix the result for reproduction.
            safe_pass: per
        """
        self.top_n = top_n
        self.conf_scale = conf_scale
        self.factor = factor
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        assert 0 < train_percentage < 1
        self.train_percentage = train_percentage
        if safe_pass is not None:
            assert 0 < safe_pass < 1
        self.safe_pass = safe_pass


class ClusterConfig:
    def __init__(self,
                 acceptable_overlap: int, perc_cutoff: int,
                 level1_limit: Optional[int] = None, level2_limit: Optional[int] = None,
                 weighted: Optional[bool] = False, log_scale: Optional[bool] = True):
        """Configuration parameters for clustering.

        Args:
            acceptable_overlap: int, overlap frequency.
                Community item has to overlap at least this amount to consider as positive.
            perc_cutoff: int, percentage range between 3 to 50.
                At level one community cluster, only top ranked percentage of items neighbours allow to be counted.
            level1_limit: int, level 1 cluster will be limit to this amount.
                If not provided, use all.
                If the value is bigger than perc_cutoff, will not have any effect (use whichever is smaller)
            level2_limit: int, level 2 cluster will be limit to this amount.
                if not provided, use all.
            weighted: bool, by default false.
                if true, the training data of the character to target is weighted by replication.
                This is an experimental setting, use with caution!
                If enable, depend on the language model you are using, it may overfit very quickly.
                For more details and reasons, check on warnings in docs.
            log_scale: bool, by default true. Only useful in weighted condition.
                If true, the weight is shrink by log scale. This is used for control overfit.
        """
        assert 3 < perc_cutoff < 50, 'not a good percentage range'
        self.perc_cutoff = perc_cutoff
        self.l1 = level1_limit
        self.l2 = level2_limit
        assert acceptable_overlap > 3, 'acceptable overlap too low'
        self.aco = acceptable_overlap
        self.weighted = weighted
        self.log_scale = log_scale


# == Configs ==
# Modify followings to change experiment parameters
class BasicConfig:
    """Required configuration parameters, e.g., file locations.
    """
    # Absolute path to all raw training data
    Data_Path = '/home/kits-adm/Datasets/offical_releases_aloha/raw_data'

    # Lock for check data match to research version.
    # You can disable this when you modify the code for experiment.
    # When enable, assert data results to be exact as research produced.
    Reproduce_Lock = False  # set to false to disable reproduce check.

    # File names within base paths.
    # If you want to reproduce, download our original data and do not change this section.
    F_Dialog = FileInfo('all_dialogs.csv',
                        'ad37d29ee832f6cb21d0a2ab08382f5f',
                        reproduce_length=1042647)
    F_DialogHead = FileInfo('all_dialogs_heads.csv',
                            '6dbe7b50e8c385e36cdb1e44352476db',
                            reproduce_length=13530074)
    F_HLAs = [FileInfo('char_features_live.csv', 'dcda13f9fc87c54a8e537b9919b391b1'),
              FileInfo('char_features_animated.csv', '6fed284b6e7b1133096efef55cf1a451')]
    F_HLAs_Length = 945519  # reproduce length


# Configuration for hla matrix factorization training
# this configuration should achieve 25% in wild setting (reported on paper), 45% on cleaned setting (experimental)
HLA_TRAIN_CONFIG = MatrixTrainingConfig(
    top_n=100, regularization=100, iterations=500,
    factor=36, conf_scale=20, random_state=649128,
    safe_pass=0.2,
)

HLA_CLUSTER_CONFIG = ClusterConfig(
    perc_cutoff=10, level2_limit=30, acceptable_overlap=10, weighted=False,
)
