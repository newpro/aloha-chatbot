"""Center control for all.
"""
from logzero import logger

# internal
from dataset import DialogHeadData, DialogData, HLAData


class DataManager:
    """Data Manager's main job: isolate dialog data based on train/test of shows, and one target character.
    """

    def _sanity_report(self):
        """Enforce sanity on instance variable of training, test, target dialogs, and report data length.

        Sanity checks are designed to capture mistakes of misuse data manager, or data integrity problem in datasets.
        """
        # -- target dialog sanity checks --
        _first_row = self.target_dialogs.iloc[0]  # first row of target character, contains meta data for checks
        assert len(self.target_dialogs) > 300,\
            "sanity fail: probably not an good idea to have testing dialogs lower than 300"
        assert _first_row.show_id in self.test_shows, "sanity fail: target has to be in test shows"
        # only one character is in target set, and is our target
        assert len(set(self.target_dialogs.char_id.unique())) == 1  # only one char in target
        assert list(self.target_dialogs.char_id.unique())[0] == self.target
        logger.info('REPORT: target {}({}) selected with {} testing dialogs.'.format(self.target,
                                                                                     _first_row.char_name,
                                                                                     len(self.target_dialogs)))

        # -- training sanity checks --
        assert len(self.train_dialogs.loc[self.train_dialogs.char_id == self.target]) == 0,\
            "sanity fail: training does not contain any target dialogs"
        # double check none of the test show is in training
        for sid in self.test_shows:  # this operation is lengthy but safer
            assert len(self.train_dialogs.loc[self.train_dialogs.show_id == sid]) == 0,\
                "sanity fail: test show {} is in training dialogs".format(sid)
        assert len(set(self.train_dialogs.char_id.unique()).
                   intersection(set(self.test_dialogs.char_id.unique()))) == 0,\
            "no overlap of same characters between train and test"

        # -- training head checks --
        assert len(self.train_heads.loc[self.train_heads.char_id == self.target]) == 0,\
            "sanity fail: training head should not include target character"
        for sid in self.test_shows:
            assert len(self.train_heads.loc[self.train_heads.show_id == sid]) == 0,\
                "sanity fail: test show {} is in training".format(sid)

        # overall report
        logger.info('-- All sanity checks passed...')
        logger.info('REPORT: total dialogs: training {}, test {}, target {}'.format(len(self.train_dialogs),
                                                                                    len(self.test_dialogs),
                                                                                    len(self.target_dialogs)))
        logger.info('REPORT: Total {} training heads, Avg {:.2f} per sentence'.format(
            len(self.train_heads), float(len(self.train_heads)) / len(self.train_dialogs)))

    def __init__(self,
                 target, test_shows,
                 dialog_data=None,
                 head_data=None,
                 hla_data=None,
                 head_count_ge=5):
        """Build data manager.

        Args:
            target: character ID of the target. This character is what the chat bot trying to talk like.
            test_shows: list of set of show id. Dialogs within those shows will be
            dialog_data: dialog dataset instance. See its API for details.
                by default none, auto build itself (recommended).
            head_data: dialog head dataset instance. See its API for details.
                by default none, auto build itself (recommended).
            hla_data: hla dataset instance. See its API for details.
                by default none, auto build itself (recommended).
            head_count_ge: int, default 5. Filter out head data that lower than this number.
        """
        # load datasets
        self.dd = dialog_data
        if not self.dd:
            self.dd = DialogData()
        self.hd = head_data
        if not self.hd:
            self.hd = DialogHeadData()
        self.hla = hla_data
        if not self.hla:
            self.hla = HLAData()
        self.test_shows = set(test_shows)

        # assign target character and dialogs
        self.target = target
        assert target, 'target can not be empty'
        # assert if target has dialog
        self.target_dialogs = self.dd.data.loc[self.dd.data.char_id == target]

        # build tests data
        self.test_dialogs = self.dd.data.loc[(self.dd.data.char_id != target)
                                             & (self.dd.data.show_id.isin(self.test_shows))]

        # build training data
        # training data is all dialog data except dialogs in test shows.
        self.train_dialogs = self.dd.data.loc[~self.dd.data.show_id.isin(self.test_shows)]

        # build head data for training
        _sents = set(self.train_dialogs.index.values)
        train_heads = self.hd.data.loc[self.hd.data.sent_id.isin(_sents)]
        self.train_heads = train_heads[
            train_heads.groupby('head_text')['head_text'].transform('count').ge(head_count_ge)]

        self._sanity_report()  # perform sanity checks and report


# sample usage for fold 1 (Sheldon)
# dm = DataManager('l4390', test_shows=['GreysAnatomy', 'TheBigBangTheory', 'TheVampireDiaries', 'HowIMetYourMother',
#                                       'Smallville', 'Seinfeld', 'GilmoreGirls', 'DawsonsCreek'])

