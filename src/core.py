"""Center control for all.
"""
from logzero import logger
import os
import re
import math
import numpy as np
import codecs

# internal
from dataset import DialogHeadData, DialogData, HLAData
from matrix_factorization import MatrixWrapper, CharCluster
from config import HLA_TRAIN_CONFIG, HLA_CLUSTER_CONFIG, BasicConfig, SENT_NEIGHBOUR_CONFIG

RE_NL = re.compile(r'\n', re.S)
RE_T = re.compile(r'\t', re.S)
RE_B = re.compile(r'\|', re.S)


def clean_line(line):
    """Remove special char in line
    """
    try:
        line = re.sub(RE_NL, ' ', line)
        line = re.sub(RE_T, ' ', line)
        line = re.sub(RE_B, ' ', line)
    except:  # those should not happened
        logger.warn('{}:|{}|'.format(type(line), line))
        logger.warn(line == np.nan)
        logger.warn(math.isnan(line))
        raise
    return line.strip()


class DataManager:
    """Data Manager's main job: isolate dialog data based on train/test of shows, and one target character.
    """
    TRAIN_F = 'train.txt'
    TEST_F = 'test.txt'
    VALID_F = 'valid.txt'

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
        assert len(self.hd.data.loc[self.hd.data.char_id == self.target]) == 0,\
            "sanity fail: training head should not include target character"
        for sid in self.test_shows:
            assert len(self.hd.data.loc[self.hd.data.show_id == sid]) == 0,\
                "sanity fail: test show {} is in training".format(sid)

        # -- test character is not in train --
        assert not (self.target in self.train_chars)
        # TODO: check on all testing character is not in training

        # overall report
        logger.info('-- All sanity checks passed...')
        logger.info('REPORT: total dialogs: training {}, test {}, target {}'.format(len(self.train_dialogs),
                                                                                    len(self.test_dialogs),
                                                                                    len(self.target_dialogs)))
        logger.info('REPORT: Total {} training heads, Avg {:.2f} per sentence'.format(
            len(self.hd.data), float(len(self.hd.data)) / len(self.train_dialogs)))

    def __init__(self,
                 target, test_shows,
                 dialog_data=None,
                 head_data=None,
                 hla_data=None):
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
        self.test_shows = set(test_shows)
        self.dd = dialog_data
        if not self.dd:
            self.dd = DialogData()
        self.hd = head_data
        if not self.hd:
            self.hd = DialogHeadData(exclude_shows=self.test_shows)
        self.hla = hla_data
        if not self.hla:
            self.hla = HLAData()

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
        self.train_chars = set(self.train_dialogs.char_id.unique())
        self._pos = None
        self._neg = None
        self.state = 'train'

        self._sanity_report()  # perform sanity checks and report

    def _get_candidates(self, gt_id, sent_matrix: MatrixWrapper, sent_pool: set, candidate_count, selection='close'):
        """Get list of candidate sentence, relative to the ground truth sentence.

        Args:
            gt_id: sentence id of ground truth sentence.
            sent_matrix: trained sentence matrix.
            sent_pool: pool of sentence category. This limits the sentence we can choose from.
                This variable is normally uses to limit to negative character sentences.
                Note: Those are categories, NOT ids.
            candidate_count: number of total sentences need to be selected.
            selection: section policy. See docs.

        Returns:
            List of sentences as candidates, with total length of candidate_count.
        """
        # TODO: add more sentence selection options: e.g., random.
        if selection is not 'close':
            raise NotImplementedError('Only sentence selection close is supported.')

        _dias = None
        if self.state == 'train':
            _dias = self.train_dialogs
        elif self.state == 'test':
            _dias = self.test_dialogs
        else:
            raise Exception('state does not recognized: {}'.format(self.state))

        # select neighbourhood sentences.
        # we ask sentences 200 times more than we need, because some of them will not be used
        res = [_dias.loc[gt_id].dia2, ]
        if self.state == 'train':
            for sent_cat, _ in sent_matrix.get_similar(gt_id, top_n=candidate_count*200, convert_back=False):
                if sent_cat in sent_pool:
                    sent_id = sent_matrix.convert(sent_cat, to_category=False, raising=False)
                    if sent_id is None:  # filtered out
                        continue
                    _sent = _dias.loc[sent_id]
                    assert _sent.char_id in self._neg, "The selected dialog should belong to a negative character."
                    res.append(_sent.dia2)
                    if len(res) >= candidate_count:
                        break
        elif self.state == 'test':  # always random draw
            for dia in self.test_dialogs.sample(n=candidate_count-1):
                res.append(dia.dia2)
        else:
            raise Exception('Unknown state: {}'.format(self.state))
        np.random.shuffle(res)
        return res

    @staticmethod
    def _format_fb_line(hlas, d1, d2, sentences, reward=''):
        # TODO: add formatter injection as an option, alllow user to easier swtich to other formatter.
        # TODO 2: add formatter parameter control as part of config file
        line_cnt = 1  # count candidate
        line = ''  # overall write line
        for hla in hlas:
            line += '{} persona: i am {}.\n'.format(line_cnt, hla)
            line_cnt += 1
        # add ground truth
        line += '{} {}\t{}\t{}\t'.format(line_cnt, clean_line(d1), clean_line(d2), reward)

        for sent in sentences:
            line += '{}|'.format(clean_line(sent))
        line = line[:-1] + '\n'  # clip last '|' and add newline
        return line

    def write(self, path=None, random_seed=None, train_ratio=0.7):  # TODO: add weight, gt, neg, sentence option in next release.
        # weight_option='no_weight', gt_option='all', neg_option='negative', sent_option='close',
        # hla_placeholder=8, decouple_hla=True, cleaned_hlas=cleaned_hlas
        if path is None:
            path = BasicConfig.Output_Path
        if not os.path.exists(path):
            logger.warn('Path does not exists, attempt to create folder {}'.format(path))
            os.mkdir(path)
        if random_seed:
            raise NotImplementedError('random seed will be there in next release')  # TODO: random seeding
        assert 0 < train_ratio < 1, 'train ratio should be in between 0 and 1'

        # file system
        train_f = os.path.join(path, self.TRAIN_F)
        valid_f = os.path.join(path, self.VALID_F)
        test_f = os.path.join(path, self.TEST_F)
        assert (not os.path.exists(train_f)) and (not os.path.exists(test_f)) and (not os.path.exists(valid_f)),\
            "File already exists, stop operation to prevent overwrite"

        # build data
        char_mw = MatrixWrapper(self.hla, col1='char_id', col2='feature')
        char_mw.get_train(HLA_TRAIN_CONFIG, report_test=False)
        cc = CharCluster(self.target, matrix_wrapper=char_mw)
        # TODO: move parameters to class scope and fix, more easy to understand.
        self._pos, self._neg = cc.retrieve(config=HLA_CLUSTER_CONFIG, limits=self.train_chars)
        logger.info('-- INSPECT: Top 20 positive character__')
        for e in self._pos[:20]:
            logger.info('{}, conf:{}, score: {}'.format(self.hla.char_note(e[0]), e[1], e[2]))
        logger.info('-- INSPECT: 20 negative character__')
        _c = 20
        for e in self._neg:
            logger.info(self.hla.char_note(e))
            _c -= 1
            if _c < 0:
                break
        logger.info('__ END INSPECTION __')

        # build sentences
        logger.info('Building sentence neighbourhood...')
        sent_mw = MatrixWrapper(self.hd, col1='sent_id', col2='head_text')
        sent_mw.get_train(train_config=SENT_NEIGHBOUR_CONFIG, report_test=False)
        # build sentence pool, a set of sentence category allow to choose from
        # under current setting (close + negative), sentence pool allows only negative character sentences
        logger.info('Building sentence pool...')
        _sent_ids = self.dd.data.loc[self.dd.data.char_id.isin(self._neg)].index.values
        # convert to category
        sent_pool = set()
        for sid in _sent_ids:
            _cat = sent_mw.convert(sid, raising=False)
            if _cat:  # not none
                sent_pool.add(_cat)
        assert sent_pool, 'sent pool is empty'
        logger.info('Finished, total {} sentences in pool.'.format(len(sent_pool)))

        # Write training
        # For each positive character and all its dialogs, add 19 other candidates
        self.state = 'train'
        cnt = 0
        train_ptr = codecs.open(train_f, 'a+')
        valid_ptr = codecs.open(valid_f, 'a+')
        for cid, _, _ in self._pos:  # TODO: add weighted adjustment for both confidence and score
            # fetch all its dialogs
            cnt += 1
            logger.info('Write train {}/{}: {}'.format(cnt, len(self._pos), self.hla.char_note(cid)))
            gt_dialogs = self.train_dialogs.loc[self.train_dialogs.char_id == cid]
            for gt in gt_dialogs.itertuples():
                _hlas = self.hla.get_hlas(cid, amount=8, draw='random')
                _cands = self._get_candidates(gt_id=gt.Index, sent_matrix=sent_mw,
                                              sent_pool=sent_pool, candidate_count=20)

                line = self._format_fb_line(hlas=_hlas, d1=gt.dia1, d2=gt.dia2, sentences=_cands)
                if np.random.rand() < train_ratio:
                    train_ptr.write(line)
                else:
                    valid_ptr.write(line)
        train_ptr.close()
        valid_ptr.close()

        # Write testing
        self.state = 'test'
        test_ptr = codecs.open(test_f, 'a+')
        logger.info('Writing tests...')
        for gt in self.test_dialogs.loc[self.test_dialogs.char_id == self.target].itertuples():
            _hlas = self.hla.get_hlas(self.target, amount=8, draw='random')
            _cands = self._get_candidates(gt_id=gt.Index, sent_matrix=sent_mw,
                                          sent_pool=sent_pool, candidate_count=20)
            line = self._format_fb_line(hlas=_hlas, d1=gt.dia1, d2=gt.dia2, sentences=_cands)
            test_ptr.write(line)
        test_ptr.close()


# sample usage for fold 1 (Sheldon)
# dm = DataManager('l4390', test_shows=['GreysAnatomy', 'TheBigBangTheory', 'TheVampireDiaries', 'HowIMetYourMother',
#                                       'Smallville', 'Seinfeld', 'GilmoreGirls', 'DawsonsCreek'])
# dm.write()
