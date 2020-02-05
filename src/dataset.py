"""Dataset loaders and pre-processing.
"""
# core library
import os
import hashlib
import pandas as pd
from logzero import logger
from typing import Optional

# internal
from config import BasicConfig, FileInfo


class CSVIntegrityException(Exception):
    pass


def _md5_checksum(file_path):
    with open(file_path, 'rb') as fh:
        m = hashlib.md5()
        while True:
            data = fh.read(8192)
            if not data:
                break
            m.update(data)
        return m.hexdigest()


class CSVData:
    COLs = None

    def _assert_col(self, col):
        if col not in self.data.columns:
            logger.exception('Column does not exists: {}'.format(col))

    def _report_reduction(self, old_length, message):
        if self.data.shape[0] != old_length:
            logger.info('Reduction report {}: {} -> {}.'.format(message, old_length, self.data.shape[0]))
        else:
            logger.info('Reduction report {}: no change {}.'.format(message, old_length))

    def clean(self,
              high_passes=None, silent_high_pass=False,
              cross_ref_high_passes=None,
              filter_empties=None,
              force_removals=None,
              shuffle_state: Optional[int] = None):
        """Clean data attribute.

        Args:
            high_passes: list of high pass filters.
                each element follows format: (column name, high pass number)
                Any items within the column that lower than high pass will be discarded.
            silent_high_pass: do not print out exclusion set for high pass.
                Set it true when it is too much to display.
            cross_ref_high_passes: list of high pass with one attribute reference to another.
                each element follows format: (column name, attribute 1, attribute 2)
                Any items map of attribute 1 to 2 and vice versa, lower than high pass value will be discarded.
                For example: ('character', 'feature', 5) indicates discard all character lower than 5 features.
            filter_empties: list of column name that data should not contain null.
                each element is string, represent column name.
            force_removals: list of items in columns that force to be removed for whatever reason.
                each element follow format: (column name, item name)
            shuffle_state: int, default none.
                If provided, data will be shuffled at the very end with this state as random seed.
        """
        if high_passes is None:
            high_passes = []
        if cross_ref_high_passes is None:
            cross_ref_high_passes = []
        if filter_empties is None:
            filter_empties = []
        if force_removals is None:
            force_removals = []

        for col, hp in high_passes:
            self._assert_col(col)
            _exclude = set()
            for item, count in self.data[col].value_counts().iteritems():
                if count < hp:
                    _exclude.add(item)
            if _exclude:
                _old_length = self.data.shape[0]
                self.data = self.data.loc[~self.data[col].isin(_exclude)]
                if not silent_high_pass:
                    logger.info('Exclude following insufficient data at column {} '
                                'with high pass {}:\n{}'.format(col, hp, _exclude))
                self._report_reduction(_old_length,
                                       message='High pass filtering on {}'.format(col))

        for col1, col2, hp in cross_ref_high_passes:
            _old_length = self.data.shape[0]
            for c1, c2 in [(col1, col2), (col2, col1)]:
                self.data = self.data[self.data.groupby(c1)[c2].transform('count').ge(hp)]
            self._report_reduction(_old_length, message='Filter out cross reference on column {}-{}'.format(col1, col2))

        for col in filter_empties:
            self._assert_col(col)
            _old_length = self.data.shape[0]
            self.data = self.data.loc[~pd.isnull(self.data[col])]
            self._report_reduction(_old_length, message='Filter empty at {}'.format(col))

        for col, item in force_removals:
            self._assert_col(col)
            logger.warning('Force removal {} from col {}'.format(item, col))
            self.data = self.data.loc[self.data[col] != item]

        if shuffle_state:
            logger.info('Shuffle with state: {}'.format(shuffle_state))
            self.data = self.data.sample(frac=1, random_state=shuffle_state).reset_index(drop=True)

    def reproduce_check(self, length: Optional[int] = None):
        """Simple reproduce check for file length matches to original paper.

        Args:
            length: file info object for current inspecting file.
        """
        if BasicConfig.Reproduce_Lock:
            if not length:
                raise Exception('Reproduce check is in effect, length has to be provided.')
            if length != self.data.shape[0]:
                raise Exception('Fail reproduce check! Length {}, actual length {}'.format(
                    length, self.data.shape[0]
                ))
        logger.info('Passed simple reproduce check.')

    def data_report(self, cols=()):
        """Request a data report printed on the csv data.

        Args:
            cols: report unique count on values of the list of columns.
        """
        if not cols:
            cols = self.COLs
        for col in cols:
            self._assert_col(col)
            logger.info('Report: column {} has {} unique values.'.format(
                col, len(self.data[col].value_counts())
            ))

    def _load(self):
        if self._loaded:
            raise Exception('Already loaded.')
        for fi in self._files:
            if self.data is None:
                self.data = self._load_to_pandas(fi)
            else:
                df = self._load_to_pandas(fi)
                self.data = self.data.append(df, ignore_index=True)
        self._loaded = True

    def _load_to_pandas(self, file_info, cols=None):
        """Return as pandas data frame and check integrity of file.

        Args:
            file_info: file info object for the file.
            cols: all required column names. All column names are required to exist.
        """
        path = os.path.join(BasicConfig.Data_Path, file_info.file_name)

        # check sum
        if file_info.md5 is not None:
            if _md5_checksum(path) != file_info.md5:
                raise CSVIntegrityException('md5 fail to match: {}\nIf you have not change the file, this indicates '
                                            'damage of file during your download process'.format(file_info.md5))

        # load to pandas
        try:
            data = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError('Can not locate file at path: {}'.format(path))

        # check column integrity
        if cols is None:
            cols = self.COLs
        assert cols, 'Column names are required to ensure data integrity.'
        for c in cols:
            if c not in data.columns:
                raise CSVIntegrityException('Column {} is missing'.format(c))

        logger.info('Load successful: {}'.format(path))
        return data

    def __init__(self, files, debug_head=None):
        """CSV loader init.

        Args:
            files: list of file info object. Information about csv files for data inputs.
        """
        logger.info('... {} CSV LOADING ...'.format(debug_head))
        if type(self) == CSVData:
            raise Exception('CSVData is an abstract class.')
        self.data = None
        for f in files:
            assert type(f) is FileInfo, 'Need list of file info objects.'
        assert files, 'Input files not allow to be none'  # can not be none
        self._files = files
        self._loaded = False


class DialogHeadData(CSVData):
    COLs = 'sent_id,show_id,char_name,char_id,head_info,head_pos,token,token_par,position,is_stop,head_text'.split(',')

    def __init__(self, file_info: Optional[FileInfo] = BasicConfig.F_DialogHead):
        super().__init__([file_info], debug_head='Dialog Head')

        self._load()
        self.clean(force_removals=[('head_pos', 'ROOT'), ('head_pos', 'punct')],
                   high_passes=[('head_text', 5)], silent_high_pass=True, shuffle_state=578153)
        self.reproduce_check(file_info.reproduce_length)
        self.data_report(['sent_id', 'head_text'])


# usage
# d = DialogHeadData()
# print(d.data)


class DialogData(CSVData):
    COLs = 'show_id,char_name,char_id,dia1,dia2'.split(',')

    def __init__(self, file_info: Optional[FileInfo] = BasicConfig.F_Dialog):
        """Load dialog data.

        Args:
            file_info: file info object to dialog csv data.
        """
        super().__init__([file_info], debug_head='Dialog')
        self._load()

        # clean the data a bit
        # Note: the character Rebekah Mikaelson (l20098) exists in both shows: TheVampireDiaries and TheOriginals
        # To avoid possible break the character isolation in testing/training splits, we remove this character.
        self.clean(high_passes=[('char_id', 600,)],  # character lower than 600 dialogs are filtered out
                   filter_empties=['dia1', 'dia2'],  # not allow any dialog to be empty
                   force_removals=[('char_id', 'l20098')])  # see above comment

        self.reproduce_check(file_info.reproduce_length)
        self.data_report('show_id,char_name,char_id'.split(','))


# Sample usage
# d = DialogData()
# print(d.data)


class HLAData(CSVData):
    COLs = 'feature,char_id,work,char_name'.split(',')

    def char_note(self, char_id):
        """Return a human readable text to represent the character for the character ID.

        Args:
            char_id: id of the characters.

        Returns:
            A human readable string represents the character.
        """
        if char_id in self._char_note_cache:
            return self._char_note_cache[char_id]
        _select = self.data.loc[self.data.char_id == char_id]
        if len(_select) == 0:  # not enough feature
            self._char_note_cache[char_id] = 'Minor character (Not enough feature)'
        else:
            _first = _select.iloc[0]  # all rows are for the same character, first row will do
            self._char_note_cache[char_id] = '[{}] {} ({})'.format(_first.work, _first.char_name, char_id)
        return self._char_note_cache[char_id]

    def __init__(self, files=BasicConfig.F_HLAs, filter_duplicate=False):
        """Load HLA data.

        Args:
            filter_duplicate: default false. Choose to filter duplicate character -> feature entry.
                The reason this is false are two: matrix factor model handle this case,
                in addition, the duplication indicates repeat features recorded on page. Higher confidence is given.
                You can disable this for your experiment.
        """
        super().__init__(files, debug_head='HLA data')
        self._load()
        self.clean(cross_ref_high_passes=[('char_id', 'feature', 5)],
                   shuffle_state=718281)

        # notice we did not filter duplicate by default
        # if you use this, disable reproduce check.
        if filter_duplicate:
            self.data.drop_duplicates(['feature', 'char_id'], keep='first', inplace=True)
        self.reproduce_check(BasicConfig.F_HLAs_Length)
        self.data_report()

        # speed up char_note function operation.
        # key: character id, value: human readable note.
        self._char_note_cache = {}


# sample usage
# d = HLAData(filter_duplicate=True)
# print(d.data)
# print(d.char_note('l4390'))  # what is l4390? (Sheldon)
