"""
Sample code for 0.1 version release.
This code write five fold cross validation data to your home directory.
Use official five fold data split and five target characters we used in research.
"""
import os
from logzero import logger
from core import DataManager

# change this
WRITE_PATH = os.path.expanduser('~/aloha_data')

assert not os.path.exists(WRITE_PATH), 'folder already exists (did you run already? delete the previous folder)'
os.mkdir(WRITE_PATH)

# fold 1: sheldon
logger.info('------ FOLD 1 -------')
dm = DataManager('l4390', test_shows=['GreysAnatomy', 'TheBigBangTheory', 'TheVampireDiaries', 'HowIMetYourMother',
                                       'Smallville', 'Seinfeld', 'GilmoreGirls', 'DawsonsCreek'])
dm.write(os.path.join(WRITE_PATH, 'fold1'))

# fold 2: Picard
logger.info('------ FOLD 2 -------')
dm = DataManager('l25431', test_shows=['SonsOfAnarchy', 'Salem', 'TheMentalist', 'VeronicaMars', 'TrueBlood',
                                       'StarTrek', 'Merlin2008', 'OneTreeHill'])
dm.write(os.path.join(WRITE_PATH, 'fold2'))

# fold 3: Monica
logger.info('------ FOLD 3 -------')
dm = DataManager('l10692', test_shows=['OnceUponATime', 'QueerAsFolk', 'Buffyverse', 'Supernatural', 'Charmed1998',
                                       'Bones', 'TheOriginals', 'Friends'])
dm.write(os.path.join(WRITE_PATH, 'fold3'))

# fold 4: Grissom
logger.info('------ FOLD 4 -------')
dm = DataManager('l7484', test_shows=['TheOC', 'Roswell', 'Alias', 'CSIVerse', 'MyLittlePonyFriendshipIsMagic',
                                      'TheLWord', 'TheSecretLifeOfTheAmericanTeenager'])
dm.write(os.path.join(WRITE_PATH, 'fold4'))

# fold 5: Marge
logger.info('------ FOLD 5 -------')
dm = DataManager('a15821', test_shows=['NCIS', 'TheSimpsons', 'TeenWolf', 'AdventureTime', 'Futurama',
                                       'TheOfficeUS', 'DoctorWho'])
dm.write(os.path.join(WRITE_PATH, 'fold5'))
