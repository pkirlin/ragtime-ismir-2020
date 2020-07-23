import shutil
import os
import pandas as pd
import logging
import sqlite3
import pickle
import pretty_midi
from pathlib import Path
import music21

class RagDataset(object):
    MASTER_RAG_DIR = (Path(__file__).parent / "../../data/raw/rag-master").resolve()  # the directory where the .mid files are

    MUSICXML_RAG_DIR = (Path(__file__).parent / "../../data/processed/rag-musicxml").resolve()  # the directory where the .mid files are

    DATABASE_CSV_FILENAME = (Path(__file__).parent / "../../data/processed/ragfiles.csv").resolve()  # pandas csv that just holds the midifile
    # fileids, filenames, and boolean values for whether they can be opened using various readers
    # such as pretty_midi, music21, etc.

    # CSV that holds information about the Ragtime dataset, including titles, artists, etc.
    # We semi-algorithmically (a little by hand) aligned this with the filenames of the MIDIs.
    COMPENDIUM_CSV_FILENAME = (
                Path(__file__).parent / "../../data/processed/compendium.csv").resolve()

    PRETTY_MIDI_SQL_FILENAME = (Path(__file__).parent / "../../data/processed/pretty_midi.sqlite").resolve()
    # this database will the pickled pretty_midi data.

    # Globals below are for accessing the pretty_midi files from cache when we need to.
    # pretty_midi_cache: if we have previously accessed a pretty_midi file, it will be cached here.
    # maps fileids -> Pretty_midi objects.
    PRETTY_MIDI_CACHE = {}

    # Database connection---not opened until needed.
    conn = None
    cursor = None
    myinstance = None

    def __init__(self):
        self.master_df = None

        # Load in the CSV if we can find it.
        if os.path.exists(RagDataset.DATABASE_CSV_FILENAME):
            self.master_df = pd.read_csv(RagDataset.DATABASE_CSV_FILENAME, index_col="fileid")

        # Load in the Ragtime Compendium CSV if we can find it.
        if os.path.exists(RagDataset.COMPENDIUM_CSV_FILENAME):
            self.compendium_df = pd.read_csv(RagDataset.COMPENDIUM_CSV_FILENAME, index_col="rtcid")

        # Cache this so we can get it from wherever and not have to pass it around
        RagDataset.myinstance = self

    @staticmethod
    def getInstance():
        assert RagDataset.myinstance is not None
        return RagDataset.myinstance

    def __del__(self):
        # Disconnect sqlite database.
        if RagDataset.conn is not None:
            RagDataset.conn.close()

        # Save out the CSV file.
        #if self.master_df is not None:
        #    self.master_df.to_csv(RagDataset.DATABASE_CSV_FILENAME, index_label="fileid")

    def save_csv(self) -> None:
        """
        Save out the CSV file.  Done automatically when object goes out of scope, but
        when using interactively, nice to have this.
        :return: None
        """
        if self.master_df is not None:
            self.master_df.to_csv(RagDataset.DATABASE_CSV_FILENAME, index_label="fileid")

    def get_all_fileids(self) -> pd.Index:
        """
        Return all fileids in the database.
        :return:
        """
        if self.master_df is None:
            print("CSV not found")
            return

        return self.master_df.index

    def get_all_pretty_midi_fileids(self) -> pd.Index:
        """
        Return all fileids in the database that can be opened with pretty_midi.
        :return:
        """
        if self.master_df is None:
            print("CSV not found")
            return

        return self.master_df[self.master_df['pretty_midi']].index

    def load_musicxml(self, fileid: str) -> music21.stream.Score:
        """
        Load a musicxml file given a fileid.
        :param fileid:
        :return:
        """

        filename = RagDataset.MUSICXML_RAG_DIR / (fileid + ".musicxml")

        m21file = music21.converter.parse(filename)
        return m21file

    def load_musicxml_merged(self, fileid: str) -> music21.stream.Score:
        """
        Load a musicxml file given a fileid, and merge multi-staff parts together.
        :param fileid:
        :return:
        """

        filename = RagDataset.MUSICXML_RAG_DIR / (fileid + ".musicxml")

        m21file = music21.converter.parse(filename)

        # This code merges multi-staff parts.
        parts = m21file.parts
        if len(parts) != 2:
            #print(fileid, "has more than 2 parts:", len(parts))
            #print(list(parts))
            # loop through part staffs.  Hope there's only 2 staffs for each part.
            prevPartStaff = None
            for part in parts:
                if isinstance(part, music21.stream.PartStaff):
                    #print("\tPartStaff:", part.partName, part.id)
                    if prevPartStaff is None:
                        prevPartStaff = part
                    else:
                        #print("Merging", prevPartStaff, "and", part)
                        assert prevPartStaff.id[:-1] == part.id[:-1]
                        prevPartStaff.mergeElements(part)
                        m21file.remove(part)
                        prevPartStaff = None

        return m21file

    def load_pretty_midi(self, fileid: str) -> pretty_midi.PrettyMIDI:
        """
        Load a pretty_midi midifile object from the sqlite db or from cache.
        :param fileid:
        :return:
        """
        if fileid in RagDataset.PRETTY_MIDI_CACHE:
            return RagDataset.PRETTY_MIDI_CACHE[fileid]

        if RagDataset.conn is None:
            RagDataset.conn = sqlite3.connect(RagDataset.PRETTY_MIDI_SQL_FILENAME)

        if RagDataset.cursor is None:
            RagDataset.cursor = RagDataset.conn.cursor()

        row = RagDataset.cursor.execute('SELECT * FROM pretty_midi WHERE fileid=?', [fileid]).fetchone()
        fileid = row[0]
        midifile = pickle.loads(row[1])
        RagDataset.PRETTY_MIDI_CACHE[fileid] = midifile

        return midifile

    def precache_pretty_midi(self) -> None:
        """
        Pre-cache the pretty_midi files so they'll load faster when we need them.

        :return:
        """
        if RagDataset.conn is None:
            RagDataset.conn = sqlite3.connect(RagDataset.PRETTY_MIDI_SQL_FILENAME)

        if RagDataset.cursor is None:
            RagDataset.cursor = RagDataset.conn.cursor()

        rows = RagDataset.cursor.execute('SELECT * FROM pretty_midi')
        for row in rows:
            fileid = row[0]
            midifile = pickle.loads(row[1])
            RagDataset.PRETTY_MIDI_CACHE[fileid] = midifile

    def get_rtc_info_by_rtcid(self, rtcid):
        """
        Return information from the RTC given a string or int rtcid.
        :param rtcid:
        :return:
        """
        # convert id to string since that's how they're stored in the compendium (b/c there are also 'new's for some reason)
        rtcid = str(rtcid)
        try:
            return self.compendium_df.loc[rtcid]
        except KeyError:
            raise ValueError(f'RTCID {rtcid} does not exist.')

    def get_rtc_info_by_fileid(self, fileid):
        """
        Return information from the RTC given a fileid (identifier from Phil's database CSV).
        :param fileid:
        :return:
        """

        # get rtcid by fileid

        rtcid = self.master_df.loc[fileid, 'rtcid']
        if rtcid == -1:
            raise ValueError("No RTC id found for " + str(fileid))
        else:
            return self.get_rtc_info_by_rtcid(rtcid)


    @staticmethod
    def _get_unique_id_for_ragfile(relfilename: str) -> str:
        """
        Given a relative path + filename, generate the unique ID for this file.
        """

        uniqid = relfilename

        # Remove special characters.
        uniqid = uniqid.replace(' ', '_')
        uniqid = uniqid.replace('/', '_')
        uniqid = uniqid.replace('-', '_')
        uniqid = uniqid.replace("'", "")
        uniqid = uniqid.replace("&", "")
        uniqid = uniqid.replace("!", "")
        uniqid = uniqid.replace("#", "")
        uniqid = uniqid.replace(",", "")
        uniqid = uniqid.replace(";", "")
        uniqid = uniqid.replace("(", "")
        uniqid = uniqid.replace(")", "")
        uniqid = uniqid.replace("[", "")
        uniqid = uniqid.replace("]", "")
        uniqid = uniqid.replace("_-_", "_")
        uniqid = uniqid.replace("__", "_")
        uniqid = uniqid.replace("__", "_")
        uniqid = uniqid.replace("__", "_")
        uniqid = uniqid.lower()

        assert uniqid.endswith('.mid')  # make sure we end with .mid

        uniqid = uniqid[:-4]  # chop off the .mid part.

        if uniqid.endswith('.'):  # make sure there wasn't a second extension before the .mid
            print(uniqid)
            return

        return uniqid

    #### FUNCTIONS THAT CHANGE DATABASE ARE BELOW HERE.
    # Generally these should only be run to start the data processing step.

    def create_master_dataframe(self, force=False):
        """
        Create the master dataframe of ragfiles.  Do this by getting a list of all files in the
        master directory, and getting the uniq_id for each one.  Put into a pandas dataframe.

        :return: The new pandas dataframe, with two columns: one for the fileid string, and one
        for the filename.
        """

        if self.master_df is not None:
            print("Not creating master dataframe, seems to already exist.  Can force if desired.")
            return

        ids = []
        filenames = []
        map = {}

        count = 0
        for root, dirs, files in os.walk(RagDataset.MASTER_RAG_DIR, topdown=True):
            for name in files:
                fullfilename = os.path.join(root, name)

                if not fullfilename.endswith('.mid'): continue  # skip non-midi files

                uniq_id = RagDataset._get_unique_id_for_ragfile(fullfilename)
                if uniq_id in map:
                    print(uniq_id, fullfilename, map[uniq_id])
                    return  # should never happen
                map[uniq_id] = fullfilename
                ids.append(uniq_id)

                filenames.append(fullfilename)

        df = pd.DataFrame(data=filenames, index=ids, columns=['filename'])
        df.index.name = "fileid"
        return df
