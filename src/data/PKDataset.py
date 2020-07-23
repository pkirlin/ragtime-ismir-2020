
import pandas as pd
import data.RagDataset
from pathlib import Path
from typing import List

class PKDataset(object):
    PK_COMPENDIUM_CSV = (Path(__file__).parent / "../../data/processed/pk-compendium2-new.csv").resolve()  # my data

    # set of patterns for 2/4 onset patterns by 16th notes, and 2/2 & 4/4 patterns in 8th notes
    # (so each pattern has 16 bits)
    PK_BINARY_ONSET_PATTERNS_CSV = (Path(__file__).parent / "../../data/processed/bitpatterns.csv").resolve()

    # set of patterns for 2/2 and 4/4 including 16th notes, so each pattern has 16 bits
    PK_BINARY_ONSET_PATTERNS16_CSV = (Path(__file__).parent / "../../data/processed/16bitpatterns.csv").resolve()

    def __init__(self):
        self.df = pd.read_csv(PKDataset.PK_COMPENDIUM_CSV)
        self.bip_df = pd.read_csv(PKDataset.PK_BINARY_ONSET_PATTERNS_CSV, usecols=[1, 2, 3], index_col='fileid')
        self.bip16_df = pd.read_csv(PKDataset.PK_BINARY_ONSET_PATTERNS16_CSV, usecols=[1,2,3], index_col='fileid')

    def get_best_version_of_rag(self, title, accept_no_silence_at_start=None, quant_cutoff=None):
        """
        Return the fileid for the "best" version of a rag.  In order of preference:
        - Filter by song title first.
        - Remove "do not use" songs.
        - If we have a "true time signature, filter by that.
        - Otherwise, take the mode time signature for all matching songs.
        -   If there's a tie, prefer 2/4 over 4/4, 4/4 over 2/2.
        - Then we filter out silence at beginning if accept_silence=False
        - Then we sort by quant_cutoff and take the best match.
        :param songtitle:
        :return:
        """

        if accept_no_silence_at_start is None:
            raise Exception("accept_no_silence_at_start must be True or False.")

        mydf = self.df
        mydf = mydf[mydf['title']==title]
        mydf = mydf[mydf['do_not_use'].isna()]

        # temporarliy remove 3/4 timesigs...these shouldn't be here
        mydf = mydf[mydf['ts_m21'] != "['3/4@0.0']"]

        if len(mydf) == 0:  # no files usable
            return None

        # Figure out if we have a "true" time sig in the PK data file.  Usually obtained from the score.
        tempdf = mydf[mydf['true_ts'].notna()]
        if len(tempdf) != 0:  # We have a true time sig.
            #print("We have a hard-coded true TS")
            true_ts = tempdf['true_ts'].iloc[0]
            mydf = mydf[ mydf['ts_m21'].str[2:5]==true_ts ]
        else:  # get the counts of the time sigs
            tempdf = mydf['ts_m21'].str[2:5].mode()
            if len(tempdf) == 1:
                #print("Single-mode TS")
                # easy, one most likely time sig, use it
                true_ts = tempdf.iloc[0]
                mydf = mydf[mydf['ts_m21'].str[2:5] == true_ts]
            else:
                #print("Multi-mode TS")
                if (tempdf=='2/4').any():
                    true_ts = '2/4'
                elif (tempdf=='4/4').any():
                    true_ts = '4/4'
                else:
                    print(tempdf)
                    true_ts = tempdf.iloc[0]
                mydf = mydf[mydf['ts_m21'].str[2:5] == true_ts]

        # if this is False, we remove songs with 0 silence at start.
        # Reasoning is b/c songs with no silence at beginning may have an upbeat and we don't
        # know if the musicxml is aligned with the time sig correctly.
        if not accept_no_silence_at_start:
            mydf = mydf[mydf['silence_beats_m21']>0]

        if quant_cutoff is not None:
            mydf = mydf[ mydf['onset_pct_m21'] >= quant_cutoff ]

        # now sort by quantized percent from music21, and take the file with the largest value.
        mydf = mydf[mydf['onset_pct_m21'] == (mydf['onset_pct_m21'].max())]

        # there should be only one value in mydf, unless there are multiple top quantized files
        if len(mydf) == 0:
            return None  # No versions of this rag with non-zero startup, presumably
        else:
            return mydf.iloc[0]

    def get_all_titles(self):
        """
        Return a list of all titles in the DB.  Note that some titles may not have very good midi renditions.
        :return:
        """
        return list(self.df['title'].unique())

    def get_melody_part_number(self, fileid) -> int:
        """
        Return the upper (treble, melody) part for a fileid.  Returns either 0 or 1.
        :param fileid:
        :return:
        """

        # look up the fileid and figure out which part has the melody
        # by examining the average pitch of each part

        series = self.df[self.df['fileid']==fileid].iloc[0]  # we know there will be one match
        p0pitch = series['part0_avgpitch']
        p1pitch = series['part1_avgpitch']
        if p0pitch > p1pitch:
            return 0
        else:
            return 1

    def get_melody_bips(self, fileid) -> List:
        """
        Returns the melody binary onset patterns for this fileid.  Patterns are always groups
        of 8, which means for 2/4 time signatures, each bit is a 16th note.  For 2/2 and 4/4,
        bits are 8th notes.
        """
        melpart_num = self.get_melody_part_number(fileid)
        if melpart_num == 0:
            return eval(self.bip_df.loc[fileid, 'part0list'])
        else:
            return eval(self.bip_df.loc[fileid, 'part1list'])

    def get_bass_bips(self, fileid) -> List:
        """
        Returns the bass binary onset patterns for this fileid.  Patterns are always groups
        of 8, which means for 2/4 time signatures, each bit is a 16th note.  For 2/2 and 4/4,
        bits are 8th notes.
        """
        melpart_num = self.get_melody_part_number(fileid)
        if melpart_num == 0:
            return eval(self.bip_df.loc[fileid, 'part1list'])
        else:
            return eval(self.bip_df.loc[fileid, 'part0list'])

    def get_melody_bips16(self, fileid) -> List:
        """
        Returns the melody binary onset patterns for this fileid, which assumes this is a 2/2
        or 4/4 song, and we want the 16th note binar onset pattern.
        """
        melpart_num = self.get_melody_part_number(fileid)
        if melpart_num == 0:
            return eval(self.bip16_df.loc[fileid, 'part0list'])
        else:
            return eval(self.bip16_df.loc[fileid, 'part1list'])

    def get_bass_bips16(self, fileid) -> List:
        """
        Returns the bass binary onset patterns for this fileid, which assumes this is a 2/2
        or 4/4 song, and we want the 16th note binar onset pattern.
        """
        melpart_num = self.get_melody_part_number(fileid)
        if melpart_num == 0:
            return eval(self.bip16_df.loc[fileid, 'part1list'])
        else:
            return eval(self.bip16_df.loc[fileid, 'part0list'])

    def get_music21_time_signature(self, fileid) -> str:
        """
        Returns '['2/4@0:0']' or the equivalents for 4/4 or 2/2.
        :param fileid:
        :return:
        """
        series = self.df[self.df['fileid'] == fileid].iloc[0]  # we know there will be one match
        return series['ts_m21']

    def get_music21_time_signature_clean(self, fileid) -> str:
        """
        Returns '2/4' or 4/4 or 2/2' as a string.
        :param fileid:
        :return:
        """
        series = self.df[self.df['fileid'] == fileid].iloc[0]  # we know there will be one match
        ts =  series['ts_m21']
        return ts[2:5]

    def get_composer(self, fileid) -> str:
        series = self.df[self.df['fileid'] == fileid].iloc[0]  # we know there will be one match
        return series['composer']

    def get_rtc_type(self, fileid) -> str:
        series = self.df[self.df['fileid'] == fileid].iloc[0]  # we know there will be one match
        return series['rtctype']

    def has_year_as_number(self, fileid) -> bool:
        series = self.df[self.df['fileid']==fileid].iloc[0]  # we know there will be one match

        return str(series['year']).isdigit()

    def get_year_as_number(self, fileid) -> int:
        series = self.df[self.df['fileid'] == fileid].iloc[0]  # we know there will be one match
        if str(series['year']).isdigit():
            return int(series['year'])
        else:
            raise Exception("can't convert to integer: " + series['year'])

    def get_year_as_category(self, fileid) -> str:
        """
        Returns the year category from the four categories:
          <1890, 1890-1901, 1902-1919, >1919,
        :param fileid:
        :return:
        """
        series = self.df[self.df['fileid'] == fileid].iloc[0]  # we know there will be one match
        if str(series['year']).isdigit():
            year = int(series['year'])
            if year < 1890:
                return "<1890"
            elif year < 1902:
                return "1890-1901"
            elif year < 1920:
                return "1902-1919"
            else:
                return ">1919"
        else:
            return series['year_alt']



