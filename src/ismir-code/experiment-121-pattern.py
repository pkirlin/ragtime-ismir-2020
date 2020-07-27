
import data.PKDataset
import data.RagDataset
import scipy.stats
import pandas as pd
from collections import Counter
import functions

pkdata = data.PKDataset.PKDataset()
rags = data.RagDataset.RagDataset()

########
# Go through all pieces and make a data frame of:
#   number of measures/bars in piece
#   number of measures/bars that contain a tied syncopation
#   number of measures/bars that contain an untied syncopation
#   "" ..contain a tied/untied syncopation in an augmented fashion

rowslist = []
totalpieces = 0  # total pieces examined; used for a sanity check.

for title in pkdata.get_all_titles():
    series = pkdata.get_best_version_of_rag(title,
                                            accept_no_silence_at_start=True,  # modify this #####
                                            quant_cutoff=.95)
    if series is not None:
        totalpieces += 1
        fileid = series['fileid']

        # melbips & bassbips are lists with the binary onset patterns for the melody
        # and bass lines.  One pattern per measure, so the length of each list is the
        # number of measures in the piece.
        melbips = pkdata.get_melody_bips(fileid)
        bassbips = pkdata.get_bass_bips(fileid)

        # Double check that the number of measures in the melody matches number of measures in bass.
        assert len(melbips) == len(bassbips)

        d = {}
        d['fileid'] = fileid

        # initialize vars per song
        barcount = 0
        pattern_count_tied = 0
        pattern_count_tied_aug = 0
        pattern_count_untied = 0
        pattern_count_untied_aug = 0

        for x in range(0, len(melbips)):
            melbip = melbips[x]
            bassbip = bassbips[x]

            # check if we have a next bar, needed for patterns that cross a barline
            melbip_nextbar = 'XXXXXXXX'
            if x+1 < len(melbips):
                melbip_nextbar = melbips[x+1]

            # if both mel and bass are silent, don't count either
            if melbip == "00000000" and bassbip == "00000000":
                continue

            # occasionally we have a binary onset pattern that got miscalculated from the MIDI
            # data.  We ignore/skip these.  These usually happened due to weird MIDI files that
            # music21 parsed incorrectly.
            if len(melbip) != 8:
                continue

            # we have a good melody bar
            barcount += 1

            # Untied, first half of bar
            if melbip[0:4] == '1101':# or melbip == '10100010':
                pattern_count_untied += 1

            # Untied, second half of bar
            if melbip[4:8] == '1101':
                pattern_count_untied += 1

            # Tied, middle of bar
            if melbip[2:6] == '1101':
                pattern_count_tied += 1

            # Tied, crossing this bar into the next one.
            if melbip[6:8] == '11' and melbip_nextbar[0:2] == '01':
                pattern_count_tied += 1

            # Untied, augmented (so takes up entire bar).
            if melbip == '10100010':
                pattern_count_untied_aug += 1

            # Tied, augmented, so takes up 2nd half of this bar and half of next bar.
            if melbip[4:8] == '1010' and melbip_nextbar[0:4] == '0010':
                pattern_count_tied_aug += 1

        # end of song
        d['barcount'] = barcount
        d['tied'] = pattern_count_tied
        d['untied'] = pattern_count_untied
        d['tied_aug'] = pattern_count_tied_aug
        d['untied_aug'] = pattern_count_untied_aug

        # add in some extra stats to look at
        d['year_cat'] = pkdata.get_year_as_category(fileid)
        d['composer'] = pkdata.get_composer(fileid)
        d['rtctype'] = pkdata.get_rtc_type(fileid)
        d['ts'] = pkdata.get_music21_time_signature_clean(fileid)

        rowslist.append(d)

# We now have a dataframe with number of 121 syncopation counts, plus year (as a category),
# composer, rtctype (ragtime compendium category identifier code), and ts = time signature.

###############
# Slice and dice based on early/late ragtime periods, and also the whole ragtime era ("earlylate")
# and the "modern" era.

df = pd.DataFrame(rowslist)
df['untied_pct'] = df['untied']/df['barcount']
df['tied_pct'] = df['tied']/df['barcount']
df['untied_aug_pct'] = df['untied_aug']/df['barcount']
df['tied_aug_pct'] = df['tied_aug']/df['barcount']

df_early = df[df['year_cat'] == '1890-1901']
df_late = df[df['year_cat'] == '1902-1919']
df_earlylate = df[ (df['year_cat'] == '1890-1901') | (df['year_cat'] == '1902-1919') ]
df_modern = df[df['year_cat'] == '>1919']

df_joplin = df[df['composer']=='Joplin, Scott']
df_scott = df[df['composer']=='Scott, James']
df_lamb = df[df['composer']=='Lamb, Joseph F.']

print("Counts:")
print("Number of early rags:      ", len(df_early))
print("Number of late rags:       ", len(df_late))
print("Number of early+late rags: ", len(df_earlylate))
print("Number of modern rags:     ", len(df_modern))
# Sanity check from paper: 110 early, 582 late, 692 combined, 362 modern (makes 1058 total).

print("Comparing tied vs untied:")
print("Early rags: untied/tied:", df_early['untied_pct'].mean(), df_early['tied_pct'].mean())
print("Late  rags: untied/tied:", df_late['untied_pct'].mean(), df_late['tied_pct'].mean())
print("Early+late: untied/tied:", df_earlylate['untied_pct'].mean(), df_earlylate['tied_pct'].mean())
print("Modern rgs: untied/tied:", df_modern['untied_pct'].mean(), df_modern['tied_pct'].mean())
# Sanity check from paper:
# Early rags: untied/tied: 0.19244127939945385 0.11808343612589921
# Late  rags: untied/tied: 0.1389819188471764 0.2353624944445536
# Early+late: untied/tied: 0.14747979407947479 0.21671986956731087
# Modern rgs: untied/tied: 0.18353242780626713 0.29421524925766807

##################
# Run tests

test_early_late_untied = scipy.stats.mannwhitneyu(df_early['untied_pct'], df_late['untied_pct'], alternative='two-sided')
test_early_late_tied   = scipy.stats.mannwhitneyu(df_early['tied_pct'],   df_late['tied_pct'],   alternative='two-sided')

test_old_modern_untied = scipy.stats.mannwhitneyu(df_earlylate['untied'], df_modern['untied_pct'], alternative='two-sided')
test_old_modern_tied   = scipy.stats.mannwhitneyu(df_earlylate['tied_pct'],   df_modern['tied_pct'],   alternative='two-sided')

test_joplin_scott_untied = scipy.stats.mannwhitneyu(df_joplin['untied_pct'], df_scott['untied_pct'], alternative='two-sided')
test_joplin_lamb_untied = scipy.stats.mannwhitneyu(df_joplin['untied_pct'], df_lamb['untied_pct'], alternative='two-sided')
test_scott_lamb_untied = scipy.stats.mannwhitneyu(df_scott['untied_pct'], df_lamb['untied_pct'], alternative='two-sided')

test_joplin_scott_tied = scipy.stats.mannwhitneyu(df_joplin['tied_pct'], df_scott['tied_pct'], alternative='two-sided')
test_joplin_lamb_tied = scipy.stats.mannwhitneyu(df_joplin['tied_pct'], df_lamb['tied_pct'], alternative='two-sided')
test_scott_lamb_tied = scipy.stats.mannwhitneyu(df_scott['tied_pct'], df_lamb['tied_pct'], alternative='two-sided')

print("Joplin tied/untied:", df_joplin['tied_pct'].mean(), df_joplin['untied_pct'].mean())
print("Scott  tied/untied:", df_scott['tied_pct'].mean(), df_scott['untied_pct'].mean())
print("Lamb   tied/untied:", df_lamb['tied_pct'].mean(), df_lamb['untied_pct'].mean())