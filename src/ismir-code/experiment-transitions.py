'''
Code generating content from section 4.3 of the paper.
'''


import data.PKDataset
import data.RagDataset
import scipy.stats
import pandas as pd
from collections import Counter
#import functions
from collections import defaultdict
import random

def compute_lhl(bar, nextbar):
    """
    Compute the LHL for a single bar of music.  We assume this bar has 8 onsets.
    We need the next bar to be able to compute syncopation for the 2nd half of this bar.
    Example:
           1  1  0  1  0  0  0  1  |  0 ...
           <---   this bar     --> |  <- next bar ... ->
    weight 0 -3 -2 -3 -1 -3 -2 -3  |  0
               +1    +2           +3  => 6 total syncopation

    :param bar:
    :return:
    """

    assert len(bar) == 8

    # determine to which depth we must build the tree
    # if there are no notes...return Nan
    if bar == '00000000':
        return 0
    # Note that there's probably a much better way to do this
    splits = list(bar)    # split the bar into 16 pieces
    weights = [0, -3, -2, -3, -1, -3, -2, -3]
    # see if we should merge them
    if splits[1] == '0' and splits[3] == '0' and splits[5] == '0' and splits[7] == '0':
        splits = [splits[0], splits[2], splits[4], splits[6] ]
        weights = [0, -2, -1, -2]
        if splits[1] == '0' and splits[3] == '0':
            splits = [ splits[0], splits[2] ]
            weights = [0, -1]
            if splits[1] == '0':
                splits = [ splits[0] ]
                weights = [0]

    #print(weights, splits)

    # now we find syncopation pairs (consecutive pairs of a ONE then a ZERO where the weight
    # on the ZERO (rest or tie) is higher than the weight on the ONE (an onset).
    # To make this easier, we make a *nine* element pattern, adding in the downbeat
    # of the next measure (which might be X if it's the end of the piece).
    splits.append(nextbar[0])
    weights.append(0)
    s = 0
    for n1, n2, w1, w2 in zip(splits[0:8], splits[1:9], weights[0:8], weights[1:9]):
        # iterate through consecutive pairs of note/rests (1/0s)
        #print(n1, n2, w1, w2)
        if n1 == '1' and n2 == '0' and w1 < w2:
            s += (w2 - w1)
    return s

def total_bips(df):
    """
    Given a dataframe, go into the dictionary in the bipcount column
    and make a single dictionary merging all the bip counts together.
    (Note that this will be skewed towards longer pieces).
    :param df:
    :return:
    """
    grandbips = Counter()
    for diction in df['bipcount']:
        grandbips += diction
    return grandbips

def average_bips(series):
    """
    Given a dataframe with bipcounts that have been normalized per bar,
    average all the frequencies across all pieces.  (so just total up and div by # of pieces).
    """
    avgbips = Counter()
    for diction in series:
        avgbips += diction
    for k, v in avgbips.items():
        avgbips[k] = v/len(series)
    return avgbips

def normalize_bips(df) -> pd.Series:
    """
    Given a dataframe, go into the bipcount dictionary and *separately for each piece*,
    normalize the bips so we get the frequency per bar of each pattern.
    """
    def frequentize_per_bar(row):
        bipcount_new = Counter()
        bipcount = row['bipcount']
        totalbars = row['barcount']
        for k, v in bipcount.items():
            #print(k, v, totalbars)
            bipcount_new[k] = v/totalbars
        #print(bipcount_new)
        return bipcount_new
    return df.apply(frequentize_per_bar, axis=1)

def normalize_bip_trans(df) -> pd.Series:
    """
    Given a dataframe, go into the biptrans dictionary and *separately for each piece*,
    normalize the bips so we get the frequency per bar of each pattern transition.
    """
    def frequentize_per_bar(row):
        bipcount_new = Counter()
        bipcount = row['biptrans']
        totalbars = row['barcount']
        for k, v in bipcount.items():
            #print(k, v, totalbars)
            bipcount_new[k] = v/totalbars
        #print(bipcount_new)
        return bipcount_new
    return df.apply(frequentize_per_bar, axis=1)


pkdata = data.PKDataset.PKDataset()
rags = data.RagDataset.RagDataset()
rowslist = []

for title in pkdata.get_all_titles():
    series = pkdata.get_best_version_of_rag(title,
                                            accept_no_silence_at_start=True,  # modify this #####
                                            quant_cutoff=.95)
    if series is not None:
        fileid = series['fileid']
        melbips = pkdata.get_melody_bips(fileid)
        bassbips = pkdata.get_bass_bips(fileid)

        assert len(melbips) == len(bassbips)

        d = {}
        d['fileid'] = fileid

        # initialize vars per song
        barcount = 0
        bip_counter = Counter()
        bip_trans_counter = Counter()

        lhl_list = []
        bip_list = []

        for x in range(0, len(melbips)):
            melbip = melbips[x]
            bassbip = bassbips[x]

            # check if we have a next bar, needed for tied 121 pattern across barline
            melbip_nextbar = 'XXXXXXXX'
            if x+1 < len(melbips):
                melbip_nextbar = melbips[x+1]

            # if both mel and bass are silent, don't count either
            if melbip == "00000000" and bassbip == "00000000":
                continue

            if len(melbip) != 8:
                continue

            # we have a good melody bar
            barcount += 1

            # binary onset pattern counting
            bip_counter[melbip] += 1
            bip_trans_counter[melbip + '-' + melbip_nextbar] += 1

            # lhl counting & bip counting
            lhl_list.append(compute_lhl(melbip, melbip_nextbar))
            bip_list.append(melbip)


        # end of song
        assert barcount == len(lhl_list)
        assert barcount == sum(bip_counter.values())
        assert barcount == sum(bip_trans_counter.values())

        # make LHL transitions
        d['lhl_trans'] = [str(i) + '-' + str(j) for i, j in zip(lhl_list, lhl_list[1:] + ['X'])]

        # make more stats
        d['barcount'] = barcount
        d['bipcount'] = bip_counter
        d['biptrans'] = bip_trans_counter
        d['bip_list'] = bip_list
        d['lhl_list'] = lhl_list
        d['lhl_sum'] = sum(lhl_list)
        d['lhl_pos_count'] = len([x for x in lhl_list if x > 0])
        d['lhl_pos_sum'] = sum([x for x in lhl_list if x > 0])

        # add in some extra stats to look at
        d['year_cat'] = pkdata.get_year_as_category(fileid)
        d['composer'] = pkdata.get_composer(fileid)
        d['rtctype'] = pkdata.get_rtc_type(fileid)
        d['ts'] = pkdata.get_music21_time_signature_clean(fileid)

        rowslist.append(d)

df = pd.DataFrame(rowslist)
df = df.set_index('fileid')

df['lhl_avg'] = df['lhl_sum']/df['barcount']
df['lhl_pos_avg'] = df['lhl_pos_sum']/df['lhl_pos_count']
df['lhl_pct_bars_sync'] = df['lhl_pos_count']/df['barcount']
#
# df_early = df[df['year_cat'] == '1890-1901']
# df_late = df[df['year_cat'] == '1902-1919']
# df_earlylate = df[ (df['year_cat'] == '1890-1901') | (df['year_cat'] == '1902-1919') ]
# df_modern = df[df['year_cat'] == '>1919']
#
# df_joplin = df[df['composer']=='Joplin, Scott']
# df_scott = df[df['composer']=='Scott, James']
# df_lamb = df[df['composer']=='Lamb, Joseph F.']
#
# bip_dist = pd.Series(average_bips(normalize_bips(df)))
# bip_dist_early = pd.Series(average_bips(normalize_bips(df_early)))
# bip_dist_late = pd.Series(average_bips(normalize_bips(df_late)))
# bip_dist_earlylate = pd.Series(average_bips(normalize_bips(df_earlylate)))
# bip_dist_modern = pd.Series(average_bips(normalize_bips(df_modern)))
#
# bip_dist_joplin = pd.Series(average_bips(normalize_bips(df_joplin)))
# bip_dist_scott = pd.Series(average_bips(normalize_bips(df_scott)))
# bip_dist_lamb = pd.Series(average_bips(normalize_bips(df_lamb)))
#
# df_big3 = df[(df['composer']=='Joplin, Scott') | (df['composer']=='Scott, James') | (df['composer']=='Lamb, Joseph F.')]
# df_nonbig3 = df[(df['composer']!='Joplin, Scott') & (df['composer']!='Scott, James') & (df['composer']!='Lamb, Joseph F.')]
#
# df_big3_late = df_big3[df_big3['year_cat']=='1902-1919']
# df_nonbig3_late = df_nonbig3[df_nonbig3['year_cat']=='1902-1919']

# print some stats
# print(f'Have {len(df)} pieces')
# print("total bars:", df['barcount'].sum())
# print("total bars with lhl>0:", df['lhl_pos_count'].sum())
# print("...which is a % of", df['lhl_pos_count'].sum()/df['barcount'].sum())
# print("avg LHL")

# # make 2 expanded dataframes containing one row per measure/one transition per row
# rowslist = []
# for num, row in df.iterrows():
#     fileid = row.name
#     bip_list = row['bip_list']
#     lhl_list = row['lhl_list']
#     assert len(bip_list)==len(lhl_list)
#     for barnum, melbip, lhl in zip(range(0, len(bip_list)), bip_list, lhl_list):
#         D = {}
#         D['fileid'] = fileid
#         D['barnum'] = barnum+1
#         D['melbip'] = melbip
#         D['lhl'] = lhl
#         rowslist.append(D)
# df_expanded = pd.DataFrame(rowslist)
#
# print("Confirming from above, total measures=", len(df_expanded))
# print("number of those measures with lhl>0", len(df_expanded[df_expanded['lhl']>0]))
# print('lhl stats whole corpus:')
# print( df_expanded['lhl'].describe() )
# print("lhl>0 stats whole corpus:")
# print( df_expanded[df_expanded['lhl']>0]['lhl'].describe() )

# Some sanity checks:
print(f'Have {len(df)} pieces')
print("total bars:", df['barcount'].sum())
print("total bars with lhl>0:", df['lhl_pos_count'].sum())

##########
# LHL data collected.  Move to next step.

# Sanity check---make sure the bip data and LHL
# data match in length.
# To be clear, bip_list and lhl_list should match the number of
# measures in each piece.
for num, data in df[['bip_list', 'lhl_list']].iterrows():
    (lst1, lst2) = data
    assert len(lst1) == len(lst2)

# This function transforms LHL numbers (which are between 0 and 7 here)
# into bins.  0 is binned by itself (doesn't change).  1 and 2
# are binned together as "10".  Anything higher than 2 is binned
# together as "100."
def bin_lhl_list(lst):
    lst2 = lst.copy()
    for i in range(0, len(lst)):
        if lst2[i] == 1 or lst2[i] == 2:
            lst2[i] = 10
        elif lst2[i] > 2:
            lst2[i] = 100
    return lst2

# lists_per_piece is now a data frame with one list per piece.
# Specifically, each list contains a 0/10/100 value corresponding
# low/medium/high LHL values per measure.
lists_per_piece=df['lhl_list'].apply(bin_lhl_list)

# Number of measures per piece on average.
# Number of transitions per piece, which must be one less than # of measures.
avg_piece_len = df['barcount'].mean()
avg_piece_trans = avg_piece_len - 1
print("Avg piece length in measures: ", avg_piece_len)
print("Avg piece # of transitions", avg_piece_trans)

### Sanity check.  Make sure 'barcount' matches length of lists:
for barcount, lhl_list in zip(df['barcount'], lists_per_piece):
    if barcount != len(lhl_list):
        print("big problem")

####################
# Get true (from the actual music) transition counts, normalized.
rows = []
total_trans = 0  # only used for sanity check
for lst in lists_per_piece:
    trans_lst = list(zip(lst[:-1], lst[1:]))
    total_trans += len(trans_lst)
    assert len(lst) == len(trans_lst)+1

    # raw counts:
    trans_counts_this_piece = Counter(trans_lst)

    # normalize them based on the average number of transitions per piece
    # (so we simulate every piece having avg_piece_trans transitions)
    for k, v in trans_counts_this_piece.items():
        trans_counts_this_piece[k] = v / len(trans_lst) * avg_piece_trans

    rows.append(trans_counts_this_piece)
true_counts = pd.DataFrame(rows).sum()

print("\nThese are the true counts of transitions in the corpus:")
print("(Normalized so every piece has the same number of transitions)")
print(true_counts)

# Sanity check: check that the total number of transitions in true_counts (the sum)
# matches the number of pieces * avg_piece_trans, which matches
print("\nThese should be very close:")
print(true_counts.sum(), len(df) * avg_piece_trans, total_trans)

###################
# Now simulate random transitions.
# Now get the randomized transition counts.
rows = []
total_trans = 0  # only used for sanity check
for lst in lists_per_piece:
    #token_counts_this_piece = Counter(lst)
    #token_set_this_piece = set(token_counts_this_piece)

    trans_counts_this_piece2 = Counter() # we're going to simulate randomness

    TIMES = 1000  # Number of random reorderings of each song.
    shuffled_lst = lst.copy()
    random.shuffle(shuffled_lst)
    for i in range(TIMES):  # simulate 10 times
        random.shuffle(shuffled_lst)
        trans_lst = list(zip(shuffled_lst[:-1], shuffled_lst[1:]))
        trans_counts_this_piece2.update(Counter(trans_lst))

    for k, v in trans_counts_this_piece2.items():
        trans_counts_this_piece2[k] = v / TIMES / len(trans_lst) * avg_piece_trans

    rows.append(trans_counts_this_piece2)
random_counts = pd.DataFrame(rows).sum()

# Sanity check: check that the total number of transitions in true_counts (the sum)
# matches the number of pieces * avg_piece_trans, which matches
print("These should be very close:")
print(random_counts.sum(), len(df) * avg_piece_trans)


###########
# Now we have true (from ground truth rag music) and simulated
# random counts, we can compare:

true_counts.name = 'true'
random_counts.name = 'random'
both_counts = true_counts.to_frame().merge(random_counts, left_index=True, right_index=True)

def binom_test_func(row):
    if not pd.isna(row['true']) and not pd.isna(row['random']):
        return scipy.stats.binom_test(row['true'], len(df) * avg_piece_trans, p=row['random']/(len(df) * avg_piece_trans))
    else:
        return pd.NA

results = both_counts.apply(binom_test_func, axis=1)
both_counts['pval'] = results
both_counts.sort_values(by='pval', inplace=True)

print()

#######
# Get transition probabilities for paper graphic:
print("Note: 0 -> LHL=0 (None); 10 -> LHL=1,2 (Low), 100 -> LHL>2 (High)")
for i in true_counts.items():
    start_lhl_bin, end_lhl_bin = i[0]
    true_freq = i[1]
    print("Transition type:", start_lhl_bin, end_lhl_bin)
    print("  Raw transition probability: ", true_freq/true_counts.sum())

print()

#######
# Get deviations from expected for paper graphic:
# Print only where p-value < .0001
print("Note: 0 -> LHL=0 (None); 10 -> LHL=1,2 (Low), 100 -> LHL>2 (High)")
for i in both_counts.iterrows():
    start_lhl_bin, end_lhl_bin = i[0]
    expected_freq = i[1].random
    true_freq = i[1].true
    pval = i[1].pval
    print("Transition type:", start_lhl_bin, end_lhl_bin)
    print("  Deviation from expected: ", (true_freq-expected_freq)/expected_freq, end='')
    if pval < 0.0001:
        print("[**Stat sig**]")
    else:
        print("[not stat sig]")
