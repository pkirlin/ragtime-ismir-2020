'''
Code generating content from section 4.2 of the paper.
'''


import data.PKDataset
import data.RagDataset
import scipy.stats
import pandas as pd
from collections import Counter
from pathlib import Path
#import functions
from collections import defaultdict
import random

CREATE_FIGURES = True
FIGURES_DIR = (Path(__file__).parent / "../../ismir-figures/").resolve()

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

df_early = df[df['year_cat'] == '1890-1901']
df_late = df[df['year_cat'] == '1902-1919']
df_earlylate = df[ (df['year_cat'] == '1890-1901') | (df['year_cat'] == '1902-1919') ]
df_modern = df[df['year_cat'] == '>1919']

df_joplin = df[df['composer']=='Joplin, Scott']
df_scott = df[df['composer']=='Scott, James']
df_lamb = df[df['composer']=='Lamb, Joseph F.']

bip_dist = pd.Series(average_bips(normalize_bips(df)))
bip_dist_early = pd.Series(average_bips(normalize_bips(df_early)))
bip_dist_late = pd.Series(average_bips(normalize_bips(df_late)))
bip_dist_earlylate = pd.Series(average_bips(normalize_bips(df_earlylate)))
bip_dist_modern = pd.Series(average_bips(normalize_bips(df_modern)))

bip_dist_joplin = pd.Series(average_bips(normalize_bips(df_joplin)))
bip_dist_scott = pd.Series(average_bips(normalize_bips(df_scott)))
bip_dist_lamb = pd.Series(average_bips(normalize_bips(df_lamb)))

df_big3 = df[(df['composer']=='Joplin, Scott') | (df['composer']=='Scott, James') | (df['composer']=='Lamb, Joseph F.')]
df_nonbig3 = df[(df['composer']!='Joplin, Scott') & (df['composer']!='Scott, James') & (df['composer']!='Lamb, Joseph F.')]

df_big3_late = df_big3[df_big3['year_cat']=='1902-1919']
df_nonbig3_late = df_nonbig3[df_nonbig3['year_cat']=='1902-1919']

###################################
# print some stats - first part of section 4.2

print(f'Have {len(df)} pieces')
print("total bars:", df['barcount'].sum())
print("total bars with lhl>0:", df['lhl_pos_count'].sum())
print("...which is a % of", df['lhl_pos_count'].sum()/df['barcount'].sum())
print("avg LHL")

# make 2 expanded dataframes containing one row per measure/one transition per row
rowslist = []
for num, row in df.iterrows():
    fileid = row.name
    bip_list = row['bip_list']
    lhl_list = row['lhl_list']
    assert len(bip_list)==len(lhl_list)
    for barnum, melbip, lhl in zip(range(0, len(bip_list)), bip_list, lhl_list):
        D = {}
        D['fileid'] = fileid
        D['barnum'] = barnum+1
        D['melbip'] = melbip
        D['lhl'] = lhl
        rowslist.append(D)
df_expanded = pd.DataFrame(rowslist)

print("Confirming from above, total measures=", len(df_expanded))
print("number of those measures with lhl>0", len(df_expanded[df_expanded['lhl']>0]))
print('lhl stats whole corpus:')
print(df_expanded['lhl'].describe() )
print("lhl>0 stats whole corpus:")
print(df_expanded[df_expanded['lhl']>0]['lhl'].describe() )

################
# Statistical significance tests for Patterns by era:

freqs_whole = pd.Series(average_bips(normalize_bips(df))).to_frame(name='freq_whole')
freqs_whole['lhl_whole'] = freqs_whole.index.map(lambda x: compute_lhl(x, '1'))
freqs_whole.sort_values(by='freq_whole', ascending=False, inplace=True)
freqs_early = pd.Series(average_bips(normalize_bips(df_early))).to_frame(name='freq_early')
freqs_early['lhl_early'] = freqs_early.index.map(lambda x: compute_lhl(x, '1'))
freqs_early.sort_values(by='freq_early', ascending=False, inplace=True)
freqs_late = pd.Series(average_bips(normalize_bips(df_late))).to_frame(name='freq_late')
freqs_late['lhl_late'] = freqs_late.index.map(lambda x: compute_lhl(x, '1'))
freqs_late.sort_values(by='freq_late', ascending=False, inplace=True)
freqs_earlylate = pd.Series(average_bips(normalize_bips(df_earlylate))).to_frame(name='freq_earlylate')
freqs_earlylate['lhl_earlylate'] = freqs_earlylate.index.map(lambda x: compute_lhl(x, '1'))
freqs_earlylate.sort_values(by='freq_earlylate', ascending=False, inplace=True)
freqs_modern = pd.Series(average_bips(normalize_bips(df_modern))).to_frame(name='freq_modern')
freqs_modern['lhl_modern'] = freqs_modern.index.map(lambda x: compute_lhl(x, '1'))
freqs_modern.sort_values(by='freq_modern', ascending=False, inplace=True)

# compare early-late and earlylate-modern
import numpy as np
compare_e_l = freqs_early.merge(freqs_late, how='outer', left_index=True, right_index=True).replace(np.nan, 0)
compare_e_m = freqs_early.merge(freqs_modern, how='outer', left_index=True, right_index=True).replace(np.nan, 0)
compare_l_m = freqs_late.merge(freqs_modern, how='outer', left_index=True, right_index=True).replace(np.nan, 0)
compare_el_m = freqs_earlylate.merge(freqs_modern, how='outer', left_index=True, right_index=True).replace(np.nan, 0)

test_e_l = scipy.stats.wilcoxon(x=compare_e_l['freq_early'], y=compare_e_l['freq_late'])
test_e_m = scipy.stats.wilcoxon(x=compare_e_m['freq_early'], y=compare_e_m['freq_modern'])
test_l_m = scipy.stats.wilcoxon(x=compare_l_m['freq_late'], y=compare_l_m['freq_modern'])
test_el_m = scipy.stats.wilcoxon(x=compare_el_m['freq_earlylate'], y=compare_el_m['freq_modern'])

print(test_e_l,test_e_m,test_l_m,sep='\n')
print("Using Sidak correction for m=3 tests: ")
print("Before correcting, choose alpha = 0.05")
print("After Sidak correction, we must use alpha=", 1-((1-0.05)**(1/3)) )
# Note how test_e_l, test_e_m, test_l_m all have pvalues < 0.0169.
# Also note how test_el_m does not.

##### bip transitions #####

bip_trans_dist = pd.Series(average_bips(normalize_bip_trans(df)))
bip_trans_dist_early = pd.Series(average_bips(normalize_bip_trans(df_early)))
bip_trans_dist_late = pd.Series(average_bips(normalize_bip_trans(df_late)))
bip_trans_dist_earlylate = pd.Series(average_bips(normalize_bip_trans(df_earlylate)))
bip_trans_dist_modern = pd.Series(average_bips(normalize_bip_trans(df_modern)))


#####################################################

# Figure 5 in the paper:

# frequentize the bips and make
# plots for frequent patterns

if CREATE_FIGURES:

    import matplotlib.pyplot as plt
    import numpy as np

    # all patterns
    freqs_whole = pd.Series(average_bips(normalize_bips(df))).to_frame(name='freq')
    freqs_whole['lhl'] = freqs_whole.index.map(lambda x: compute_lhl(x, '1'))
    freqs_whole.sort_values(by='freq', ascending=False, inplace=True)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.15)
    fig.set_size_inches(5, 3)
    data=freqs_whole['freq'].head(10)
    plt.bar(x=np.arange(len(data)), height=list(data))
    ax=plt.axes()
    ax.set_xticks(range(len(data.index)))
    ax.set_xticklabels(data.index, rotation=90)

    # patterns with lhl>0
    freqs_gt0 = freqs_whole[freqs_whole['lhl']>0]
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.15)
    fig.set_size_inches(5, 3)
    data=freqs_gt0['freq'].head(10)
    plt.bar(x=np.arange(len(data)), height=list(data))
    ax=plt.axes()
    ax.set_xticks(range(len(data.index)))
    ax.set_xticklabels(data.index, rotation=90)

    # one plot with both
    fig, ax =plt.subplots(1,2, sharey=True)
    fig = plt.gcf()
    plt.gcf().subplots_adjust(bottom=0.35)
    fig.set_size_inches(5, 3)

    data=freqs_whole['freq'].head(10)
    ax[0].bar(x=np.arange(len(data)), height=list(data))
    ax[0].set_xticks(range(len(data.index)))
    ax[0].set_xticklabels(data.index, rotation=90)

    data=freqs_gt0['freq'].head(10)
    ax[1].bar(x=np.arange(len(data)), height=list(data))
    ax[1].set_xticks(range(len(data.index)))
    ax[1].set_xticklabels(data.index, rotation=90)
    ax[0].set_ylabel('Average frequency per measure')
    fig.savefig(Path(FIGURES_DIR / 'exp-bip-all-and-gt0.pdf'), bbox_inches='tight')



########## FIGURE 6 ##############

if CREATE_FIGURES:
    data1 = freqs_early[freqs_early['lhl_early'] > 0]['freq_early']
    data2 = freqs_late[freqs_late['lhl_late'] > 0]['freq_late']
    data3 = freqs_modern[freqs_modern['lhl_modern'] > 0]['freq_modern']

    plt.figure()
    fig = plt.gcf()
    # plt.gcf().subplots_adjust(bottom=0.15)
    fig.set_size_inches(5, 2)
    merged_index = set(
        freqs_early[freqs_early['lhl_early'] > 0]['freq_early'].head(10).index).union(
        set(freqs_late[freqs_late['lhl_late'] > 0]['freq_late'].head(10).index).union(
            set(freqs_modern[freqs_modern['lhl_modern'] > 0]['freq_modern'].head(10).index)
        )
    )

    alldata = data1.to_frame().merge(data2.to_frame(), how='outer', left_index=True,
                                     right_index=True).replace(np.nan, 0).merge(
        data3.to_frame(), how='outer', left_index=True, right_index=True).replace(np.nan, 0)
    alldata['sortnum'] = alldata.max(axis=1)
    alldata.sort_values(by='sortnum', inplace=True, ascending=False)

    take = 10  # how many bars to make

    # make bar heights
    bars_early = list(alldata['freq_early'])[:take]
    bars_late = list(alldata['freq_late'])[:take]
    bars_modern = list(alldata['freq_modern'])[:take]

    # Set position of bar on X axis
    barWidth = .25
    r1 = np.arange(len(bars_early))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    plt.bar(r1, bars_early, color='red', width=barWidth, edgecolor='white', label='Early ragtime')
    plt.bar(r2, bars_late, color='blue', width=barWidth, edgecolor='white', label='Late ragtime')
    plt.bar(r3, bars_modern, color='black', width=barWidth, edgecolor='white', label='Modern era')

    # Add xticks on the middle of the group bars
    # plt.xlabel('Binary onset pattern')#, fontweight='bold')
    plt.xticks([r + barWidth for r in range(take)], alldata.index[:take], rotation=90)
    plt.legend()
    fig.savefig(Path(FIGURES_DIR / 'exp-bip-by-era.pdf'), bbox_inches='tight')

