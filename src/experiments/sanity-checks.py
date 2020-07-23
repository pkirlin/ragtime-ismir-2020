#%%
import data.RagDataset
import data.PKDataset

pkdata = data.PKDataset.PKDataset()
rags = data.RagDataset.RagDataset()

# Print all titles, no qualifications.
bad = 0
good = 0
for title in pkdata.get_all_titles():
    # print(title)
    series = pkdata.get_best_version_of_rag(title, accept_no_silence_at_start=True,
                                            quant_cutoff=None)
    if series is None:
        bad += 1
    else:
        good += 1

print("Total PK database has", len(pkdata.df), "files, and", good, "unique titles.")

# Print all titles, quant_cutoff at 95%.
bad = 0
good = 0
for title in pkdata.get_all_titles():
    # print(title)
    series = pkdata.get_best_version_of_rag(title, accept_no_silence_at_start=True,
                                            quant_cutoff=0.95)
    if series is None:
        bad += 1
    else:
        good += 1
        #print(title, good)

print("At 95% cutoff, we have", good, "titles.")

# Print all titles, quant_cutoff at 95%, and only files with silence at beginning.
bad = 0
good = 0
for title in pkdata.get_all_titles():
    # print(title)
    series = pkdata.get_best_version_of_rag(title, accept_no_silence_at_start=False,
                                            quant_cutoff=0.95)
    if series is None:
        bad += 1
    else:
        good += 1
        # print(title, good)

print("At 95% cutoff with only silence, we have", good, "titles.")

"""  This is the message when we have 4-Feb in pk-compenium2.csv.
Total PK database has 2016 files, and 1289 unique titles.
At 95% cutoff, we have 1056 titles.
At 95% cutoff with only silence, we have 463 titles.
"""

"""  This is the correct data:
Total PK database has 2016 files, and 1291 unique titles.
At 95% cutoff, we have 1058 titles.
At 95% cutoff with only silence, we have 465 titles.

Total PK database has 2016 files, and 1291 unique titles.
At 95% cutoff, we have 1058 titles.
At 95% cutoff with only silence, we have 459 titles.
(this last one is after fixing the silence_m21 values in the compendium.
"""


#%%
# time sig tallies

# Print all titles, quant_cutoff at 95%.
bad = 0
good = 0
import collections
c = collections.Counter()
for title in pkdata.get_all_titles():
    # print(title)
    series = pkdata.get_best_version_of_rag(title, accept_no_silence_at_start=True,
                                            quant_cutoff=0.95)
    if series is None:
        bad += 1
    else:
        fileid = series['fileid']
        c[pkdata.get_music21_time_signature_clean(fileid)] += 1