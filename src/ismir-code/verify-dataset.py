'''
This code verifies that the corpus is in good shape by printing some
simple statistics.  It also verifies that all the data files can be opened
and processed.
'''

import data.RagDataset
import data.PKDataset

pkdata = data.PKDataset.PKDataset()
rags = data.RagDataset.RagDataset()

#############################################################################################

# Print info about all MIDIs.  These correspond to all the MIDI files
# originally available that weren't corrupted (ones able to be loaded correctly).
num_midis = len(rags.get_all_fileids())
print("We have", num_midis, "MIDI files in the RAG-C corpus.")

#############################################################################################

# Print all titles, quant_cutoff at 95%, and only files with silence at beginning.
# Note: This should result in 1058 files passing all these tests (unique title, quantization
#   cutoff >= 95%, and series cannot be None [None happens if there were only files in the DB
#   with < 95% quantization accuracy.

bad = 0
good = 0
ts_22 = 0
ts_24 = 0
ts_44 = 0
for title in pkdata.get_all_titles():
    series = pkdata.get_best_version_of_rag(title, accept_no_silence_at_start=True,
                                            quant_cutoff=0.95)
    if series is None:
        bad += 1
    else:
        good += 1
        fileid = series['fileid']
        ts = pkdata.get_music21_time_signature_clean(fileid)
        if ts == '2/2': ts_22 += 1
        elif ts == '2/4': ts_24 += 1
        elif ts == '4/4': ts_44 += 1
        else: assert False  # this had better not happen...

print("At 95% cutoff, accepting silence at the start, we have", good, "acceptable pieces.")
print(f"\nTime signature counts: 2/4: {ts_24}, 4/4: {ts_44}, 2/2: {ts_22}")
# Sanity check from paper:
# We have 11557 MIDI files in the RAG-C corpus.
# At 95% cutoff, accepting silence at the start, we have 1058 acceptable pieces.
# Time signature counts: 2/4: 810, 4/4: 214, 2/2: 34