import os,sys
import numpy as np
from pprint import pprint
from streamManager import iStream


#path_stream = "/reg/data/ana03/scratch/zhensu/share/axel/test.stream"
path_stream = "/cds/data/psdm/cxi/cxic0415/scratch/axlevy/psocake/r0100/cxic0415_0100_py2peaknet1.stream"

extract = iStream()
extract.initial(fstream = path_stream)
extract.get_label()
extract.get_info()

print "Number of Reflection lists in stream: ", len(extract.label.index)

for which_idx_extract in range(len(extract.label.index)):
    print "idx:"
    pprint(which_idx_extract)
    image_serial_number_pos = extract.label.index[which_idx_extract][0] + 3
    first_indexed_peak_pos = extract.label.index[which_idx_extract][6] + 2
    last_indexed_peak_pos = extract.label.index[which_idx_extract][-3] - 1 #this position must be included
    print "Image serial number:"
    pprint(extract.content[image_serial_number_pos])

# print "Stream content of this idx:"
# pprint(extract.content[start_position:stop_position+1])

def get_fs_ss_panel(content):
    data = []
    for line in content[2:-1]:
        fs = float(line.split()[-3])
        ss = float(line.split()[-2])
        panel = line.split()[-1].strip()
        data.append((fs, ss, panel))
    return data

# fs_ss_panel_list = get_fs_ss_panel(extract.content[start_position:stop_position+1])

# pprint(fs_ss_panel_list)