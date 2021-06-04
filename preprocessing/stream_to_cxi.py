import numpy as np
import h5py
from streamManager import iStream
import argparse
import os
import shutil

def get_event_number(line):
    id = int(line.split()[1].split('/')[2])
    return id

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filename", "-f", type=str, required=True, help="Path to .stream file")
    p.add_argument("--events_per_cxi", type=int, default=1, help="Number of events per .cxi file")
    return p.parse_args()

def main():
    args = parse_args()

    save_dir = args.filename.split('.')[0] + '_peak_finding_and_indexing'
    print("Saving directory: " + save_dir)

    if os.path.exists(save_dir):
        y = 'y'
        val = input("The saving directory exists. Overwrite? (y/n)")
        if val == 'y':
            shutil.rmtree(save_dir)
            print("Directory removed.")

    os.makedirs(save_dir)

    print("Reading stream file " + args.filename + "...")

    extract = iStream()
    extract.initial(fstream=args.filename)
    extract.get_label()
    extract.get_info()

    print("Number of reflection lists in stream: " + str(len(extract.label.index)))
    # n_cxi_files = len(extract.label.index) // args.events_per_cxi #some events are removed here
    n_cxi_files = 1

    for idx_cxi in range(n_cxi_files):
        print()
        print("Writing cxi file " + str(idx_cxi + 1) + "/" + str(n_cxi_files) + "...")
        name_cxi = '/cxi_' + str(idx_cxi) + '.cxi'
        print("Name: " + name_cxi)
        cxi_file = h5py.File(save_dir + name_cxi, 'w')
        event_numbers = []
        LCLS = cxi_file.create_group('LCLS')
        for idx_list in range(args.events_per_cxi):
            print("List " + str(idx_list))
            idx_stream = idx_cxi * args.events_per_cxi + idx_list
            event_number_pos = extract.label.index[idx_stream][0] + 2
            event_number = get_event_number(extract.content[event_number_pos])
            event_numbers.append(event_number)
        event_numbers = np.array(event_numbers)
        LCLS.create_dataset('eventNumber', data=event_numbers)
        cxi_file.close()
        print(name_cxi + " closed.")

    print("Done with preprocessing.")

if __name__ == "__main__":
    main()