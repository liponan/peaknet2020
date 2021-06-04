import numpy as np
import h5py
from streamManager import iStream
import argparse
import os
import shutil

def get_event_number(extract, idx_stream):
    event_number_pos = extract.label.index[idx_stream][2]
    line = extract.content[event_number_pos]
    id = int(line.split()[1].split('/')[2])
    return id

def get_nPeaks(extract, idx_stream):
    nPeaks_pos = extract.label.index[idx_stream][3]
    line = extract.content[nPeaks_pos]
    nPeaks = int(line.split()[2])
    return nPeaks

def get_fs_ss_XPos_YPos(extract, idx_stream, nPeaks):
    peak_pos_0 = extract.label.index[idx_stream][4] + 2
    fs_list = []
    ss_list = []
    XPos = []
    YPos = []
    for k in range(nPeaks):
        peak_pos = peak_pos_0 + k
        line = extract.content[peak_pos]
        fs = float(line.split()[0])
        ss = float(line.split()[1])
        panel = line.split()[4]
        qa_list = panel.split('q')[1].split('a')
        q = int(qa_list[0])
        a = int(qa_list[1])
        fs_list.append(fs)
        ss_list.append(ss)
        XPos.append(388 * q) # see CXILabel/__getitem__ in data.py
        YPos.append(185 * (a // 2))  # see CXILabel/__getitem__ in data.py
    return fs_list, ss_list, XPos, YPos


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filename", "-f", type=str, required=True, help="Path to .stream file")
    p.add_argument("--events_per_cxi", type=int, default=10, help="Number of events per .cxi file")
    p.add_argument("--max_n_peaks", type=int, default=2048, help="Maximum number of peaks")
    p.add_argument("--default_detector", type=str, default="CxiDs1.0:Cspad.0", help="Default detector type")
    return p.parse_args()

def main():
    args = parse_args()
    max_n_peaks = args.max_n_peaks

    print()
    default_detector = args.default_detector
    print("The detector is unknown.")
    print("Default detector " + default_detector + " will be used.")

    save_dir = args.filename.split('.')[0] + '_peak_finding_and_indexing'
    print("Saving directory: " + save_dir)

    if os.path.exists(save_dir):
        y = 'y'
        val = input("The saving directory exists. Overwrite? (y/n)")
        if val == 'y':
            shutil.rmtree(save_dir)
            print("Directory removed.")
        else:
            return

    os.makedirs(save_dir)

    print("Reading stream file...")

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
        name_cxi = 'file_' + str(idx_cxi) + '.cxi'
        print("Name: " + name_cxi)
        cxi_file = h5py.File(save_dir + '/' +name_cxi, 'w')

        event_numbers = []
        LCLS = cxi_file.create_group('LCLS')
        nPeaks_list = []
        peak2 = np.zeros((args.events_per_cxi, max_n_peaks), dtype=float)
        peak1 = np.zeros((args.events_per_cxi, max_n_peaks), dtype=float)
        result_1 = cxi_file.create_group('entry_1/result_1')
        detector_1 = cxi_file.create_group('entry_1/instrument_1/detector_1')
        peakXPosRaw = np.zeros((args.events_per_cxi, max_n_peaks), dtype=float)
        peakYPosRaw = np.zeros((args.events_per_cxi, max_n_peaks), dtype=float)

        for idx_list in range(args.events_per_cxi):
            print("List " + str(idx_list))
            idx_stream = idx_cxi * args.events_per_cxi + idx_list

            event_number = get_event_number(extract, idx_stream)
            event_numbers.append(event_number)
            nPeaks = get_nPeaks(extract, idx_stream)
            if nPeaks > max_n_peaks:
                nPeaks = max_n_peaks
            nPeaks_list.append(nPeaks)
            fs_list, ss_list, XPos, YPos = get_fs_ss_XPos_YPos(extract, idx_stream, nPeaks)
            fs_array = np.zeros((max_n_peaks), dtype=float)
            fs_array[:nPeaks] = np.array(fs_list)
            ss_array = np.zeros((max_n_peaks), dtype=float)
            ss_array[:nPeaks] = np.array(ss_list)
            peak2[idx_list] = fs_array[:]
            peak1[idx_list] = ss_array[:]
            XPos_array = np.zeros((max_n_peaks), dtype=float)
            XPos_array[:nPeaks] = np.array(XPos)
            YPos_array = np.zeros((max_n_peaks), dtype=float)
            YPos_array[:nPeaks] = np.array(YPos)
            peakXPosRaw[idx_list] = XPos_array[:]
            peakYPosRaw[idx_list] = YPos_array[:]
            #
        event_numbers = np.array(event_numbers)
        LCLS.create_dataset('eventNumber', data=event_numbers)
        nPeaks_array = np.array(nPeaks_list)
        result_1.create_dataset('nPeaks', data=nPeaks_array)
        result_1.create_dataset('peak2', data=peak2)
        result_1.create_dataset('peak1', data=peak1)
        detector_1.create_dataset('description', data=default_detector)
        result_1.create_dataset('peakXPosRaw', data=peakXPosRaw)
        result_1.create_dataset('peakYPosRaw', data=peakYPosRaw)
        cxi_file.close()
        print("eventNumber:")
        print(event_numbers)
        print("nPeaks:")
        print(nPeaks_array)
        print(name_cxi + " closed.")

    print("Done with preprocessing.")

if __name__ == "__main__":
    main()