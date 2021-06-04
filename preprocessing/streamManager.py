import numpy as np
import h5py
import os, sys


class iLabel:
    def __init__(self):
        self.id = None
        self.level = None
        self.hit = None
        self.index = None
        self.map = None
        self.status = False
        self.default()

    def default(self):
        self.level = [1, 1, 1, 1, 2, 1, 2, 2, 2, 1]
        self.map = [-1 if x == 1 else idx - 1 - self.level[:idx][::-1].index(x - 1) for idx, x in enumerate(self.level)]
        self.id = ['----- Begin chunk -----',
                   'Event: //',
                   'num_peaks',
                   'Peaks from peak search',
                   'End of peak list',
                   '--- Begin crystal',
                   'Reflections measured after indexing',
                   'End of reflections',
                   '--- End crystal',
                   '----- End chunk -----']


class iStream:
    def __init__(self):
        self.info = None
        self.crystal = None
        self.label = iLabel()
        self.backup = iLabel()
        self.fstream = None
        self.content = None
        # backup saves default label

    def initial(self, fstream=None, label_id=None, label_level=None):
        """
        update stream file, label_id and label_level, default id and level hold for no inputs
        """
        if os.path.exists(str(fstream)):
            self.clear()
            self.fstream = fstream
            try:
                with open(fstream, 'r') as f:
                    self.content = f.readlines()
            except:
                pass
        if self.content is None: raise Exception('### bad stream file ###')

        self.label.default()
        if isinstance(label_level, list):
            self.label.level = label_level
            self.label.map = [-1 if x == 1 else idx - 1 - self.label.level[:idx][::-1].index(x - 1) for idx, x in
                              enumerate(self.label.level)]
        if isinstance(label_id, list):
            self.label.id = label_id

    def clear(self):
        iStream.__init__(self)

    def keep_label(self):
        if self.label.index is None or self.label.hit is None:
            return
        else:
            self.backup.index = self.label.index.copy()
            self.backup.hit = self.label.hit.copy()
            self.label.status = True

    def update_content(self, newContent):
        """
        You must call this function to make changes, you can't change content manually like self.content=0
        """
        self.content = newContent
        self.label.status = False

    def get_label_alg2(self):
        # print 'getting label (alg=2)'
        FinLabel = []
        FinLabelall = []
        tmpLabel = [[] for x in self.label.id]

        for line, val in enumerate(self.content):
            istart = False
            for j in range(len(self.label.id)):
                if self.label.id[j] in val:
                    if j == 0 and len(tmpLabel[0]) > 0: istart = True; break
                    tmpLabel[j].append(line)
                    break

            if istart or line == len(self.content) - 1:
                [x.append(-1) for x in tmpLabel]

                label_back = [[]]
                for idx in range(1, len(self.label.level)):
                    if len(tmpLabel[idx]) == 0: label_back.append([]); continue
                    blank = []
                    for num in tmpLabel[idx][:]:
                        blank.append(max([x for x in tmpLabel[self.label.map[idx]][:] if x <= num]))
                    label_back.append(blank)

                single = [list(x) for x in np.array(np.meshgrid(*tmpLabel)).T.reshape(-1, len(tmpLabel))]
                for row in single:
                    tmpRow = [x for x in row if x >= 0]
                    if tmpRow != sorted(tmpRow): continue
                    for idx in range(len(row)):
                        if row[idx] != -1:
                            if self.label.map[idx] == -1:
                                continue
                            else:
                                tmpRow = tmpLabel[self.label.map[idx]][:]
                                if row[self.label.map[idx]] == -1: row = None; break
                                if max([x for x in tmpRow if x < row[idx]]) == row[self.label.map[idx]]:
                                    continue
                                else:
                                    row = None; break
                        else:
                            if self.label.map[idx] == -1:
                                if len(tmpLabel[idx]) == 1:
                                    continue
                                else:
                                    row = None; break
                            else:
                                if len(tmpLabel[idx]) == 1: continue
                                if row[self.label.map[idx]] == -1: continue
                                if row[self.label.map[idx]] not in label_back[idx]: continue
                                row = None;
                                break

                    if row is None: continue
                    FinLabelall.append(row)
                    if -1 not in row: FinLabel.append(row)
                tmpLabel = [[] for x in self.label.id]
                tmpLabel[j].append(line)
        self.label.index = np.array(FinLabel)
        self.label.hit = np.array(FinLabelall)

    def get_label_alg1(self):
        # print 'getting label (alg=1)'
        tmpCount = []
        tmpLabel = []
        for i, val in enumerate(self.content):
            for j in range(len(self.label.id)):
                if self.label.id[j] in val:
                    tmpLabel.append(i)
                    tmpCount.append(j)
                    continue
        istart = tmpCount.index(0)
        tmpCount = tmpCount[istart:] + [-1]
        tmpLabel = tmpLabel[istart:] + [-1]

        FinLabel = []
        FinLabelall = []

        ilabel = [tmpLabel[0]]
        icount = [0]
        for i in range(1, len(tmpCount)):
            val = tmpCount[i]
            if val > 0:
                icount.append(val)
                ilabel.append(tmpLabel[i])
            else:
                tmpRow = ilabel[0:5]
                tmpInd = [idx for idx, x in enumerate(icount) if x == 5]
                if len(tmpInd) == 0:
                    FinLabelall.append(tmpRow + [-1, -1, -1, -1] + ilabel[-1:])
                else:
                    for x in tmpInd:
                        FinLabel.append(tmpRow + ilabel[x:(x + 4)] + ilabel[-1:])
                        FinLabelall.append(tmpRow + ilabel[x:(x + 4)] + ilabel[-1:])
                icount = [0]
                ilabel = [tmpLabel[i]]
        self.label.index = np.array(FinLabel)
        self.label.hit = np.array(FinLabelall)

    def get_label(self):
        """
        This function finds line number of each label_id
        """
        if self.content is None:
            raise Exception('### no stream content ###')

        tmpStream = iStream()
        if tmpStream.label.id == self.label.id:
            if self.label.status:
                self.label.index = self.backup.index.copy()
                self.label.hit = self.backup.hit.copy()
                return
            self.get_label_alg1()
            self.label.status = True
            self.backup.index = self.label.index.copy()
            self.backup.hit = self.label.hit.copy()
            return
        else:
            self.get_label_alg2()
        # status True or False indicates whether it's labeled with default label id

    def get_eventList(self, fcxi):
        try:
            f = h5py.File(fcxi, 'r')
            ievent = f['LCLS/eventNumber'][()]
            f.close()
            return ievent
        except:
            raise Exception('### no cxi file ###')

    def get_realPeak(self):
        """
        return real peak position (maxNpeak, 3, numLabel)
        """
        if self.content is None:
            raise Exception('### no stream content ###')

        tmpStream = iStream()
        tmpStream.content = self.content
        if not self.label.status:
            tmpStream.get_label()
            self.backup.index = tmpStream.label.index.copy()
            self.backup.hit = tmpStream.label.hit.copy()
            self.label.status = True
        else:
            tmpStream.label.index = self.backup.index

        if len(tmpStream.label.index) == 0: return np.zeros((0, 3, 0))

        ilabel = tmpStream.label.index[:, 3:5].copy()
        maxNpeak = int(np.amax(ilabel[:, 1] - ilabel[:, 0] - 2))
        irealPeak = np.zeros((maxNpeak, 3, len(ilabel)))

        for i in range(len(ilabel)):
            peakList = tmpStream.content[(ilabel[i, 0] + 2): ilabel[i, 1]]
            for j in range(ilabel[i, 1] - ilabel[i, 0] - 2):
                val = peakList[j].split()
                irealPeak[j, 0, i] = float(val[0])
                irealPeak[j, 1, i] = float(val[1])
                irealPeak[j, 2, i] = float(val[3])
        return irealPeak

    def get_predPeak(self):
        """
        return predicted peak position (maxNpeak, 7, numLabel)
        """
        if self.content is None:
            raise Exception('### no stream content ###')

        tmpStream = iStream()
        tmpStream.content = self.content
        if not self.label.status:
            tmpStream.get_label()
            self.backup.index = tmpStream.label.index.copy()
            self.backup.hit = tmpStream.label.hit.copy()
            self.label.status = True
        else:
            tmpStream.label.index = self.backup.index

        if len(tmpStream.label.index) == 0: return np.zeros((0, 7, 0))

        ilabel = tmpStream.label.index[:, 6:8].copy()
        maxNpeak = int(np.amax(ilabel[:, 1] - ilabel[:, 0] - 2))
        ipredPeak = np.zeros((maxNpeak, 7, len(ilabel)))

        for i in range(len(ilabel)):
            peakList = tmpStream.content[(ilabel[i, 0] + 2): ilabel[i, 1]]
            for j in range(ilabel[i, 1] - ilabel[i, 0] - 2):
                val = peakList[j].split()
                ipredPeak[j, 0, i] = float(val[0])
                ipredPeak[j, 1, i] = float(val[1])
                ipredPeak[j, 2, i] = float(val[2])
                ipredPeak[j, 3, i] = float(val[3])
                ipredPeak[j, 4, i] = float(val[4])
                ipredPeak[j, 5, i] = float(val[7])
                ipredPeak[j, 6, i] = float(val[8])
        return ipredPeak

    def get_info(self, alg=2):
        """
        return info and crystal matrix, only uses default label id
        """
        if self.content is None:
            raise Exception('### no stream content ###')

        tmpStream = iStream()
        tmpStream.content = self.content
        if not self.label.status:
            tmpStream.get_label()
            self.backup.index = tmpStream.label.index.copy()
            self.backup.hit = tmpStream.label.hit.copy()
            self.label.status = True
        else:
            tmpStream.label.index = self.backup.index

        data = np.ones((len(tmpStream.label.index), 23)) * (-16)

        for i in range(len(tmpStream.label.index)):
            val = tmpStream.content[tmpStream.label.index[i, 1]]
            data[i, 0] = int(val.split('/')[-2].split('r')[1])
            fcxi = val.split(':')[1].lstrip().rstrip()

            val = tmpStream.content[tmpStream.label.index[i, 2]]
            data[i, 2] = int(val.split('//')[-1])

            if data[i, 3] < -15:
                data[i, 3] = tmpStream.label.index[i, 4] - tmpStream.label.index[i, 3] - 2
            else:
                raise Exception('### error occurs for nPeaks ###')
            if data[i, 4] < -15:
                data[i, 4] = tmpStream.label.index[i, 7] - tmpStream.label.index[i, 6] - 2
            else:
                raise Exception('### error occurs for nReflection ###')

            for j in range(tmpStream.label.index[i, 5], tmpStream.label.index[i, 6]):
                val = tmpStream.content[j]
                if 'Cell parameters' in val:
                    try:
                        eventList = self.get_eventList(fcxi)
                        data[i, 1] = eventList[int(data[i, 2])]
                    except:
                        pass
                    data[i, 14] = float(val.split()[2])
                    data[i, 15] = float(val.split()[3])
                    data[i, 16] = float(val.split()[4])
                    data[i, 17] = float(val.split()[6])
                    data[i, 18] = float(val.split()[7])
                    data[i, 19] = float(val.split()[8])
                elif 'astar =' in val:
                    data[i, 5] = float(val.split()[2])
                    data[i, 6] = float(val.split()[3])
                    data[i, 7] = float(val.split()[4])
                elif 'bstar =' in val:
                    data[i, 8] = float(val.split()[2])
                    data[i, 9] = float(val.split()[3])
                    data[i, 10] = float(val.split()[4])
                elif 'cstar =' in val:
                    data[i, 11] = float(val.split()[2])
                    data[i, 12] = float(val.split()[3])
                    data[i, 13] = float(val.split()[4])
                elif 'det_shift' in val:
                    data[i, 20] = float(val.split()[3])
                    data[i, 21] = float(val.split()[6])

        self.info = np.around(data[:, 0:5]).astype(int)
        self.crystal = data[:, 5:22].copy()

    def save_label(self, fsave=None):
        """
        save label.index, label.hit and label.comment; if fsave exists, will simply modify it
        """
        if self.label.index is None or self.label.hit is None:
            raise Exception('### no label list ###')
        if not isinstance(fsave, str):
            raise Exception('### no save file ###')

        comment = self.get_comment()

        if os.path.exists(fsave):
            f = h5py.File(fsave, 'r+')
            try:
                f.__delitem__('label')
            except:
                pass
        else:
            f = h5py.File(fsave, 'w')
        wrIndex = f.create_dataset('label/index', np.array(self.label.index).shape, dtype='int')
        wrHit = f.create_dataset('label/hit', np.array(self.label.hit).shape, dtype='int')
        wrComment = f.create_dataset('label/comment', (1,), dtype='S' + str(len(comment)))
        wrIndex[...] = np.array(self.label.index)
        wrHit[...] = np.array(self.label.hit)
        wrComment[...] = np.array(comment)
        f.close()

    def save_info(self, fsave=None):
        """
        save info.info, info.crystal and info.comment; if fsave exists, will simply modify it
        """
        if self.info is None or self.crystal is None:
            raise Exception('### no label list ###')
        if not isinstance(fsave, str):
            raise Exception('### no save file ###')

        comment = self.get_comment()

        if os.path.exists(fsave):
            f = h5py.File(fsave, 'r+')
            try:
                f.__delitem__('info')
            except:
                pass
        else:
            f = h5py.File(fsave, 'w')
        wrInfo = f.create_dataset('info/info', np.array(self.info).shape, dtype='int')
        wrCrystal = f.create_dataset('info/crystal', np.array(self.crystal).shape)
        wrComment = f.create_dataset('info/comment', (1,), dtype='S' + str(len(comment)))
        wrInfo[...] = np.array(self.info)
        wrCrystal[...] = np.array(self.crystal)
        wrComment[...] = np.array(comment)
        f.close()

    def get_comment(self):
        """
        This is comment about the data structure
        """
        comment = '### column name of label ###' + '\n'
        if self.label.id is not None:
            for i, x in enumerate(self.label.id):
                comment = comment + str(i).zfill(2) + ': ' + x + '\n'
        tmpComment = ['',
                      '### column name of info ###',
                      '00: run number',
                      '01: event number',
                      '02: series number',
                      '03: numPeaks',
                      '04: numReflection',
                      '',
                      '### column name of crystal ###',
                      '00-02: astar',
                      '03-05: bstar',
                      '06-08: cstar',
                      '09-14: lattice (a,b,c,al,be,ga)',
                      '15-16: xy shift',
                      '',
                      '### column name of realPeak ###',
                      '00: fs/px',
                      '01: ss/px',
                      '02: intensity',
                      '',
                      '### column name of predPeak ###',
                      '00: h',
                      '01: k',
                      '02: l',
                      '03: intensity',
                      '04: sigma',
                      '05: fs/px',
                      '06: ss/px']
        for x in tmpComment:
            comment = comment + x + '\n'
        return comment