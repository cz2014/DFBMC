import numpy as np
import copy

ds = 2
# 1 for 30x30 cut
# 2 for 30x30 uncut
# 3 for 120x120
#test_mode = 0
# 1 for memory test
# 2 for time test

restart = 2
# 0 for no-restart test
# 1 for restart write test
# 11 for write scat only
# 12 for write index and trans only
# 13 for write all but scat from fort.708
# 2 for restart read

#if test_mode == 1:
#    tracemalloc.start()

folder_prefix = './MoSe2_tri/'
prefix = folder_prefix+'ds'

if ds == 1:
    NK = 900
    NBND = 4
    FILENAME = 'fort.708-cut'
elif ds == 2:
    NKlen = 120
    NK = 1261
    NBND = 1
    NBNDout = 1
    IJBNDlist = [0]
    BNDlist = [0]
    FILENAME = folder_prefix+'fort.708'
elif ds == 3:
    NK = 32400
    NBND = 1
    NBNDout = 1
    IJBNDlist = [0]
    BNDlist = [0]
    FILENAME = 'fort.708'

def kindex_add(ik, iq):
    NK = NKlen
    iky = ik%NK 
    ikx = ik//NK 
    iqy = iq%NK
    iqx = iq//NK 

    jky = (iky+iqy)%NK 
    jkx = (ikx+iqx)%NK 

    return jkx*NK + jky 

def do_chunk(chunk, nk, nbnd, nbndout, scut, ijbndlist):
    nline = int(len(chunk)/36)
    trans = [[[] for _ in range(nbndout)] for _ in range(nk) ]
    index = [[[] for _ in range(nbndout)] for _ in range(nk) ]
    ijbndlist = np.array(ijbndlist)

    for i in range(nline):
        ijbnd = int(chunk[10+i*36:15+i*36]) - 1
        if ijbnd in ijbndlist:
          ttmp = float(chunk[15+i*36:35+i*36])
          if ttmp > scut:
              ik = int(chunk[5+i*36:10+i*36]) - 1
             #ijbnd = int(chunk[10+i*36:15+i*36]) - 1
              ibnd = ijbnd//nbnd

              jk = int(chunk[0+i*36:5+i*36]) - 1

              trans[ik][ibnd].append(ttmp)
              index[ik][ibnd].append([jk, ijbnd%nbnd])

    return trans, index

def reader_trans(filename, nk, nbnd, nbndout, scut, ijbndlist):
    scat = np.zeros((nk, nbndout))

    trans = [[[] for _ in range(nbndout)] for _ in range(nk) ]
    index = [[np.zeros((0,2), dtype=int) for _ in range(nbndout)] for _ in range(nk) ]

    fo = open(filename, 'r')

    chunk_count = 1
    while True:
        chunk = fo.read(36*1000000) # 36*1e5 approximately 3.6MB

        if chunk_count%5 == 0:
            print("read size ~ ", chunk_count*3.433227539e1, " MB")

        chunk_count += 1

        if not chunk:
            break

        trans_tmp, index_tmp = do_chunk(chunk, nk, nbnd, nbndout, scut, ijbndlist)
        for ik in range(nk):
            for ibnd in range(nbndout):
                trans[ik][ibnd] = np.append(trans[ik][ibnd], trans_tmp[ik][ibnd])
                if index_tmp[ik][ibnd]:
                    index[ik][ibnd] = np.append(index[ik][ibnd], index_tmp[ik][ibnd], axis=0)

    for ik in range(nk):
        for ibnd in range(nbndout):
            trans[ik][ibnd] = np.array(trans[ik][ibnd])*13.6
            index[ik][ibnd] = np.array(index[ik][ibnd],dtype=int)
            # trans format: [ik](list)[ibnd](list)[trans_rate](array)
            # index format: [ik](list)[ibnd](list)[iq, jbnd](array)

            scat[ik][ibnd] = np.sum(trans[ik][ibnd])

            trans[ik][ibnd] = trans[ik][ibnd] / scat[ik][ibnd]

#       trans[ik] = np.array(trans[ik])

    trans = np.array(trans)

    fo.close()

    return index, trans, scat
    # in eV units

def reader_scat(filename, nk, nbndout, bndlist):

    fo = open(filename, 'r')

    line = fo.readline()
    line = fo.readline()

    line = fo.readline()
    scat = np.zeros((nk, nbndout))
    bande = np.zeros((nk, nbndout))
    bndlist = np.array(bndlist)
    while line:
        line = line.split()
        ik = int(line[0]) - 1
        ibnd = int(line[1]) - 1
        im = float(line[4])
        ie = float(line[2])
        if ibnd in bndlist:
            scat[ik][ibnd] += im
            bande[ik][ibnd] = ie 

        line = fo.readline()

    # scat = scat*0.001
    bande = bande - np.min(bande)
    return scat, bande 

def reader_velocity(filename, nk, nbndout, nbnd, ibndlist):
    kvec = np.zeros((nk, 3))
    velocity = np.zeros((nk, nbndout, 3))
    bande = np.zeros((nk, nbndout))

    fo = open(filename, 'r')

    for ik in range(nk):
        for ibnd in range(nbnd):
            line = fo.readline()

            if (line[0] == '#'):
                line = fo.readline()
                line = fo.readline()
                line = fo.readline()

            if ibnd in ibndlist:
                line = line.split()

                if (ibnd == 0):
                    kvec[ik][0] = float(line[1])
                    kvec[ik][1] = float(line[2])
                    kvec[ik][2] = float(line[3])

                bande[ik][ibnd] = float(line[4])

                velocity[ik][ibnd][0] = float(line[5])
                velocity[ik][ibnd][1] = float(line[6])
                velocity[ik][ibnd][2] = float(line[7])

    fo.close()

    return kvec, bande, velocity

def writer_data(trans, index, scat, prefix="datasets"):

    np.save(prefix+"_trans.npy", trans)
    np.save(prefix+"_index.npy", index)
    np.save(prefix+"_scat.npy", scat)

def writer_scat(scat, prefix='datasets'):
    np.save(prefix+"_scat.npy", scat)

def writer_traind(trans, index, prefix="datasets"):
    np.save(prefix+"_trans.npy", trans)
    np.save(prefix+"_index.npy", index)

def reader_restart(prefix="datasets"):
    trans = np.load(prefix+"_trans.npy" , allow_pickle=True)
    index = np.load(prefix+"_index.npy", allow_pickle=True)
    scat = np.load(prefix+"_scat.npy", allow_pickle=True)

    return trans, index, scat

def printer_scat(filename, scat, bande):
    fw = open(filename, 'w')
    for ik in range(len(scat)):
        for ibnd in range(len(scat[0])):
            fw.write("%20.12f %20.12f \n" %(bande[ik][ibnd], scat[ik][ibnd]))

    fw.close()

# def trans_ejump(trans, index, bande):


def test_step():
    if restart == 0:
        index, trans_cut, scat_cut = reader_trans(FILENAME, NK, NBND, NBNDout, 1e-6, IJBNDlist)
    elif restart == 1:
        index, trans_cut, _ = reader_trans(FILENAME, NK, NBND, NBNDout, 1e-8, IJBNDlist)
        scat_cut, _ = reader_scat('linewidth.elself', NK, NBNDout, BNDlist)
        writer_data(trans_cut, index, scat_cut, prefix)
    elif restart == 11:
        scat_cut, _ = reader_scat('linewidth.elself', NK, NBNDout, BNDlist)
        writer_scat(scat_cut, prefix)
    elif restart == 12:
        index, trans_cut, _ = reader_trans(FILENAME, NK, NBND, NBNDout, 1e-8, IJBNDlist)
        writer_traind(trans_cut, index, prefix)
    elif restart == 13:
        index, trans_cut, scat_cut = reader_trans(FILENAME, NK, NBND, NBNDout, 1e-8, IJBNDlist)
        writer_data(trans_cut, index, scat_cut, prefix)

    elif restart == 2:
        trans_cut, index, scat_cut = reader_restart(prefix)
        scat_lw, bande = reader_scat(folder_prefix+'linewidth_wgauss20.elself', NK, 1, [0])
        printer_scat(folder_prefix+'scat_output', scat_lw, bande) 

test_step()

