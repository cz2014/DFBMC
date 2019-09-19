############################################################################################
## This code is example for time and memory test
##
##
## 09/19/2019 -cz
############################################################################################

import numpy as np 
import matplotlib.pyplot as plt
import copy 
import sys 
import math 
import tracemalloc
from profile import Profile  



ds = 3
# 1 for 30x30 cut
# 2 for 30x30 uncut
# 3 for 120x120
test_mode = 0
# 1 for memory test
# 2 for time test

restart = 1
# 0 for no-restart test
# 1 for restart write test
# 2 for restart read

if test_mode == 1:
    tracemalloc.start()



if ds == 1:
    NK = 900
    NBND = 4
    FILENAME = 'fort.708-cut'
elif ds == 2:
    NK = 900
    NBND = 4
    FILENAME = 'fort.708-full'
elif ds == 3:
    NK = 14400
    NBND = 4
    FILENAME = 'trans_cut6_120_re.120'



# def do_chunk(trans, index, chunk, nbnd, scut):  
#     nline = int(len(chunk)/36)

#     for i in range(nline):
#         ttmp = float(chunk[15+i*36:35+i*36])
#         if ttmp > scut:
#             ik = int(chunk[5+i*36:10+i*36]) - 1
#             ijbnd = int(chunk[10+i*36:15+i*36]) - 1
#             ibnd = ijbnd//nbnd 

#             # jk = kindex_add(ik, int(chunk[0+i*36:5+i*36])-1) 
#             jk = int(chunk[0+i*36:5+i*36]) - 1

#             trans[ik][ibnd].append(ttmp)
#             index[ik][ibnd].append([jk, ijbnd%nbnd])

def do_chunk(chunk, nk, nbnd, scut):  
    nline = int(len(chunk)/36)
    trans = [[[] for _ in range(nbnd)] for _ in range(nk) ]
    # index = [[np.zeros((0,2), dtype=int) for _ in range(nbnd)] for _ in range(nk) ]  
    index = [[[] for _ in range(nbnd)] for _ in range(nk) ]

    for i in range(nline):
        ttmp = float(chunk[15+i*36:35+i*36])
        if ttmp > scut:
            ik = int(chunk[5+i*36:10+i*36]) - 1
            ijbnd = int(chunk[10+i*36:15+i*36]) - 1
            ibnd = ijbnd//nbnd 

            # jk = kindex_add(ik, int(chunk[0+i*36:5+i*36])-1) 
            jk = int(chunk[0+i*36:5+i*36]) - 1

            trans[ik][ibnd].append(ttmp)
            # index[ik][ibnd] = np.append(index[ik][ibnd], [[jk, ijbnd%nbnd]], axis=0) 
            index[ik][ibnd].append([jk, ijbnd%nbnd])

    return trans, index 

def reader_trans(filename, nk, nbnd, scut=-1):
    # trans = []
    # index = []
    scat = np.zeros((nk, nbnd))

    # tmp = []
    # for ibnd in range(nbnd):
    #     tmp.append([])
    # for ik in range(nk):
    #     trans.append(copy.deepcopy(tmp))
    #     index.append(copy.deepcopy(tmp))

    trans = [[[] for _ in range(nbnd)] for _ in range(nk) ]
    index = [[np.zeros((0,2), dtype=int) for _ in range(nbnd)] for _ in range(nk) ] 
    # index = [[[] for _ in range(nbnd)] for _ in range(nk) ]

    
    fo = open(filename, 'r')

    chunk_count = 1
    while True:
        chunk = fo.read(36*1000000) # 36*1e5 approximately 3.6MB 
        
        if chunk_count%5 == 0:
            print("read size ~ ", chunk_count*3.433227539e1, " MB")
            if test_mode == 1:
                print(tracemalloc.get_traced_memory())
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno') # lineno filename traceback

                for stat in top_stats[:10]:
                    print(stat)

        chunk_count += 1

        if not chunk:
            break 

        trans_tmp, index_tmp = do_chunk(chunk, nk, nbnd, scut) 
        for ik in range(nk):
            for ibnd in range(nbnd):
                trans[ik][ibnd] = np.append(trans[ik][ibnd], trans_tmp[ik][ibnd])
                if index_tmp[ik][ibnd]:
                    index[ik][ibnd] = np.append(index[ik][ibnd], index_tmp[ik][ibnd], axis=0)


    if test_mode == 1:
        print(tracemalloc.get_traced_memory())
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno') # lineno filename traceback

        for stat in top_stats[:10]:
            print(stat)


    for ik in range(nk):
        for ibnd in range(nbnd):
            trans[ik][ibnd] = np.array(trans[ik][ibnd])*13.6
            index[ik][ibnd] = np.array(index[ik][ibnd],dtype=int)
            # trans format: [ik](list)[ibnd](list)[trans_rate](array)
            # index format: [ik](list)[ibnd](list)[iq, jbnd](array)

            scat[ik][ibnd] = np.sum(trans[ik][ibnd])

            trans[ik][ibnd] = trans[ik][ibnd] / scat[ik][ibnd] 

        trans[ik] = np.array(trans[ik])

    trans = np.array(trans)


    fo.close()

    return index, trans, scat 
    # in eV units

def writer_data(trans, index, scat, filename="datasets.npz"):

    # # info includes nk, nbnd and length of tran[ink][ibnd]
    # info = np.array([],dtype=int)
    # info = np.append(info, nk)
    # info = np.append(info, nbnd)

    # for k in trans:
    #     for bnd in k:
    #         info = np.append(info, len(bnd))
    
    np.savez(filename, trans, index, scat)

def reader_restart(filename="datasets.npz"):
    tmpfile = np.load(filename, allow_pickle=True,)
    trans = tmpfile['arr_0']
    index = tmpfile['arr_1']
    scat = tmpfile['arr_2']

    return trans, index, scat 

def scat_plot(scat1, scat2, nk, nbnd):
    nlist = nk*nbnd 
    if (scat2 == []):
        comp = np.reshape(scat1, (nlist,-1))
    else:
        comp = np.reshape(scat1, (nlist,-1)) - np.reshape(scat2, (nlist, -1)) 

    plt.hist(comp, bins=50)

    plt.show()
    plt.close()


def trans_count(trans, nk, nbnd):
    tot = 0
    for ik in range(nk):
        for ibnd in range(nbnd):
            tot = tot + trans[ik][ibnd].nbytes

    return tot 


def scat_lost(scatcut, scatuncut, nk, nbnd):
    lostp = (scatuncut - scatcut)/scatuncut 

    scat_plot(lostp, [], nk, nbnd)


def test_step():
    if restart == 0:
        index, trans_cut, scat_cut = reader_trans(FILENAME, NK, NBND, 1e-6)
    elif restart == 1:
        index, trans_cut, scat_cut = reader_trans(FILENAME, NK, NBND, 1e-7)
        writer_data(trans_cut, index, scat_cut, 'datasets-cut1e-7.npz')
    elif restart == 2:
        trans_cut, index, scat_cut = reader_restart('datasets-cut1e-6.npz')


    # if test_mode == 1:
    #     print(tracemalloc.get_traced_memory())
    #     snapshot = tracemalloc.take_snapshot()
    #     top_stats = snapshot.statistics('lineno') # lineno filename traceback

    #     for stat in top_stats[:10]:
    #         print(stat)

    # scat_plot(scat_cut, [], NK, NBND)

    # print("sum of cut:", np.sum(scat_cut))
    # print("bytes of cut:", trans_count(trans_cut, NK, NBND))
    # print("min of cut:", np.min(scat_cut))
    # print(scat_cut[0])

def test_main():

    # tmp1, trans_cut, scat_cut = reader_trans(FILENAME, NK, NBND, 1e-5)
    # tmp1, trans_uncut, scat_uncut = reader_trans(FILENAME, NK, NBND, 1e-6)

    trans_cut, _, scat_cut = reader_restart("datasets-cut1e-6.npz")
    trans_uncut, _, scat_uncut = reader_restart("datasets-cut1e-7.npz")

    scat_plot(scat_cut, [], NK, NBND)
    scat_plot(scat_uncut, [], NK, NBND)

    scat_plot(scat_uncut, scat_cut, NK, NBND)

    print("sum of cut:", np.sum(scat_cut))
    print("bytes of cut:", trans_count(trans_cut, NK, NBND))
    print("min of cut:", np.min(scat_cut))
    print("sum of uncut:", np.sum(scat_uncut))
    print("bytes of uncut:", trans_count(trans_uncut, NK, NBND))
    print("min of uncut:", np.min(scat_uncut))

    scat_lost(scat_cut, scat_uncut, NK, NBND)

    # klen, bande = reader_bands("mos2-30x30.gnu", 900, 17, np.array([13,14,15,16]))

if test_mode == 1:
    test_step()
elif test_mode == 2:
    p = Profile() 
    p.run('test_step()')
    # p.run('reader_restart()')
    p.print_stats()
else:
    # index, trans_cut, scat_cut = reader_trans(FILENAME, NK, NBND, 1e-7)
    # writer_data(trans_cut, index, scat_cut, 'datasets-cut1e-7.npz')
    # test_main()
    test_step()