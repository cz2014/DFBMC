import numpy as np 
import copy

ds = 2

if ds == 1:
    Fin = "./MoSe2_tri/linewidth_wgauss20.elself"
    Fout = "./MoSe2_tri/scat_output"
    NK = 1261
    NBNDout = 4
    BNDlist = [0,1,2,3]
elif ds == 2:
    Fin = "./MoSe2_k180_tri/linewidth_tri.elself"
    Fout = "./MoSe2_k180_tri/scat_output"
    NK = 2791
    NBNDout = 4
    BNDlist = [0,1,2,3]

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

def writer_scat(filename, scat, bande):
    fw = open(filename, 'w')
    for ik in range(len(scat)):
        for ibnd in range(len(scat[0])):
            fw.write("%20.12f %20.12f \n" %(bande[ik][ibnd], scat[ik][ibnd]))

    fw.close()


scat, bande = reader_scat(Fin, NK, NBNDout, BNDlist)
writer_scat(Fout, scat, bande)