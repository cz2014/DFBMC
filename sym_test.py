############################################################################################
## This program create all the input files needed 
## 1. tt_geninterp.kpt for irr-bz velocity calculation for wannier program (not used now)
## 2. kpt.dat for e-p coupling for epw program
## 3. trangular_epw for to do trianle method in epw
##
## Following files are needed:
## 1. info including symmetric information
## 2. scf.out including irr k points
## 3. tt_geninterp.dat for wannier band energy
##
## 09/19/2019 -cz
############################################################################################


import numpy as np 
import math

NK = 120
NKtot = NK*NK 
NBND = 4

infiles_generate = 2
# 1 for to generate wannier and epw points
# 2 for to generate triangular_epw

reci_vec = np.array([[0.314027,  0.181303],[0.000000,  0.362607]]) 
reci_vec = 2*math.pi*reci_vec 
inv_reci_vec = np.linalg.inv(reci_vec) 

def floor_list(in_ndarray):
    out_ndarray = in_ndarray.reshape(-1).copy()
    for i in range(len(out_ndarray)):
        out_ndarray[i] = out_ndarray[i] - math.floor(out_ndarray[i])

    out_ndarray = np.reshape(out_ndarray, np.shape(in_ndarray))

    return out_ndarray

def get_kfrac(ik_list):
    kcart_list = np.zeros((len(ik_list),2))
    for ik in range(len(ik_list)):
        kcart_list[ik][0] = float(ik_list[ik]//NK)/NK 
        kcart_list[ik][1] = float(ik_list[ik]%NK)/NK 

    return kcart_list
    
def cart2frac(kcart_list):
    kfrac_list = np.zeros((len(kcart_list),2))
    for ik in range(len(kcart_list)):
        kfrac_list[ik][0], kfrac_list[ik][1] = np.dot(kcart_list[ik], inv_reci_vec)

    return kfrac_list

def frac2cart(kfrac_list):
    kcart_list = np.zeros(np.shape(kfrac_list))
    for ik in range(len(kfrac_list)):
        kcart_list[ik][0], kcart_list[ik][1] = np.dot(kfrac_list[ik], reci_vec)

    return kcart_list

def get_kindex2(kfrac_list):
    ik_list = np.zeros((len(kfrac_list),), dtype=int)
    for ik in range(len(kfrac_list)): 
        ikx = int(round(kfrac_list[ik][0] * NK))%NK
        iky = int(round(kfrac_list[ik][1] * NK))%NK
        ik_list[ik] = iky + ikx * NK 

    return ik_list 

def reader_scfout(FILENAME, NK, ndim=2):
    fo = open(FILENAME, 'r')

    line = fo.readline()
    while line:        
        line = line.split()
        if not line:
            line = fo.readline()
            continue

        if line[0] == 'number' and line[3] == 'points=':
            nirrk = int(line[4])
            irrckx = np.zeros((nirrk, 3))
            irrwk = np.zeros((nirrk, ))

            line = fo.readline()
            for i in range(nirrk):
                line = fo.readline().replace('(', ' ').replace(')', ' ').split()
                irrckx[i][0] = float(line[3])
                irrckx[i][1] = float(line[4])
                irrckx[i][2] = float(line[5])
                irrwk[i] = float(line[9])


            irrfkx = np.zeros((nirrk, 3))
            line = fo.readline()
            line = fo.readline()
            for i in range(nirrk):
                line = fo.readline().replace('(', ' ').replace(')', ' ').split()
                irrfkx[i][0] = float(line[3])
                irrfkx[i][1] = float(line[4])
                irrfkx[i][2] = float(line[5])

        line = fo.readline()

    fo.close()
    fo = open('info', 'r')
    line = fo.readline()
    while line:
        line = line.split()
        if not line:
            line = fo.readline()
            continue

        if line[1] == 'symmetry':
            nsym = int(line[0])
            sym_matrix = np.zeros((nsym, 3, 3))
            for i in range(nsym//6):
                line = fo.readline()
                line = fo.readline().split()
                sym_matrix[i*6:(i+1)*6, 0] = np.reshape([int(s) for s in line], (6,3))
                line = fo.readline().split()
                sym_matrix[i*6:(i+1)*6, 1] = np.reshape([int(s) for s in line], (6,3))
                line = fo.readline().split()
                sym_matrix[i*6:(i+1)*6, 2] = np.reshape([int(s) for s in line], (6,3))
            
            if nsym%6 != 0:
                tmp = nsym%6
                line = fo.readline()
                line = fo.readline().split()
                sym_matrix[i*6:i*6+tmp, 0] = np.reshape([int(s) for s in line], (tmp,3))
                line = fo.readline().split()
                sym_matrix[i*6:i*6+tmp, 1] = np.reshape([int(s) for s in line], (tmp,3))
                line = fo.readline().split()
                sym_matrix[i*6:i*6+tmp, 2] = np.reshape([int(s) for s in line], (tmp,3))

        line = fo.readline()

    if ndim == 2:
        sym_matrix = sym_matrix[:,0:2,0:2]
        irrckx = irrckx[:,0:2]
        irrfkx = irrfkx[:,0:2]

    return sym_matrix, 2*math.pi*irrckx, irrwk, irrfkx

def kmap_array(sym_matrix, irrfkx):
    bz2ibz = np.ones((NKtot, ), dtype=int)*-1
    bz_sym = np.ones((NKtot, ), dtype=int)*-1

    nirrk = len(irrfkx)
    nsym = len(sym_matrix)

    for irrk in range(nirrk):
        p2bz_ckx = np.array([np.dot(sym_matrix[i], irrfkx[irrk]) for i in range(nsym)])
        p2bz_ik = get_kindex2(p2bz_ckx)

        bz2ibz[p2bz_ik] = irrk
        bz_sym[p2bz_ik] = np.arange(nsym)

    if -1 in bz2ibz:
        print("problems here!!!")
    if -1 in bz_sym:
        print("problems here!!!")

    return bz2ibz, bz_sym

def output_infiles(irrfkx, irrwk):
    nirrk = len(irrfkx)

    # write wannier input
    fw = open('tt_geninterp.kpt', 'w')
    fw.write("The .rst line is a comment (its maximum allowed length is 500 characters).i\n")
    fw.write("crystal\n")
    fw.write("%i\n" %(nirrk))
    for i in range(nirrk):
        fw.write("%i %20.12f %20.12f %20.12f \n" %(i+1, irrfkx[i][0], irrfkx[i][1], 0.0))
    fw.close()

    # write epw k points input
    fw = open('kpt.dat', 'w')
    fw.write("%i\n" %(nirrk))
    for i in range(nirrk):
        fw.write("  %20.12f %20.12f %20.12f %20.12f\n" %(irrfkx[i][0], irrfkx[i][1], 0.0, irrwk[i]))
    fw.close()

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


def output_epwinfiles(bz2ibz, bande):
    nkirr = np.max(bz2ibz) + 1
    ibz2bz = np.zeros((nkirr, ), dtype=int)

    ibz2bz = [np.argwhere(bz2ibz==i)[0][0] for i in range(nkirr)]

    fw = open('triangular_epw', 'w')
    # fw.write('NKirr, NKtot, NBND \n')
    fw.write('%10i %10i %10i\n' %(nkirr, NKtot, NBND))
    for i in range(nkirr):
        fw.write('%10i %10i \n' %(i+1, ibz2bz[i]+1))

    for ik in range(NKtot):
        for ibnd in range(NBND):
            fw.write('%10i %10i %20.12e \n' %(ik+1, ibnd+1, bande[ik][ibnd]))

    fw.close()


def main_test(FILENAME):
    sym_matrix, irrckx, irrwk, irrfkx = reader_scfout(FILENAME, NK) 
    bz2ibz, bz_sym = kmap_array(sym_matrix, irrfkx)
    if infiles_generate == 1:
        output_infiles(irrfkx, irrwk)
    elif infiles_generate == 2:
        _, bande, _ = reader_velocity('tt_geninterp.dat', NKtot, NBND, NBND, range(NBND))
        output_epwinfiles(bz2ibz, bande)



main_test('scf'+str(NK)+'.out')