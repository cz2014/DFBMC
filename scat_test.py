import numpy as np 
import copy
from math import pi, e, floor 

todolist = 5
# 1 for to translate linewidth to scattering rate
# 2 for to translate linewidth to mobility
# 3 for to calculate energy jump 
# 4 for to check detailed balance 
# 5 for to reset detailed balance
ds = 3
restart = 0

if ds == 1:
    prefix = "./WS2/"
    Fin = "linewidth.elself"
    Fout = "scat_output"
    Fscf = "scf120.out"
    Fvel = "tt_geninterp.dat"
    Fds = 'ds'
    # NK = 721
    NK = 1261
    # NK = 2791
    # NK = 7651
    BNDlist = [0]
    NBNDout = len(BNDlist) 
    NBND = 4
    NKlen = 120
    NKtot = NKlen**2
elif ds == 2:
    prefix = "./MoSe2_k180_tri/"
    Fin = "linewidth.elself"
    Fout = "scat_output"
    Fscf = "scf180.out"
    Fvel = "tt_geninterp.dat"
    # NK = 721
    # NK = 1261
    NK = 2791
    # NK = 7651
    BNDlist = [0]
    NBNDout = len(BNDlist) 
    NBND = 4
    NKlen = 180
    NKtot = NKlen**2
elif ds ==  3:
    prefix = "./WS2/"
    Fin = "linewidth.elself"
    Fout = "scat_output"
    Fvel = "tt_geninterp.dat"
    Fds = 'ds'
    Fscf = "scf120.out"
    NK = 1261
    NKlen = 120
    NKtot = NKlen**2
    BNDlist = [0] 
    NBNDout = len(BNDlist)
    NBND = 4 

def get_kfrac(ik_list):
    NK = NKlen
    kcart_list = np.zeros((len(ik_list),2))
    for ik in range(len(ik_list)):
        kcart_list[ik][0] = float(ik_list[ik]//NK)/NK 
        kcart_list[ik][1] = float(ik_list[ik]%NK)/NK 

    return kcart_list

def kindex_add(ik, iq):
    NK = NKlen
    iky = ik%NK 
    ikx = ik//NK 
    iqy = iq%NK
    iqx = iq//NK 

    jky = (iky+iqy)%NK 
    jkx = (ikx+iqx)%NK 

    return jkx*NK + jky 

def get_kindex2(kfrac_list):
    ik_list = np.zeros((len(kfrac_list),), dtype=int)
    for ik in range(len(kfrac_list)):
        ikx = int(round(kfrac_list[ik][0] * NKlen))%NKlen
        iky = int(round(kfrac_list[ik][1] * NKlen))%NKlen
        ik_list[ik] = iky + ikx * NKlen

    return ik_list

def floor_list(in_ndarray):
    out_ndarray = in_ndarray.reshape(-1).copy()
    for i in range(len(out_ndarray)):
        out_ndarray[i] = out_ndarray[i] - floor(out_ndarray[i])

    out_ndarray = np.reshape(out_ndarray, np.shape(in_ndarray))

    return out_ndarray

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

    scat = scat*0.001
    bande = bande - np.min(bande)
    return scat, bande 

def writer_scat(filename, scat, bande):
    fw = open(filename, 'w')
    for ik in range(len(scat)):
        for ibnd in range(len(scat[0])):
            fw.write("%20.12f %20.12f \n" %(bande[ik][ibnd], scat[ik][ibnd]))

    fw.close()

def reader_scfout(FILENAME, ndim=2):
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

    return sym_matrix, 2*pi*irrckx, irrwk, irrfkx

def reader_fort780():
    try:
        fo = open(prefix+'fort.780', 'r')
    except:
        raise Exception("fort.780 not exists!")

    lines = fo.readlines()

    nqtot = int(lines[-1].split()[0])
    nmode = int(lines[-1].split()[1])

    phonone = np.zeros((nqtot,nmode))
    for line in lines:
        iq = int(line.split()[0]) - 1
        imode = int(line.split()[1]) - 1
        phonone[iq][imode] = float(line.split()[2])

    fo.close()
    return phonone

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

def dmob(vx, vy, tau, ke):
    kT = 0.025852

    f = 1.0/(e**(ke/kT)+1)
    dm = vx*vy*(1-f)*f/kT*tau 
    dn = f 

    return dm, dn 

def mobility(bande, velocity, scat, bz2ibz, units=1):
    # if units == 0: bande in ev; velocity in cm/s; scat in s-1
    # if units == 1: bande in ev; velocity in eV/A; scat in eV

    if units == 1:
        velocity = np.array(velocity)*1.519267582e7
        scat = np.array(scat)*2/6.5821e-16

    fermi = -0.1
    _bande = bande - fermi 

    ncarrier = 0.0
    mob = 0.0
    
    if np.all(bz2ibz) != None:
        for ik in range(len(velocity)):
            for ibnd in range(len(velocity[0])):
                vx = velocity[ik][ibnd][0] 
                tau = 1.0/scat[bz2ibz[ik]][ibnd] 
                ke = _bande[bz2ibz[ik]][ibnd]
                dm, dn = dmob(vx, vx, tau, ke) 
                mob += dm 
                ncarrier += dn 
    else:
        for ik in range(len(velocity)):
            for ibnd in range(len(velocity[0])):
                vx = velocity[ik][ibnd][0] 
                tau = 1.0/scat[ik][ibnd] 
                ke = _bande[ik][ibnd] 
                dm, dn = dmob(vx, vx, tau, ke) 
                mob += dm 
                ncarrier += dn 

    return mob/ncarrier 

def reader_restart(prefix="datasets"):
    trans = np.load(prefix+"_trans.npy" , allow_pickle=True)
    index = np.load(prefix+"_index.npy", allow_pickle=True)
    scat = np.load(prefix+"_scat.npy", allow_pickle=True)

    return trans, index, scat

def writer_data(trans, index, scat, prefix="datasets"):

    np.save(prefix+"_trans.npy", trans)
    np.save(prefix+"_index.npy", index)
    np.save(prefix+"_scat.npy", scat)


def distri(ke, type=0):
    kT = 0.025852
    # kT = 0.03

    f = 1.0/(e**(ke/kT)+1)

    return f 

def check_ejump(trans, index, scat, bandeirr, bz2ibz, irrwk, plot=0):
    nkirr = NK 
    ibz2bz = np.zeros((nkirr, ), dtype=int)

    ibz2bz = [np.argwhere(bz2ibz==i)[0][0] for i in range(nkirr)]

    ejump = np.zeros((nkirr,NBNDout))

    for ik in range(nkirr):
        for ibnd in range(NBNDout):
            fstat = [bz2ibz[kindex_add(ibz2bz[ik], index[ik][ibnd][j][0])] for j in range(len(index[ik][ibnd]))]
            fbnd = [index[ik][ibnd][j][1] for j in range(len(index[ik][ibnd]))]

            ejump_bz = bandeirr[fstat, fbnd] - bandeirr[ik][ibnd] 
            ejump[ik][ibnd] = np.average(ejump_bz, weights=trans[ik][ibnd])
    # test
    # ejump[1260][0] = 0.014 

    if plot == 1:
        fw = open(prefix+Fout, 'w')
        for ik in range(nkirr):
            for ibnd in range(NBNDout):
                fw.write("%20.12e %20.12e \n"%(bandeirr[ik][ibnd], ejump[ik][ibnd]))

        fw.close()

    etmp = bandeirr+0.5
    weight = np.zeros((nkirr, NBNDout))
    for ik in range(nkirr):
        for ibnd in range(NBNDout):
            weight[ik][ibnd] = distri(etmp[ik][ibnd])*scat[ik][ibnd]*irrwk[ik]

    ejump_avg = np.average(ejump, weights=weight/np.sum(weight)) 
    print("average energy jump: ", ejump_avg)

def printer_matrix(m):
    fw = open(prefix+"matrix_output", 'w')
    m = np.array(m) 
    if len(np.shape(m)) == 2:
        if len(m)<len(m[0]):
            m = np.transpose(m)
        for ik in range(len(m)):
            for ibnd in range(len(m[0])):
                fw.write("%20.12f "%(m[ik][ibnd]))

            fw.write("\n")
    elif len(np.shape(m)) == 1:
        for ik in range(len(m)):
            fw.write("%20.12f \n"%m[ik])

    fw.close() 


def check_detailed_balance(ik, trans, index, scat, bande, bz2ibz=None, bz_sym=None, sym_matrix=None):
    kT = 0.025852

    if np.all(bz2ibz) == None:
        nk = len(trans[ik][0]) # assume band = 0
        ktrans = trans[ik][0]*scat[ik][0]
        kqtrans = np.zeros((nk, ))
        for i in range(nk):
            iq = index[ik][0][i][0]
            iqy = iq%NKlen
            iqx = iq//NKlen 
            imqy = (NKlen-iqy)%NKlen 
            imqx = (NKlen-iqx)%NKlen 
            minus_iq = imqx*NKlen + imqy 
            ikq = kindex_add(ik, iq)

            kq2k_index = np.where(index[ikq][0][:,0] == minus_iq)[0]

            if np.size(kq2k_index)>0:
                kq2k_index = kq2k_index[0]
                _kqtrans = trans[ikq][0][kq2k_index]*scat[ikq][0]
                _kqtrans = _kqtrans*(e**((bande[ik][0]-bande[ikq][0])/kT))
            else:
                print("cannot kq to k transition! ") 
                _kqtrans = 0.0 

            kqtrans[i] = _kqtrans 
    else:
        ik_irr = bz2ibz[ik]
        nk = len(trans[ik_irr][0]) # assume band = 0
        ktrans = trans[ik_irr][0]*scat[ik_irr][0]
        kqtrans = np.zeros((nk, ))
        ibnd = 0
        ibz2bz = np.zeros((NK, ), dtype=int)
        ibz2bz = [np.argwhere(bz2ibz==i)[0][0] for i in range(NK)]
        sym_matrix_inv = [np.linalg.inv(sym_matrix[i]) for i in range(len(sym_matrix))]
        sym_matrix_inv = np.array(sym_matrix_inv)
        for i in range(nk):
            iq = index[ik_irr][ibnd][i][0]
            jbnd = index[ik_irr][ibnd][i][1]
            # ik = ibz2bz[ik_irr]
            ikq = kindex_add(ibz2bz[ik_irr], iq)
            ikq_irr = bz2ibz[ikq]

            minus_qfx = np.dot(sym_matrix_inv[bz_sym[ikq]], get_kfrac([iq])[0]*-1.0)
            minus_iq = get_kindex2([floor_list(minus_qfx)])[0]

            kq2k_index = np.where(index[ikq_irr][jbnd][:,0] == minus_iq)[0]

            if np.size(kq2k_index)>0:
                kq2k_index = kq2k_index[0]
                _kqtrans = trans[ikq_irr][0][kq2k_index]*scat[ikq_irr][0]
                _kqtrans = _kqtrans*(e**((bande[ik_irr][0]-bande[ikq_irr][0])/kT))
            else:
                print("cannot kq to k transition! ") 
                _kqtrans = 0.0 

            kqtrans[i] = _kqtrans 


    printer_matrix([ktrans, kqtrans])


    for i in range(nk):
        p1 = kqtrans[i]
        p2 = ktrans[i]

        if p1 > 1e-30:
            ftmp = p2/p1 

            delta1 = (ftmp-1)/(1+ftmp**2)
            delta2 = ftmp*(1-ftmp)/(1+ftmp**2)

            kqtrans[i] = p1*(1+delta1)
            ktrans[i] = p2*(1+delta2) 

    printer_matrix([ktrans, kqtrans])

reset_test = 1
if reset_test == 1:
    maxdelta1 = 0.0
    maxdelta2 = 0.0

def reset_detailed_balance(trans, index, scat, bande, bz2ibz=None, bz_sym=None, sym_matrix=None):
    global maxdelta1, maxdelta2
    kT = 0.025852

    _trans = copy.deepcopy(trans)
    _index = copy.deepcopy(index)
    _scat = copy.deepcopy(scat)

    if np.all(bz2ibz) == None:
        for ik in range(NK):
            for ibnd in range(NBNDout):
                nk = len(trans[ik][ibnd])

                for i in range(nk):
                    iq = index[ik][ibnd][i][0]
                    jbnd = index[ik][ibnd][i][1]
                    iqy = iq%NKlen
                    iqx = iq//NKlen
                    imqy = (NKlen-iqy)%NKlen
                    imqx = (NKlen-iqx)%NKlen 
                    minus_iq = imqx*NKlen + imqy 
                    ikq = kindex_add(ik, iq)

                    kq2k_index = np.where(index[ikq][jbnd][:,0] == minus_iq)[0]

                    if np.size(kq2k_index)>0:
                        kq2k_index = kq2k_index[0]
                        p1 = trans[ikq][jbnd][kq2k_index]*scat[ikq][jbnd]
                        p1e = p1*(e**((bande[ik][ibnd]-bande[ikq][jbnd])/kT))
                    else:
                        p1e = 0.0 

                    
                    p2 = trans[ik][ibnd][i]*scat[ik][ibnd]
                    # ftmp = p2/p1e 
                    if p1e > 1e-30:
                        ftmp = p2/p1e 
                        delta2 = (1.0+ftmp)/(1.0+ftmp**2)
                        delta1 = ftmp*delta2 
                        if reset_test == 1:
                            if maxdelta1 < abs(delta1-1.0):
                                maxdelta1 = abs(delta1-1.0)
                                print("maxdelta1: ", maxdelta1, "Energy at: ", bande[ikq][jbnd])
                            if maxdelta2 < abs(delta2-1.0):
                                maxdelta2 = abs(delta2-1.0)
                                print("maxdelta2: ", maxdelta2, "Energy at: ", bande[ik][ibnd])

                        _trans[ikq][jbnd][kq2k_index] = p1*delta1 
                        _trans[ik][ibnd][i] = p2*delta2 
                    else:
                        _trans[ikq][jbnd][kq2k_index] = p1
                        _trans[ik][ibnd][i] = p2
    else:
        ibz2bz = np.zeros((NK, ), dtype=int)
        ibz2bz = [np.argwhere(bz2ibz==i)[0][0] for i in range(NK)]
        sym_matrix_inv = [np.linalg.inv(sym_matrix[i]) for i in range(len(sym_matrix))]
        sym_matrix_inv = np.array(sym_matrix_inv)

        for ik_irr in range(NK):
            for ibnd in range(NBNDout):
                nk = len(trans[ik_irr][ibnd])

                for i in range(nk):
                    iq = index[ik_irr][ibnd][i][0]
                    jbnd = index[ik_irr][ibnd][i][1]
                    ik = ibz2bz[ik_irr]
                    ikq = kindex_add(ik, iq)
                    ikq_irr = bz2ibz[ikq]

                    minus_qfx = np.dot(sym_matrix_inv[bz_sym[ikq]], get_kfrac([iq])[0]*-1.0)
                    minus_iq = get_kindex2([floor_list(minus_qfx)])[0]

                    kq2k_index = np.where(index[ikq_irr][jbnd][:,0] == minus_iq)[0]

                    if np.size(kq2k_index)>0:
                        kq2k_index = kq2k_index[0]
                        p1 = trans[ikq_irr][jbnd][kq2k_index]*scat[ikq_irr][jbnd]
                        p1e = p1*(e**((bande[ik_irr][ibnd]-bande[ikq_irr][jbnd])/kT))
                    else:
                        p1e = 0.0 

                    p2 = trans[ik_irr][ibnd][i]*scat[ik_irr][ibnd]

                    if p1e > 1e-30:
                        ftmp = p2/p1e 
                        delta2 = (1.0+ftmp)/(1.0+ftmp**2)
                        delta1 = ftmp*delta2 
                        if reset_test == 1:
                            if maxdelta1 < abs(delta1-1.0):
                                maxdelta1 = abs(delta1-1.0)
                                print("maxdelta1: ", maxdelta1, "Energy at: ", bande[ikq_irr][jbnd])
                            if maxdelta2 < abs(delta2-1.0):
                                maxdelta2 = abs(delta2-1.0)
                                print("maxdelta2: ", maxdelta2, "Energy at: ", bande[ik_irr][ibnd])

                        _trans[ikq_irr][jbnd][kq2k_index] = p1*delta1 
                        _trans[ik_irr][ibnd][i] = p2*delta2 
                    else:
                        _trans[ikq_irr][jbnd][kq2k_index] = p1
                        _trans[ik_irr][ibnd][i] = p2
                        
    for ik in range(NK):
        for ibnd in range(NBNDout):
            _scat[ik][ibnd] = np.sum(_trans[ik][ibnd])
            _trans[ik][ibnd] = _trans[ik][ibnd]/_scat[ik][ibnd]

    return _trans, _index, _scat 



if todolist == 1:
    scat, bande = reader_scat(prefix+Fin, NK, NBNDout, BNDlist)
    writer_scat(prefix+Fout, scat, bande)
elif todolist == 2:
    scat, bande = reader_scat(prefix+Fin, NK, NBNDout, BNDlist)
    scat = scat*2/6.5821e-16 # mev to s-1
    sym_matrix, _, irrwk, irrfkx = reader_scfout(prefix+Fscf)
    bz2ibz, _ = kmap_array(sym_matrix, irrfkx)
    _, _bande, velocity = reader_velocity(prefix+Fvel,NKtot, NBNDout, NBND, BNDlist)
    eva2cms = 1.519267582e7 
    velocity = velocity*eva2cms

    mob = mobility(bande, velocity, scat, bz2ibz)
    print(prefix+Fin+": ", mob, " cm2/(Vs)")
elif todolist == 3:
    trans, index, scat = reader_restart(prefix+Fds)
    scat_lw, bandeirr = reader_scat(prefix+Fin, NK, NBNDout, BNDlist)
    sym_matrix, _, irrwk, irrfkx = reader_scfout(prefix+Fscf)
    bz2ibz, _ = kmap_array(sym_matrix, irrfkx)

    _, bandefbz, velocity = reader_velocity(prefix+Fvel,NKtot, NBNDout, NBND, BNDlist)

    ibz2bz = np.zeros((NK, ), dtype=int)
    ibz2bz = [np.argwhere(bz2ibz==i)[0][0] for i in range(NK)]
    bandeirr = np.array([bandefbz[ibz2bz[ik]] for ik in range(NK)])

    check_ejump(trans, index, scat, bandeirr, bz2ibz, irrwk, 1)
elif todolist == 4:
    if restart == 0:
        trans, index, scat = reader_restart(prefix+Fds)
        scat_lw, bande_lw = reader_scat(prefix+Fin, NK, NBNDout, BNDlist)
        scat_lw = scat_lw*2/6.5821e-16 # ev to s-1
        _, bande_wan, velocity = reader_velocity(prefix+Fvel, NKtot, NBNDout, NBND, BNDlist)
        eva2cms = 1.519267582e7 
        velocity = velocity*eva2cms
        
        # mob = mobility(bande_wan, velocity, scat_lw, None, units=0)
        # print(prefix+Fin+": ", mob, " cm2/(Vs)")
        
        # bande_lw = bande_lw - np.min(bande_lw)
        # bande_wan = bande_wan - np.min(bande_wan)
        # tmp = np.zeros((NK, 2))
        # tmp[:,0] = bande_lw[:,0] 
        # tmp[:,1] = bande_wan[:,0]
        # printer_matrix(tmp)

        # phonone = reader_fort780()

        # for i in [4840, 4841, 4842, 4843, 4844, 4845]:
        #     check_detailed_balance(i, trans, index, scat, bande_wan)

        trans_new, index_new, scat_new = reset_detailed_balance(trans, index, scat, bande_lw)

        printer_matrix([bande_lw[:,0], scat_lw[:,0], scat[:,0], scat_new[:,0]])

        writer_data(trans_new, index_new, scat_new, prefix+'ds_new')

    elif restart == 1:
        # trans, index, scat = reader_restart(prefix+Fds)
        Fds = 'ds_new'
        trans_new, index_new, scat_new = reader_restart(prefix+Fds)
        scat_lw, bande_lw = reader_scat(prefix+Fin, NK, NBNDout, BNDlist)

        # printer_matrix([bande_lw[:,0], scat_lw[:,0], scat[:,0], scat_new[:,0]])

        _, bande_wan, velocity = reader_velocity(prefix+Fvel, NKtot, NBNDout, NBND, BNDlist)
        eva2cms = 1.519267582e7 
        velocity = velocity*eva2cms

        mob = mobility(bande_lw, velocity, scat_new*2/6.5821e-16, None)
        print(prefix+Fin+": ", mob, " cm2/(Vs)")

elif todolist == 5:
    if restart == 0:
        trans, index, scat = reader_restart(prefix+Fds)
        scat_lw, bande_lw = reader_scat(prefix+Fin, NK, NBNDout, BNDlist)
        _, bande_wan, velocity = reader_velocity(prefix+Fvel, NKtot, NBNDout, NBND, BNDlist)
        sym_matrix, _, irrwk, irrfkx = reader_scfout(prefix+Fscf)
        bz2ibz, bz_sym = kmap_array(sym_matrix, irrfkx)

        mob = mobility(bande_lw, velocity, scat, bz2ibz)
        print(prefix+Fin+": ", mob, " cm2/(Vs)")      

        # for i in [4840, 4841, 4842, 4843, 4844, 4845]:
        #     check_detailed_balance(i, trans, index, scat, bande_lw, bz2ibz, bz_sym, sym_matrix)   

        trans_new, index_new, scat_new = reset_detailed_balance(trans, index, scat, bande_lw, bz2ibz, bz_sym, sym_matrix)

        printer_matrix([bande_lw[:,0], scat_lw[:,0]/2*6.5821e-4, scat[:,0], scat_new[:,0]])

        mob = mobility(bande_lw, velocity, scat_new, bz2ibz)
        print(prefix+"ds_mew: ", mob, " cm2/(Vs)")  

        writer_data(trans_new, index_new, scat_new, prefix+'ds_new')

