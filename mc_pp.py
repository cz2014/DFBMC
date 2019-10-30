import pickle 
import numpy as np 
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import copy 
from math import pi
import os 

prefix = './WS2_strained/'

class MC_Datasets():
    def __init__(self):
        self.nelec = 0
        self.irrbz = 0
        self.NK = 0
        self.NBNDv = 0
        self.NBND = 0
        self.Wan_Band_List = []
        self.reci_vec = np.zeros((2,2))
        self.totstep = 0
        self.elecf_list = []
        self.sbs_kint = []
        self.sbs_bnd = []
        self.weight_t = []
        self.bande = []
        self.velocity = []
        self.scat = []

def get_kfrac(ik_list):
    NK = NKlen
    kcart_list = np.zeros((len(ik_list),2))
    for ik in range(len(ik_list)):
        kcart_list[ik][0] = float(ik_list[ik]//NK)/NK 
        kcart_list[ik][1] = float(ik_list[ik]%NK)/NK 

    return kcart_list

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

    velocity = velocity*1.519267582e7 
    bande = bande - np.min(bande)
    return kvec, bande, velocity

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

def get_kindex2(kfrac_list):
    ik_list = np.zeros((len(kfrac_list),), dtype=int)
    for ik in range(len(kfrac_list)):
        ikx = int(round(kfrac_list[ik][0] * NKlen))%NKlen
        iky = int(round(kfrac_list[ik][1] * NKlen))%NKlen
        ik_list[ik] = iky + ikx * NKlen

    return ik_list

def kmap_array(sym_matrix, irrfkx):
    NKtot = NKlen*NKlen 
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

def reader_stat(filename, ds):
    f = open(filename, 'rb')

    ds.nelec = pickle.load(f)
    ds.irrbz = pickle.load(f)
    ds.NK = pickle.load(f)
    ds.NBNDv = pickle.load(f)
    ds.NBND = pickle.load(f)
    ds.Wan_Band_List = pickle.load(f)
    ds.reci_vec = pickle.load(f)
    ds.totstep = pickle.load(f)
    ds.elecf_list = pickle.load(f)

    ds.sbs_kint = []
    ds.sbs_bnd = []
    ds.sbs_t = []
    ds.weight_t = []
    for i in range(len(ds.elecf_list)):
        ds.sbs_kint.append(pickle.load(f))
        ds.sbs_bnd.append(pickle.load(f))
        ds.sbs_t.append(pickle.load(f))
        ds.weight_t.append(pickle.load(f))
    
    f.close()

def test_check(ds):
    for kint_list in ds.sbs_kint:
        icross = 0
        for i in range(ds.totstep-1):
            if kint_list[2*i+1] != kint_list[2*i+2]:
                icross += 1

        print(icross)

def get_mean_filter(f, ds, ief, stat_dict):
    bool_filter = [f(x) for x in ds.sbs_kint[ief]]
    new_kint = np.extract(bool_filter, ds.sbs_kint[ief])
    new_bnd = np.extract(bool_filter, ds.sbs_bnd[ief])
    new_weight = np.extract(bool_filter, ds.weight_t[ief]) 
    new_bande = ds.bande[new_kint, new_bnd]
    new_vx = ds.velocity[new_kint, new_bnd, 0]
    new_vy = ds.velocity[new_kint, new_bnd, 1]
    new_scat = ds.scat[new_kint, new_bnd]

    if np.sum(new_weight) < 1e-30:
        stat_dict['Vx(cm/s)'].append(0.0)
        stat_dict['Vy(cm/s)'].append(0.0)
        stat_dict['Energy(eV)'].append(0.0)
        stat_dict['Population'].append(0.0)
        stat_dict['Scattering(eV)'].append(0.0)
    else:
        stat_dict['Vx(cm/s)'].append(np.average(new_vx, weights=new_weight))
        stat_dict['Vy(cm/s)'].append(np.average(new_vy, weights=new_weight))
        stat_dict['Energy(eV)'].append(np.average(new_bande, weights=new_weight))
        stat_dict['Population'].append(np.sum(new_weight)/np.sum(ds.weight_t[ief]))
        stat_dict['Scattering(eV)'].append(np.average(new_scat, weights=new_weight))
    

def stat_under_filter(f, ds):
    stat_dict = {}
    stat_dict['Electric Field(V/m)'] = ds.elecf_list[:, 0]
    stat_dict['Vx(cm/s)'] = list()
    stat_dict['Vy(cm/s)'] = list()
    stat_dict['Energy(eV)'] = list()
    stat_dict['Population'] = list()
    stat_dict['Scattering(eV)'] = list()

    for i in range(len(ds.elecf_list)):
        get_mean_filter(f, ds, i, stat_dict) 

    return stat_dict 

def hist_E_filter(f, ds):
    for ief in range(len(ds.elecf_list)):
        bool_filter = [f(x) for x in ds.sbs_kint[ief]]
        new_kint = np.extract(bool_filter, ds.sbs_kint[ief])
        new_bnd = np.extract(bool_filter, ds.sbs_bnd[ief])
        new_weight = np.extract(bool_filter, ds.weight_t[ief]) 
        new_bande = ds.bande[new_kint, new_bnd]

        plt.hist(new_bande, weights=new_weight, bins=50)
        plt.title('E = '+str(ds.elecf_list[ief][0]))
        plt.show()
        plt.close()

    
def printer_dict(stat_dict):
    for key in stat_dict.keys():
        print(key, end='    ')
    print('')

    for i in range(len(stat_dict['Electric Field(V/m)'])):
        for key in stat_dict.keys():
            print("% 12.8e" %(stat_dict[key][i]), end='    ')
        print(' ')
        
def plot_energy_under_filter(f, ds, save=False):
    if save and not os.path.exists(prefix+'plot_energy'):
        os.makedirs(prefix+'plot_energy')

    res = []
    for ief in range(len(ds.elecf_list)):
        bool_filter = [f(x) for x in ds.sbs_kint[ief]]
        new_kint = np.extract(bool_filter, ds.sbs_kint[ief])
        new_bnd = np.extract(bool_filter, ds.sbs_bnd[ief])
        new_weight = np.extract(bool_filter, ds.weight_t[ief]) 
        new_bande = ds.bande[new_kint, new_bnd]

        plt.xlabel('Energy(eV)')
        plt.ylabel('frequency stat')
        res_ief, x, _ = plt.hist(new_bande, bins=30, weights=new_weight, density=True)
        res.append(copy.deepcopy(res_ief))

        if save:
            plt.savefig(prefix+'plot_energy/E_stat_'+str(ief)+'.png')
        else:
            plt.show()
        plt.close() 
    
    x = np.array(x)
    res_x = 0.5*x[1:] + 0.5*x[0:-1]
    for ief in range(len(ds.elecf_list)):
        res[ief] = np.array(res[ief])/np.sum(res[ief])
        plt.plot(res_x, res[ief], label='%6.2e'%(ds.elecf_list[ief][0]))

    plt.xlabel('Energy(eV)')
    plt.ylabel('frequency stat')
    plt.legend()
    plt.show()

# only vx is implemented
def plot_velocity_under_filter(f, ds, save=False):
    if save and not os.path.exists(prefix+'plot_velocity'):
        os.makedirs(prefix+'plot_velocity')

    res = []
    for ief in range(len(ds.elecf_list)):
        bool_filter = [f(x) for x in ds.sbs_kint[ief]]
        new_kint = np.extract(bool_filter, ds.sbs_kint[ief])
        new_bnd = np.extract(bool_filter, ds.sbs_bnd[ief])
        new_weight = np.extract(bool_filter, ds.weight_t[ief]) 
        new_vx = ds.velocity[new_kint, new_bnd, 0]

        plt.xlabel('Velocity(eV)')
        plt.ylabel('frequency stat')
        res_ief, x, _ = plt.hist(new_vx, bins=20, weights=new_weight, density=True)
        res.append(copy.deepcopy(res_ief))

        if save:
            plt.savefig(prefix+'plot_velocity/V_stat_'+str(ief)+'.png')
        else:
            plt.show()
        plt.close() 
    
    fig = plt.figure()
    ax = plt.subplot(111)
    x = np.array(x)
    res_x = 0.5*x[1:] + 0.5*x[0:-1]
    for ief in range(len(ds.elecf_list)):
        res[ief] = np.array(res[ief])/np.sum(res[ief])
        plt.plot(res_x, res[ief], label='%6.2e'%(ds.elecf_list[ief][0]))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.xlabel('Velocity(eV)')
    plt.ylabel('frequency stat')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def scat_under_filter(f1, f2, ds):
    counter = np.zeros((len(ds.elecf_list), ))
    for ief in range(len(ds.elecf_list)):
        for i in range(ds.totstep):
            ik = ds.sbs_kint[ief][2*i]
            jk = ds.sbs_kint[ief][2*i+1]
            if f1(ik) and f2(jk):
                counter[ief] += 1

    # plt.plot(ds.elecf_list[:,0], counter)
    # plt.show()

    return counter 

def plot_scat_valley(fk, fq, ds):
    plt.plot(ds.elecf_list[:,0], scat_under_filter(fk, fk, ds), label='K to K')
    plt.plot(ds.elecf_list[:,0], scat_under_filter(fk, fq, ds), label='K to Q')
    plt.plot(ds.elecf_list[:,0], scat_under_filter(fq, fk, ds), label='Q to K')
    plt.plot(ds.elecf_list[:,0], scat_under_filter(fq, fq, ds), label='Q to Q')

    plt.legend()
    plt.show()

def scat_E_under_filter(f1, f2, ds):
    E_shift = np.zeros((len(ds.elecf_list), ))
    for ief in range(len(ds.elecf_list)):
        for i in range(ds.totstep):
            ik = ds.sbs_kint[ief][2*i]
            ibnd = ds.sbs_bnd[ief][2*i]
            jk = ds.sbs_kint[ief][2*i+1]
            jbnd = ds.sbs_bnd[ief][2*i+1]
            if f1(ik) and f2(jk):
                E_shift[ief] += ds.bande[jk][jbnd] - ds.bande[ik][ibnd]

    return E_shift

def plot_scat_E_valley(fk, fq, ds):
    plt.plot(ds.elecf_list[:,0], scat_E_under_filter(fk, fk, ds), label='K to K')
    plt.plot(ds.elecf_list[:,0], scat_E_under_filter(fk, fq, ds), label='K to Q')
    plt.plot(ds.elecf_list[:,0], scat_E_under_filter(fq, fk, ds), label='Q to K')
    plt.plot(ds.elecf_list[:,0], scat_E_under_filter(fq, fq, ds), label='Q to Q')

    plt.legend()
    plt.show()

# def transfer_E_under_filter(f1, f2, ds):
#     E_shift = np.zeros((len(ds.elecf_list), ))
#     for ief in range(len(ds.elecf_list)):
#         for i in range(ds.totstep):
#             ik = ds.sbs_kint[ief][2*i]
#             jk = ds.sbs_kint[ief][2*i+1]
#             jbnd = ds.sbs_bnd[ief][2*i+1]
#             if f1(ik) and f2(jk):
#                 E_shift[ief] += ds.bande[jk][jbnd]

#     return E_shift

def plot_transfer_E_valley(fk, fq, ds, plot=True):
    q_lost = np.zeros((len(ds.elecf_list), ))
    for ief in range(len(ds.elecf_list)):
        for i in range(ds.totstep):
            ik = ds.sbs_kint[ief][2*i]
            ibnd = ds.sbs_bnd[ief][2*i]
            jk = ds.sbs_kint[ief][2*i+1]
            if fq(ik) and fk(jk):
                q_lost[ief] += ds.bande[ik][ibnd]

    q_gain = np.zeros((len(ds.elecf_list), ))
    for ief in range(len(ds.elecf_list)):
        for i in range(ds.totstep):
            ik = ds.sbs_kint[ief][2*i]
            jk = ds.sbs_kint[ief][2*i+1]
            jbnd = ds.sbs_bnd[ief][2*i+1]
            if fk(ik) and fq(jk):
                q_gain[ief] += ds.bande[jk][jbnd]

    net_gain = q_gain - q_lost  
    if plot:
        plt.plot(ds.elecf_list[:,0], q_lost, label='valley lost')
        plt.plot(ds.elecf_list[:,0], q_gain, label='valley gain')
        plt.plot(ds.elecf_list[:,0], net_gain, label='valley net gain')

        plt.legend()
        plt.show()
    
    return net_gain

def hist_transfer_E_valley(fk, fq, ds):
    for ief in range(len(ds.elecf_list)):
        E_shift = []
        for i in range(ds.totstep):
            ik = ds.sbs_kint[ief][2*i]
            ibnd = ds.sbs_bnd[ief][2*i]
            jk = ds.sbs_kint[ief][2*i+1]
            jbnd = ds.sbs_bnd[ief][2*i+1]
            if fk(ik) and fq(jk):
                E_shift.append(ds.bande[jk][jbnd]-ds.bande[ik][ibnd])

        plt.hist(E_shift, bins=40)
        plt.title('Electric field(V/s): '+str(ds.elecf_list[ief][0]))
        plt.show()
        plt.close()


def flight_E_under_filter(f1, f2, ds):
    E_shift = np.zeros((len(ds.elecf_list), ))
    for ief in range(len(ds.elecf_list)):
        for i in range(ds.totstep-1):
            ik = ds.sbs_kint[ief][2*i+1]
            ibnd = ds.sbs_bnd[ief][2*i+1]
            jk = ds.sbs_kint[ief][2*i+2]
            jbnd = ds.sbs_bnd[ief][2*i+2]
            if f1(ik) and f2(jk):
                E_shift[ief] += ds.bande[jk][jbnd] - ds.bande[ik][ibnd]

    return E_shift

def plot_flight_E_valley(fk, fq, ds):
    plt.plot(ds.elecf_list[:,0], flight_E_under_filter(fk, fk, ds), label='K to K')
    plt.plot(ds.elecf_list[:,0], flight_E_under_filter(fk, fq, ds), label='K to Q')
    plt.plot(ds.elecf_list[:,0], flight_E_under_filter(fq, fk, ds), label='Q to K')
    plt.plot(ds.elecf_list[:,0], flight_E_under_filter(fq, fq, ds), label='Q to Q')

    plt.legend()
    plt.show()

def plot_valley_gain_and_lost(fk, fq, ds, plot_portion=True):
    flight_gain = flight_E_under_filter(fk, fk, ds) 
    diff_valley_gain = plot_transfer_E_valley(fq, fk, ds, plot=False) 
    same_valley_gain = scat_E_under_filter(fk, fk, ds) 

    flight_gain = np.array(flight_gain)
    diff_valley_lost = -1.0*np.array(diff_valley_gain)
    same_valley_lost = -1.0*np.array(same_valley_gain)

    plt.plot(ds.elecf_list[:,0], flight_gain, label='flight energy gain')
    plt.plot(ds.elecf_list[:,0], diff_valley_lost, label='intervalley lost')
    plt.plot(ds.elecf_list[:,0], same_valley_lost, label='intravalley lost')

    plt.legend()
    plt.show()
    plt.close() 

    if plot_portion:
        diff_valley_lost = diff_valley_lost/flight_gain
        same_valley_lost = same_valley_lost/flight_gain

        plt.plot(ds.elecf_list[:,0], diff_valley_lost, label='intervalley lost')
        plt.plot(ds.elecf_list[:,0], same_valley_lost, label='intravalley lost')

        plt.legend()
        plt.show()
        plt.close() 

    

## different kinds of filter function:
# input k index and a list of objective points, return closest point index and distance
def finder_closest(obj_p, ik):
    fkx = get_kfrac([ik])[0]
    # obj_dis = np.array([np.linalg.norm(np.dot(ds.reci_vec, obj_fkx - fkx)) for obj_fkx in obj_p])
    obj_dis = np.array([np.linalg.norm(np.dot(obj_fkx - fkx, ds.reci_vec)) for obj_fkx in obj_p])

    return np.argmin(obj_dis), np.min(obj_dis)

def classify_KfromQ(ik):
    K_valley = [[1.0/3.0, 1.0/3.0], [2.0/3.0, 2.0/3.0]]
    Q_valley = [[1./6., 1./6.], [1./6., 2./3.], [2./3., 1./6.], [1./3., 5./6.], [5./6., 1./3.], [5./6., 5./6.]]
    _, dis_K = finder_closest(K_valley, ik)
    _, dis_Q = finder_closest(Q_valley, ik)

    if dis_K < dis_Q:
        return True 
    else:
        return False 

def classify_QfromK(ik):
    if classify_KfromQ(ik) is True:
        return False 
    else:
        return True 

## better classification functions:
# init K, Q k index lists:
def init_K_Q():
    k_in_K = np.zeros((ds.NK**2, ), dtype=int)
    k_in_Q = np.zeros((ds.NK**2, ), dtype=int)
    for i in range(ds.NK**2):
        if classify_KfromQ(i):
            k_in_K[i] = 1
        else:
            k_in_Q[i] = 1

    return k_in_K, k_in_Q

def classify_KfromQ_v2(ik):
    if k_in_K[ik] == 1:
        return True 
    else:
        return False 

def classify_QfromK_v2(ik):
    if k_in_Q[ik] == 1:
        return True 
    else:
        return False 

## plot distribution:
def plot_dist(ief, save=False):
    weight_ik = np.zeros((ds.NK**2, ))

    for i in range(ds.totstep):
        weight_ik[ds.sbs_kint[ief][i]] += ds.weight_t[ief][i]

    fkx_list = np.array([np.dot(kx, ds.reci_vec) for kx in get_kfrac(range(ds.NK**2))])

    fig = plt.figure()

    X = np.reshape(fkx_list[:,0], (ds.NK, ds.NK))
    Y = np.reshape(fkx_list[:,1], (ds.NK, ds.NK))
    Z = np.reshape(weight_ik, (ds.NK, ds.NK))

    ## plot in 3d colormap
    # ax = Axes3D(fig)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    # ax.set_xlabel('kx(1/a0)')
    # ax.set_ylabel('ky(1/a0)')
    # ax.set_zlabel('Ennergy(eV)')
    # ax.set_xlim(np.min(fkx_list), np.max(fkx_list))
    # ax.set_ylim(np.min(fkx_list), np.max(fkx_list))

    ## plot in 2d colormap
    plt.xlabel('kx(2pi/a)')
    plt.ylabel('ky(2pi/a)')
    plt.axis('equal')
    cmap_color=plt.cm.get_cmap('RdYlBu_r')
    plt.pcolormesh(X, Y, Z, cmap=cmap_color)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    plt.title('electron distribution')
    plt.annotate(s='Electric field(V/s): '+str(ds.elecf_list[ief][0]), xy=(0,0), xytext=(0,0.02))

    
    if save:
        plt.savefig(prefix+'distri_'+str(ief)+'.png', dpi=400)
    else:
        plt.show()
    plt.close() 


ds = MC_Datasets()
reader_stat(prefix+'mc_stat.pkl', ds)
NKlen = ds.NK 
k_in_K, k_in_Q = init_K_Q()
_, ds.bande, ds.velocity = reader_velocity(prefix+'tt_geninterp.dat', ds.NK**2, ds.NBND, ds.NBNDv, ds.Wan_Band_List)

if ds.irrbz == 1:
    sym_matrix, _, irrwk, irrfkx = reader_scfout(prefix+"scf"+str(NKlen)+".out")
    bz2ibz, bz_sym = kmap_array(sym_matrix, irrfkx)

    _scat, _ = reader_scat(prefix+"linewidth.elself", ds.NK**2, ds.NBND, ds.Wan_Band_List)
    ds.scat = np.array([_scat[bz2ibz[i]] for i in range(ds.NK**2)])
elif ds.irrbz == 0:
    ds.scat, _ = reader_scat(prefix+"linewidth.elself", ds.NK**2, ds.NBND, ds.Wan_Band_List)

def f(x):
    # return classify_KfromQ_v2(x)
    # return classify_QfromK_v2(x) 
    return True 

# res = stat_under_filter(f, ds)
# printer_dict(res)

# plot_velocity_under_filter(classify_KfromQ_v2, ds, save=True) 
# plot_velocity_under_filter(classify_QfromK_v2, ds, save=True) 


# for i in range(len(ds.elecf_list)):
#     plot_dist(i, save=True)
    # plot_dist(i)

# plot_transfer_E_valley(classify_KfromQ_v2, classify_QfromK_v2, ds)
# plot_transfer_E_valley(classify_QfromK_v2, classify_KfromQ_v2, ds)


# hist_transfer_E_valley(classify_KfromQ_v2, classify_KfromQ_v2, ds)
# hist_transfer_E_valley(classify_KfromQ_v2, classify_QfromK_v2, ds)
# hist_transfer_E_valley(classify_QfromK_v2, classify_KfromQ_v2, ds)
# hist_transfer_E_valley(classify_QfromK_v2, classify_QfromK_v2, ds)

# plot_flight_E_valley(classify_KfromQ_v2, classify_QfromK_v2, ds)
# plot_scat_E_valley(classify_KfromQ_v2, classify_QfromK_v2, ds)

# plot_valley_gain_and_lost(classify_KfromQ_v2, classify_QfromK_v2, ds) 

hist_E_filter(classify_KfromQ_v2, ds)