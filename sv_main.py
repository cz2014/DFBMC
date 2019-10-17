import numpy as np 
import sv_class as sv 

eflist1 = [[0,0],[1e5,0],[3e5,0],[6e5,0],[10e5,0],[15e5,0],[2e6,0],[3e6,0],[4e6,0],[7e6,0],[10e6,0]]
eflist2 = [[-1e5,0],[-3e5,0]]
eflist3 = [[13e6,0],[16e6,0],[20e6,0],[25e6,0],[30e6,0]]
eflist4 = [[0,0],[0,3e5],[0,6e5],[0,15e5],[0,3e6],[0,7e6],[0,13e6],[0,20e6],[0,30e6]]
eflist5 = [[0,0],[1e5,0],[3e5,0],[6e5,0],[10e5,0],[15e5,0],[2e6,0],[3e6,0],[4e6,0],[7e6,0],[10e6,0],[13e6,0],[16e6,0],[20e6,0],[25e6,0],[30e6,0]]
eflist6 = [[0,0],[3e5,0],[6e5,0],[15e5,0],[3e6,0],[7e6,0],[13e6,0],[20e6,0],[30e6,0]]
eflist7 = [[30e6,0]]
eflist8 = [[0,0],[3e5,0],[5e5,0],[7e5,0],[9e5,0],[11e5,0],[13e5,0],[15e5,0]]
eflist10 = [[0,0],[1e4,0],[2e4,0],[3e4,0],[4e4,0],[5e4,0],[6e4,0],[7e4,0],[8e4,0],[9e4,0],[10e4,0]]

eflist9 = np.zeros((5,2))
eflist9[:,0] = np.linspace(3.97e6, 15e6, num=5, endpoint=True)

lattice_para = np.zeros((5,2,2))
lattice_para[0] = [[0.314027,  0.181303],[0.000000,  0.362607]] # MoS2
lattice_para[1] = [[0.301267,  0.173937],[0.000000,  0.347873]] # MoSe2 
lattice_para[2] = [[0.280915,  0.162186],[0.000000,  0.324372]] # MoTe2 
lattice_para[3] = [[0.313724, 0.181128],[0.000000,  0.362257]] # WS2
lattice_para[4] = [[0.301183, 0.173888 ],[0.000000,  0.347776]] # WSe2
 
sv_test = sv.MCseries(
    NK=120, 
    NBND=1, 
    NBNDv=4, 
    reci_vec=lattice_para[3], 
    prefix='./WS2',
    test_mode=0,
    # governed by test_mode
    # stat_e=1,
    # plot_e=1,
    # stat_v=1,
    # end
    nelec=1,
    ds_file='ds_new',
    # ds_file='ds180_4bnd_cut1e-8',
    wan_file='tt_geninterp.dat',
    scf_file='scf120.out',
    Wan_Band_List=[0],
    selfscat=1,
    restart=2,
    irrbz=1,
    sp_fermi=1,
    elecf_list=eflist9,
    totstep=10000,
    # output='sv.out',
    # enable_para=True,
    # Nprocess=4,
    iverbosity=0
    )

if __name__ == '__main__':
    sv_test.MCseries()