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


sv_test = sv.MCseries(
    NK=120, 
    NBND=1, 
    NBNDv=4, 
    reci_vec=[[0.314027,  0.181303],[0.000000,  0.362607]], 
    prefix='./MoSe2_tri',
    test_mode=2,
    # governed by test_mode
    # stat_e=1,
    # plot_e=1,
    # stat_v=1,
    # end
    nelec=1,
    ds_file='ds90_1bnd_cut1e-8',
    wan_file='tt_geninterp.dat',
    scf_file='scf120.out',
    Wan_Band_List=[0],
    selfscat=1,
    restart=2,
    irrbz=1,
    sp_fermi=1,
    elecf_list=eflist8,
    totstep=10000,
    # output='sv.out',
    # enable_para=True,
    # Nprocess=4
    )

if __name__ == '__main__':
    sv_test.MCseries()