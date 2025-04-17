import math

import numpy as np

import ndsl.constants as constants
import ndsl.dsl.gt4py_utils as utils
from ndsl import CubedSphereCommunicator, QuantityFactory
from ndsl.grid import GridData
from ndsl.grid.gnomonic import great_circle_distance_lon_lat, lon_lat_midpoint
from pyFV3.dycore_state import DycoreState
from pyFV3.initialization import init_utils

NHALO = constants.N_HALO_DEFAULT # TODO: Where to put this?
SURFACE_PRESSURE = 1.0e5  # units of (Pa), from Table VI of DCMIP2016 #TODO: where to put this?
OMG = 7.848e-6
RK    = 7.848e-6
R = 4.0 # Wave Number 4 (likely)
UBAR = 0.0 # TODO: Remove? Not yet used?
GH0 = 8.0e3 * constants.GRAV


def preinit_for_all_sw(state: DycoreState,
                       shape,
                       grid_data: GridData
):
    """
    Pre-initialization from test_cases.F90 that applies to all shallow water tests

    Args:
        state: DycoreState - modified
        shape: TODO
    """
    
    nx, ny, nz = init_utils.local_compute_size(shape)
    
    state.pe[:] = 0.0
    state.pt[:] = 1.0

    # TODO: Do we have to initialize f0, fC?
    
    # Initialize Halo Corners
    state.delp[:NHALO, :NHALO] = 0.0
    state.delp[:NHALO, NHALO + ny :] = 0.0
    state.delp[NHALO + nx :, :NHALO] = 0.0
    state.delp[NHALO + nx :, NHALO + ny :] = 0.0
    
    # TODO: Below, taken from baroclinic test. Do we need these?
    state.ua[:] = 1e35
    state.va[:] = 1e35
    state.uc[:] = 1e30
    state.vc[:] = 1e30
    state.w[:] = 0 # vertical component of the wind JKNOTE
    state.delz[:] = 1.0e25
    state.phis[:] = 1.0e25
    state.ps[:] = SURFACE_PRESSURE
    eta = np.zeros(nz)
    eta_v = np.zeros(nz)
    islice, jslice, slice_3d, slice_2d = init_utils.compute_slices(nx, ny)

    # TODO: setup_pressure_fields may need eta file. What is that?
    init_utils.setup_pressure_fields(
        eta=eta,
        eta_v=eta_v,
        delp=state.delp[slice_3d],
        ps=state.ps[slice_2d],
        pe=state.pe[slice_3d],
        peln=state.peln[slice_3d],
        pk=state.pk[slice_3d],
        pkz=state.pkz[slice_3d],
        ak=utils.asarray(grid_data.ak.data),
        bk=utils.asarray(grid_data.bk.data),
        ptop=grid_data.ptop,
    )


def init_rhwave_winds(p1, p2): # TODO document p1, p2
    """ TODO
    """
    muv = init_utils._find_midpoint_unit_vectors(p1, p2)
    p3 = muv["midpoint"]
    e2 = muv["unit_dir"] 
    ex = muv["exv"] 
    ey = muv["eyv"]
    utmp = (
        constants.RADIUS * OMG * np.cos(p3[:, :, 1]) + constants.RADIUS * RK
        * (np.cos(p3[:, :, 1])**(R-1))
        * (R * np.sin(p3[:, :, 1])**2 - np.cos(p3[:, :, 1])**2)*np.cos(R*p3[:, :, 0])
    )
    vtmp = (
        -1 * constants.RADIUS * RK * R * np.sin(p3[:, :, 1])
        * np.sin(R * p3[:, :, 0]) * np.cos(p3[:, :, 1])**(R-1)
    )
    return utmp * np.sum(e2 * ex, 2) + vtmp * np.sum(e2 * ey, 2)


def init_rhwave_delp(state: DycoreState,
                     grid_data: GridData
):
    agd0 = grid_data.lon_agrid.data[:]
    agd1 = grid_data.lat_agrid.data[:]    
    A = (0.5 * OMG * (2 * constants.OMEGA + OMG) * (np.cos(agd1)**2)
         + 0.25 * RK * RK * (np.cos(agd1)**(R + R))
         * ((R + 1) * (np.cos(agd1)**2) + (2 * R * R - R - 2) - 2 * (R * R) * np.cos(agd1)**(-2)))
    B = ((2 * (constants.OMEGA + OMG) * RK / ((R+1) * (R+2)))
         * (np.cos(agd1)**R) * ((R*R+2 * R + 2) - ((R + 1) * np.cos(agd1))**2 ))
    C = 0.25 * RK * RK * (np.cos(agd1)**(2 * R)) * ((R + 1) * (np.cos(agd1)**2) - (R+2))    
    return (GH0 + constants.RADIUS * constants.RADIUS
            * ( A + B * np.cos(R * agd0) + C * np.cos(2 * R * agd0)))


def init_for_rhwave(state: DycoreState,
                    grid_data: GridData
):
    """
    Initialization specific to Rossby Wave number 4 from test_cases.F90

    TODO Update Inputs
    """
    
    state.phis[:] = 0.0

    # Initialize delp
    state.delp[:,:,0] = init_rhwave_delp(state, grid_data)
    state.delp[:,:,0] = state.delp[:,:,0] - state.phis[:]

    # TODO: Check why p1, p2 from grid is different in baroclinic example (pa1, pa2)
    grid = np.transpose(
        np.stack(
            [grid_data._horizontal_data.lon.data, grid_data._horizontal_data.lat.data]
        ),
        [1, 2, 0],
    )

    # Initialize u winds
    p1 = grid[:-1, :, :]
    p2 = grid[1:, :, :]
    state.u[:-1, :, 0] = init_rhwave_winds(p1, p2)

    # Initialize v winds
    p1 = grid[:, :-1, :]
    p2 = grid[:, 1:, :]
    state.v[:, :-1, 0] = init_rhwave_winds(p1, p2)
    
    # TODO: Pay attention to the slice indices. u and v are similarly calculated.

    """ From test_cases.F90 (case 6): 
         call mp_update_dwinds(u, v, npx, npy, npz, domain, bd)
         call dtoa( u, v,ua,va,dx,dy,dxa,dya,dxc,dyc,npx,npy,ng,bd)
         !call mpp_update_domains( ua, va, domain, gridtype=AGRID_PARAM)
         call atoc(ua,va,uc,vc,dx,dy,dxa,dya,npx,npy,ng, gridstruct%bounded_domain, domain, bd)
         initWindsCase=initWindsCase6
    """

    
def postinit_for_all_sw(state):
    """ Post-initialization from test_cases.F90 that applies to all shallow water tests

    Args:
        state: DycoreState - modified
    """
    
    ''' TODO
      cl = get_tracer_index(MODEL_ATMOS, 'cl')
      cl2 = get_tracer_index(MODEL_ATMOS, 'cl2')
      if (cl > 0 .and. cl2 > 0) then
         call terminator_tracers(is,ie,js,je,isd,ied,jsd,jed,npz, &
              q, delp,ncnst,agrid(isd:ied,jsd:jed,1),agrid(isd:ied,jsd:jed,2),bd)
         call mpp_update_domains(q,domain)
      endif
    '''

    state.delp[:,:,1:] = state.delp[:,:,0][:,:,np.newaxis]
        
    ''' TODO:
      phi0  = delp

      call init_winds(UBAR, u,v,ua,va,uc,vc, initWindsCase, npx, npy, ng, ndims, nregions, gridstruct%bounded_domain, gridstruct, domain, tile, bd)
    '''

    state.u[:,:,1:] = state.u[:,:,0][:,:,np.newaxis]
    state.v[:,:,1:] = state.v[:,:,0][:,:,np.newaxis]
    
    state.ps[:] = state.delp[:,:,0]
    
    '''
    call mp_update_dwinds(u, v, npx, npy, npz, domain, bd)
    '''    

    
def init_rhwave_state(
    grid_data: GridData,
    quantity_factory: QuantityFactory,
    hydrostatic: bool,
    comm: CubedSphereCommunicator,
) -> DycoreState:
    """
    Create a DycoreState TODO: explain more

    TODO Fix Inputs, For now, taken from baroclinic

    Args:
        grid_data:              current selected grid data values
        quantity_factory:       QuantityFactory
        hydrostatic:            flag for hydrostatic methods
        comm:                   CubedSphereCommunicator

    Returns:
        DycoreState
    """
    sample_quantity = grid_data.lat
    shape = (*sample_quantity.data.shape[0:2], grid_data.ak.data.shape[0])
    numpy_state = init_utils.empty_numpy_dycore_state(shape)

    preinit_for_all_sw(numpy_state, shape, grid_data)
    init_for_rhwave(numpy_state, grid_data)
    postinit_for_all_sw(numpy_state)

    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    comm.halo_update(state.phis, n_points=NHALO)
    comm.vector_halo_update(state.u, state.v, n_points=NHALO)
    # TODO: anymore comm updates? delp?

    return state
