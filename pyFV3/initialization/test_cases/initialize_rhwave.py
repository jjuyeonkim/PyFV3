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
    
    # Initializing to values the Fortran does for easy comparison
    state.pe[:] = 0.0
    state.pt[:] = 1.0

    '''
      f0(:,:) = huge(dummy)
      fC(:,:) = huge(dummy)
      do j=jsd,jed+1
         do i=isd,ied+1
            fC(i,j) = 2.*omega*( -1.*cos(grid(i,j,1))*cos(grid(i,j,2))*sin(alpha) + &
                                     sin(grid(i,j,2))*cos(alpha) )
         enddo
      enddo
      do j=jsd,jed
         do i=isd,ied
            f0(i,j) = 2.*omega*( -1.*cos(agrid(i,j,1))*cos(agrid(i,j,2))*sin(alpha) + &
                                     sin(agrid(i,j,2))*cos(alpha) )
         enddo
      enddo
      call mpp_update_domains( f0, domain )
      if (cubed_sphere) call fill_corners(f0, npx, npy, YDir)#end
    '''
    #fC = grid_data.fC() # TODO: already in gridData?, but I don't know if they match?
    
    # Initialize the halo corners
    state.delp[:NHALO, :NHALO] = 0.0
    state.delp[:NHALO, NHALO + ny :] = 0.0
    state.delp[NHALO + nx :, :NHALO] = 0.0
    state.delp[NHALO + nx :, NHALO + ny :] = 0.0
    
    # TODO: Below, taken from baroclinic test. Do we need these?
    state.ua[:] = 1e35
    state.va[:] = 1e35
    state.uc[:] = 1e30
    state.vc[:] = 1e30
    state.w[:] = 1.0e30
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
    #print(f"***A*** state.delp[:,:,1] ({state.delp[:,:,1].shape})\n{state.delp[:,:,1]}")

    
def init_for_rhwave(state: DycoreState,
                    grid_data: GridData
):
    """
    Initialization specific to Rossby Wave number 4 from test_cases.F90

    TODO Update Inputs eventually; for now, taken from baroclinic initialization

    Inputs lon, lat, lon_agrid, lat_agrid, ee1, ee2, es1, ew2, ptop are defined by the
           grid and can be computed using an instance of the MetricTerms class.
    Inputs eta and eta_v are vertical coordinate columns derived from the ak and bk
           variables, also found in the Metric Terms class.
    """
    # TODO: Where is grav defined? ~/pace/NDSL/ndsl/constants.py?
    ubar = 0.0 # TODO What is ubar?
    gh0 = 8.0e3 * constants.GRAV # TODO: what is gh0
    r = 4.0 # TODO What is r?
    omg = 7.848e-6
    rk    = 7.848e-6
    phis = 0.0 # TODO: Why is this phis scalar but used as 2d array below?
    # TODO: What is agrid(i, j, 2), agrid(i, j, 2)
    # lon_agrid=utils.asarray(grid_data.lon_agrid.data[slice_2d_buffer]),
    # lat_agrid=utils.asarray(grid_data.lat_agrid.data[slice_2d_buffer]),
    agd1 = grid_data.lat_agrid.data[:] # TODO: likely wrong
    #print(f"agd1 ({agd1.shape})\n{agd1}")
    agd2 = grid_data.lon_agrid.data[:] # TODO: likely wrong
    #print(f"agd2 ({agd2.shape})\n{agd2}")
    
    A = (0.5 * omg * (2 * constants.OMEGA + omg) * (np.cos(agd2)**2)
         + 0.25 * rk * rk * (np.cos(agd2)**(r + r))
         * ((r + 1) * (np.cos(agd2)**2) + (2 * r * r - r - 2) - 2 * (r * r) * np.cos(agd2)**(-2)))
    B = ((2 * (constants.OMEGA + omg) * rk / ((r+1) * (r+2)))
         * (np.cos(agd2)**r) * ((r*r+2 * r + 2) - ((r + 1) * np.cos(agd2))**2 ))
    C = 0.25 * rk * rk * (np.cos(agd2)**(2 * r)) * ((r + 1) * (np.cos(agd2)**2) - (r+2))
    
    #print(f"A ({A.shape})\n{A}")
    #print(f"B ({B.shape})\n{B}")
    #print(f"C ({C.shape})\n{C}")
    state.delp[:,:,1] = (gh0 + constants.RADIUS * constants.RADIUS
                         * ( A + B * np.cos(r * agd1) + C * np.cos(2 * r * agd1)))
    #print(f"***B*** state.delp[:,:,1] ({state.delp[:,:,1].shape})\n{state.delp[:,:,1]}")
    state.delp[:,:,1] = state.delp[:,:,1] - phis # TODO: Subtract 0?
    #print(f"***C*** state.delp[:,:,1] ({state.delp[:,:,1].shape})\n{state.delp[:,:,1]}")

    # TODO: Check why p1, p2 from grid is different in baroclinic example (pa1, pa2)
    grid = np.transpose(
        np.stack(
            [grid_data._horizontal_data.lon.data, grid_data._horizontal_data.lat.data]
        ),
        [1, 2, 0],
    )
    p1 = grid[:-1, :, :]
    p2 = grid[1:, :, :]
    
    muv = init_utils._find_midpoint_unit_vectors(p1, p2)
    p3 = muv["midpoint"]
    e2 = muv["unit_dir"] 
    ex = muv["exv"] 
    ey = muv["eyv"]
    utmp = (constants.RADIUS * omg * np.cos(p3[:, :, 1]) + constants.RADIUS * rk * (np.cos(p3[:, :, 1])**(r-1)) * (r * np.sin(p3[:, :, 1])**2 - np.cos(p3[:, :, 1])**2)*np.cos(r*p3[:, :, 0]))
    vtmp = -1 * constants.RADIUS * rk * r * np.sin(p3[:, :, 1]) * np.sin(r * p3[:, :, 0]) * np.cos(p3[:, :, 1])**(r-1)
    #print(f"********************muv p3 {p3.shape}: {p3}")
    #print(f"********************muv e2 {e2.shape}: {e2}")
    #print(f"********************muv ex {ex.shape}: {ex}")
    #print(f"********************muv ey {ey.shape}: {ey}")
    #print(f"********************muv utmp {utmp.shape}: {utmp}")
    #print(f"********************muv vtmp {vtmp.shape}: {vtmp}")

    state.u[:-1, :, 0] = utmp * np.sum(e2 * ex, 2) + vtmp * np.sum(e2 * ey, 2)
    

    p1 = grid[:, :-1, :]
    p2 = grid[:, 1:, :]
    muv = init_utils._find_midpoint_unit_vectors(p1, p2)
    p3 = muv["midpoint"]
    e2 = muv["unit_dir"] 
    ex = muv["exv"] 
    ey = muv["eyv"]
    utmp = (constants.RADIUS * omg * np.cos(p3[:, :, 1]) + constants.RADIUS * rk * (np.cos(p3[:, :, 1])**(r-1)) * (r * np.sin(p3[:, :, 1])**2 - np.cos(p3[:, :, 1])**2)*np.cos(r*p3[:, :, 0]))
    vtmp = -1 * constants.RADIUS * rk * r * np.sin(p3[:, :, 1]) * np.sin(r * p3[:, :, 0]) * np.cos(p3[:, :, 1])**(r-1)
    state.v[:, :-1, 0] = utmp * np.sum(e2 * ex, 2) + vtmp * np.sum(e2 * ey, 2)
    
    # TODO: Pay attention to the slice indices. u and v are similarly calculated.

    """ From test_cases.F90 (case 6): 
         do j=js,je
            do i=is,ie+1
               p1(:) = grid(i  ,j ,1:2)
               p2(:) = grid(i,j+1 ,1:2)
               call mid_pt_sphere(p1, p2, p3)
               call get_unit_vect2(p1, p2, e2)
               call get_latlon_vector(p3, ex, ey)
               utmp = radius*omg*cos(p3(2)) +                      &
                      radius*rk*(cos(p3(2))**(R-1))*(R*sin(p3(2))**2-cos(p3(2))**2)*cos(R*p3(1))
               vtmp = -radius*rk*R*sin(p3(2))*sin(R*p3(1))*cos(p3(2))**(R-1)
               v(i,j,1) = utmp*inner_prod(e2,ex) + vtmp*inner_prod(e2,ey)
            enddo
         enddo
         do j=js,je+1
            do i=is,ie
               p1(:) = grid(i,  j,1:2)
               p2(:) = grid(i+1,j,1:2)
               call mid_pt_sphere(p1, p2, p3)
               call get_unit_vect2(p1, p2, e1)
               call get_latlon_vector(p3, ex, ey)
               utmp = radius*omg*cos(p3(2)) +                      &
                      radius*rk*(cos(p3(2))**(R-1))*(R*sin(p3(2))**2-cos(p3(2))**2)*cos(R*p3(1))
               vtmp = -radius*rk*R*sin(p3(2))*sin(R*p3(1))*cos(p3(2))**(R-1)
               u(i,j,1) = utmp*inner_prod(e1,ex) + vtmp*inner_prod(e1,ey)
            enddo
         enddo
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
    
    ''' TODO: Can I ignore these mpp_update_domains calls?
      call mpp_update_domains( delp, domain )
      call mpp_update_domains( phis, domain )
    '''
    
    ''' TODO:
      phi0  = delp

      call init_winds(UBar, u,v,ua,va,uc,vc, initWindsCase, npx, npy, ng, ndims, nregions, gridstruct%bounded_domain, gridstruct, domain, tile, bd)
! Copy 3D data for Shallow Water Tests
    '''

    state.u[:,:,1:] = state.u[:,:,0][:,:,np.newaxis]
    state.v[:,:,1:] = state.v[:,:,0][:,:,np.newaxis]
    
    '''

      do j=js,je
         do i=is,ie
            ps(i,j) = delp(i,j,1)
         enddo
      enddo


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

    return state
