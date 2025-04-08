import math

import numpy as np

import ndsl.constants as constants
import ndsl.dsl.gt4py_utils as utils
from ndsl import CubedSphereCommunicator, QuantityFactory
from ndsl.grid import GridData
from ndsl.grid.gnomonic import great_circle_distance_lon_lat, lon_lat_midpoint
from pyFV3.dycore_state import DycoreState
from pyFV3.initialization import init_utils

def _rhwave_initialization():
    """
    TODO What does this do?

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
    phis = 0.0 # TODO: Why is this phis scalar, while the DycoreState version is 2d?
    '''
    # JKNOTE: This looks like it's iterating over a 2D grid to determine the A, B, C values?
    # JKNOTE: delp is pressure_thickness_atmospheric_layer
    # JKNOTE: This looks like what Oliver was talking about -- where the z is 1 in the delp because it's a 2d problem for shallow water?
    # A, B, C are scalars
    # Also -- looking at this makes me wonder if I should be thinking of using gt4py stencil somehow? I need to do a tutorial...
         do j=js,je # TODO: Where does js and je come from? comes from bd in the fortran code.
            do i=is,ie
               A = 0.5*omg*(2.*costants.OMEGA+omg)*(COS(agrid(i,j,2))**2) + &
                   0.25*rk*rk*(COS(agrid(i,j,2))**(r+r)) * &
                   ( (r+1)*(COS(agrid(i,j,2))**2) + (2.*r*r-r-2.) - &
                     2.*(r*r)*COS(agrid(i,j,2))**(-2.) )
               B = (2.*(constants.OMEGA+omg)*rk / ((r+1)*(r+2))) * (COS(agrid(i,j,2))**r) * &
                    ( (r*r+2.*r+2.) - ((r+1.)*COS(agrid(i,j,2)))**2 )
               C = 0.25*rk*rk*(COS(agrid(i,j,2))**(2.*r)) * ( &
                   (r+1) * (COS(agrid(i,j,2))**2.) - (r+2.) )
               delp(i,j,1) =gh0 + radius*radius*(A+B*COS(r*agrid(i,j,1))+C*COS(2.*r*agrid(i,j,1)))
               delp(i,j,1) = delp(i,j,1) - phis(i,j)
            enddo
         enddo
    '''
    """ From test_cases.F90:

      case(6)
         #Ubar = 0.
         gh0  = 8.E3*Grav
         R    = 4.
         omg  = 7.848E-6
         rk    = 7.848E-6
         phis = 0.0
         do j=js,je
            do i=is,ie
               A = 0.5*omg*(2.*omega+omg)*(COS(agrid(i,j,2))**2) + &
                   0.25*rk*rk*(COS(agrid(i,j,2))**(r+r)) * &
                   ( (r+1)*(COS(agrid(i,j,2))**2) + (2.*r*r-r-2.) - &
                     2.*(r*r)*COS(agrid(i,j,2))**(-2.) )
               B = (2.*(omega+omg)*rk / ((r+1)*(r+2))) * (COS(agrid(i,j,2))**r) * &
                    ( (r*r+2.*r+2.) - ((r+1.)*COS(agrid(i,j,2)))**2 )
               C = 0.25*rk*rk*(COS(agrid(i,j,2))**(2.*r)) * ( &
                   (r+1) * (COS(agrid(i,j,2))**2.) - (r+2.) )
               delp(i,j,1) =gh0 + radius*radius*(A+B*COS(r*agrid(i,j,1))+C*COS(2.*r*agrid(i,j,1)))
               delp(i,j,1) = delp(i,j,1) - phis(i,j)
            enddo
         enddo
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
    pass


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
    # TODO: What exactly is this? Found in both tc and baroclinic inits.
    sample_quantity = grid_data.lat
    shape = (*sample_quantity.data.shape[0:2], grid_data.ak.data.shape[0])
    nx, ny, nz = init_utils.local_compute_size(shape)
    numpy_state = init_utils.empty_numpy_dycore_state(shape)

    # Initializing to values the Fortran does for easy comparison
    numpy_state.pe[:] = 0.0
    numpy_state.pt[:] = 1.0
    
    NHALO = constants.N_HALO_DEFAULT # TODO: Where to put this?
    numpy_state.delp[:NHALO, :NHALO] = 0.0
    numpy_state.delp[:NHALO, NHALO + ny :] = 0.0
    numpy_state.delp[NHALO + nx :, :NHALO] = 0.0
    numpy_state.delp[NHALO + nx :, NHALO + ny :] = 0.0    
    # TODO: more numpy_state.* initialization?
    
    _rhwave_initialization(
    )

    # TODO: Actual DycoreState init
    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    # TODO Halo Update?
    #comm.halo_update(state.phis, n_points=NHALO)
    #comm.vector_halo_update(state.u, state.v, n_points=NHALO)

    return state
