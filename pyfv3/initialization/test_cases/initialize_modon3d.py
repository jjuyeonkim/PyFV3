import numpy as np

import ndsl.constants as constants
import ndsl.dsl.gt4py_utils as utils
from ndsl import CubedSphereCommunicator, QuantityFactory
from ndsl.dsl.typing import Float
from ndsl.grid import GridData
from ndsl.grid.gnomonic import great_circle_distance_lon_lat, lon_lat_midpoint
from pyfv3.dycore_state import DycoreState
from pyfv3.initialization import init_utils

# TODO: Why isn't this a class with things like NHALO passed around as member variables and _init_background_state as member functions? 

NHALO = constants.N_HALO_DEFAULT

def _init_background_state(numpy_state):
    """
    Args:
        numpy_state: DycoreState modified to initialize ps, phis, u, v, q
    """
    #ps(:,:) = p00
    p00 = Float(1000e2)
    numpy_state.ps[:] = p00
    numpy_state.phis[:] = Float(0.0)
    numpy_state.u[:] = Float(0.0)
    numpy_state.v[:] = Float(0.0)
    numpy_state.q[:] = Float(0.0)


def _init_delta_p():
   pass

def _init_westerly_wind_burst():
   pass

def _add_easterly_wind_burst():
   pass

def _convert_back_to_temperature():
   pass

def init_state(
    grid_data: GridData,
    quantity_factory: QuantityFactory,
    comm: CubedSphereCommunicator,
) -> DycoreState:
    """
    Create a DycoreState object with quantities initialized for the
    3D Modon Soliton test case applied to the cubed sphere grid.
    """
    sample_quantity = grid_data.lat
    shape = (*sample_quantity.data.shape[0:2], grid_data.ak.data.shape[0])
    numpy_state = init_utils.empty_numpy_dycore_state(shape)

    _init_background_state(numpy_state)
    _init_delta_p()
    _init_westerly_wind_burst()
    _add_easterly_wind_burst()
    _convert_back_to_temperature()
    # Nest Test?
    # Delz, w calculation for non-hydrostatic?
    
    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    comm.halo_update(state.phis, n_points=NHALO)

    comm.vector_halo_update(state.u, state.v, n_points=NHALO)


    return state

'''
      else if (test_case == 45 .or. test_case == 46) then    ! NGGPS test?

! Background state
         f0 = 0.;  fC = 0.
         pt0 = 300.   ! potentil temperature
         p00 = 1000.e2
         ps(:,:) = p00
         phis = 0.0
         u(:,:,:) = 0.
         v(:,:,:) = 0.
         q(:,:,:,:) = 0.

         if (adiabatic) then
             zvir = 0.
         else
             zvir = rvgas/rdgas - 1.
         endif

! Initialize delta-P
        do k=1,npz
            do j=js,je
               do i=is,ie
                  delp(i,j,k) = ak(k+1)-ak(k) + ps(i,j)*(bk(k+1)-bk(k))
               enddo
            enddo
         enddo

         do j=js,je
            do i=is,ie
               pe(i,1,j) = ptop
               peln(i,1,j) = log(pe(i,1,j))
                 pk(i,j,1) = exp(kappa*peln(i,1,j))
            enddo
            do k=2,npz+1
            do i=is,ie
                 pe(i,k,j) = pe(i,k-1,j) + delp(i,j,k-1)
               peln(i,k,j) = log(pe(i,k,j))
                 pk(i,j,k) = exp(kappa*peln(i,k,j))
            enddo
            enddo
         enddo

! Initiate the westerly-wind-burst:
         ubar = soliton_Umax
         r0 = soliton_size
         p0w(1) = pi*0.5
         p0w(2) = 0.
         p0e(1) = p0w(1) + pi
         p0e(2) = 0.


     do k=1,npz
        do j=js,je
           do i=is,ie+1
              p1(:) = grid(i  ,j ,1:2)
              p2(:) = grid(i,j+1 ,1:2)
              call mid_pt_sphere(p1, p2, p3)
              r = great_circle_dist( p0w, p3, radius )
              utmp = ubar*exp(-(r/r0)**2)
              call get_unit_vect2(p1, p2, e2)
              call get_latlon_vector(p3, ex, ey)
              v(i,j,k) = utmp*inner_prod(e2,ex)
           enddo
        enddo
        do j=js,je+1
           do i=is,ie
              p1(:) = grid(i,  j,1:2)
              p2(:) = grid(i+1,j,1:2)
              call mid_pt_sphere(p1, p2, p3)
              r = great_circle_dist( p0w, p3, radius )
              utmp = ubar*exp(-(r/r0)**2)
              call get_unit_vect2(p1, p2, e1)
              call get_latlon_vector(p3, ex, ey)
              u(i,j,k) = utmp*inner_prod(e1,ex)
           enddo
        enddo

! Add easterly-wind-brust:
        if (nsolitons > 0) then
        p0(1) = p0(1) + pi
        p0(2) = 0.

        do j=js,je
           do i=is,ie+1
              p1(:) = grid(i  ,j ,1:2)
              p2(:) = grid(i,j+1 ,1:2)
              call mid_pt_sphere(p1, p2, p3)
              r = great_circle_dist( p0e, p3, radius )
              utmp = ubar*exp(-(r/r0)**2)
              call get_unit_vect2(p1, p2, e2)
              call get_latlon_vector(p3, ex, ey)
              v(i,j,k) = v(i,j,k) - utmp*inner_prod(e2,ex)
           enddo
        enddo
        do j=js,je+1
           do i=is,ie
              p1(:) = grid(i,  j,1:2)
              p2(:) = grid(i+1,j,1:2)
              call mid_pt_sphere(p1, p2, p3)
              r = great_circle_dist( p0e, p3, radius )
              utmp = ubar*exp(-(r/r0)**2)
              call get_unit_vect2(p1, p2, e1)
              call get_latlon_vector(p3, ex, ey)
              u(i,j,k) = u(i,j,k) - utmp*inner_prod(e1,ex)
           enddo
        enddo
        endif !nsolitons > 0

        do j=js,je
           do i=is,ie
              pkz(i,j,k) = (pk(i,j,k+1)-pk(i,j,k))/(kappa*(peln(i,k+1,j)-peln(i,k,j)))

#ifdef USE_PT
              pt(i,j,k) = pt0/p00**kappa
! Convert back to temperature:
              pt(i,j,k) = pt(i,j,k)*pkz(i,j,k)
#else
              pt(i,j,k) = pt0
#endif
              q(i,j,k,1) = 0.
           enddo
        enddo

     enddo

#ifdef NEST_TEST
     do k=1,npz
     do j=js,je
     do i=is,ie
        q(i,j,k,:) = agrid(i,j,1)*0.180/pi
     enddo
     enddo
     enddo
#else
!     call checker_tracers(is,ie, js,je, isd,ied, jsd,jed,  &
!                          ncnst, npz, q, agrid(is:ie,js:je,1), agrid(is:ie,js:je,2), 9., 9.)
#endif

        if ( .not. hydrostatic ) then
            do k=1,npz
               do j=js,je
                  do i=is,ie
                     delz(i,j,k) = rdgas*pt(i,j,k)/grav*log(pe(i,k,j)/pe(i,k+1,j))
                        w(i,j,k) = 0.0
                  enddo
               enddo
            enddo
         endif
      else if (test_case == 55 .or. test_case == 56 .or. test_case == 57 .or. test_case == 58) then
'''