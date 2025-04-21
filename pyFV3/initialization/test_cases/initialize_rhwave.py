""" Test case initialization for Rossby-Haurwitz wave 4

Corresponds to Fortran shallow-water test #6 found in tools/test_cases.F90 of
https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere.git 
"""

import math
import numpy as np

import ndsl.constants as constants
from ndsl import CubedSphereCommunicator, QuantityFactory
from ndsl.grid import GridData
from pyFV3.dycore_state import DycoreState
from pyFV3.initialization import init_utils


NHALO = constants.N_HALO_DEFAULT
OMG = 7.848e-6
RK = 7.848e-6
R = 4.0 # Wave Number (likely)
GH0 = 8.0e3 * constants.GRAV


def preinit_for_all_sw(state: DycoreState,
                       shape,
                       grid_data: GridData
):
    """ Pre-initialization for all shallow water tests

    Args:
        state: DycoreState modified to update pe, pt, delp
        shape: tuple
        grid_data: GridData
    """    
    state.pe[:] = 0.0
    state.pt[:] = 1.0

    # Initialize Halo Corners
    nx, ny, _ = init_utils.local_compute_size(shape)
    state.delp[:NHALO, :NHALO] = 0.0
    state.delp[:NHALO, NHALO + ny :] = 0.0
    state.delp[NHALO + nx :, :NHALO] = 0.0
    state.delp[NHALO + nx :, NHALO + ny :] = 0.0


def init_rhwave_winds(p1, p2):
    """ Initializes u or v winds specific to Rossby-Haurwitz wave test

    Args
        p1 : np.ndarray
        p2 : np.ndarray

    Returns
        np.ndarray representing u or v D-winds
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
    """ Initializes delp specific to Rossby-Haurwitz wave test

    Args
        state DycoreState
        grid_Data GridData

    Returns
        np.ndarray representing delp values
    """
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
    """ Initialization specific to Rossby-Haurwitz wave test

    Args
        state DycoreState, modified to update the 
        grid_Data GridData
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

def init_winds(state: DycoreState,
               grid_data: GridData
):
    # TODO: state.ua, state,va, state.uc, state.vc, d?
    sample_quantity = grid_data.lat
    shape = (*sample_quantity.data.shape[0:2], grid_data.ak.data.shape[0])
    nx, ny, _ = init_utils.local_compute_size(shape)
    '''
    peln=state.peln[slice_3d_buffer]
    qvapor=state.qvapor[slice_3d_buffer]
    delp=state.delp[slice_3d_buffer]
    u=state.u[slice_3d_buffer]
    v=state.v[slice_3d_buffer]
    pt=state.pt[slice_3d_buffer]
    phis=state.phis[slice_2d_buffer]
    delz=state.delz[slice_3d_buffer]
    w=state.w[slice_3d_buffer]
    lon=utils.asarray(grid_data.lon.data[slice_2d_buffer])
    lat=utils.asarray(grid_data.lat.data[slice_2d_buffer])
    lon_agrid=utils.asarray(grid_data.lon_agrid.data[slice_2d_buffer])
    lat_agrid=utils.asarray(grid_data.lat_agrid.data[slice_2d_buffer])
    ee1=utils.asarray(grid_data.ee1.data[slice_3d_buffer])
    ee2=utils.asarray(grid_data.ee2.data[slice_3d_buffer])
    es1=utils.asarray(grid_data.es1.data[slice_3d_buffer])
    ew2=utils.asarray(grid_data.ew2.data[slice_3d_buffer])
    ptop=grid_data.ptop
    '''
    psi = np.zeros([nx, ny])
    psi[:] = 1.e25
    psi_b = np.zeros([nx, ny])
    psi_b[:] = 1.e25
    #  do j=jsd,jed
    #     do i=isd,ied
    #        psi(i,j) = (-1.0 * Ubar * radius *( sin(agrid(i,j,2))                  *cos(alpha) - &
    #                                        cos(agrid(i,j,1))*cos(agrid(i,j,2))*sin(alpha) ) )
    #     enddo
    #enddo
    pass

'''
!     init_winds :: initialize the winds
!
      subroutine init_winds(UBar, u,v,ua,va,uc,vc, defOnGrid, npx, npy, ng, ndims, nregions, bounded_domain, gridstruct, domain, tile, bd)
 ! defOnGrid = -1:null_op, 0:All-Grids, 1:C-Grid, 2:D-Grid, 3:A-Grid, 4:A-Grid then Rotate, 5:D-Grid with unit vectors then Rotate

      type(fv_grid_bounds_type), intent(IN) :: bd
      real  ,    intent(INOUT) :: UBar
      real ,      intent(INOUT) ::    u(bd%isd:bd%ied  ,bd%jsd:bd%jed+1)
      real ,      intent(INOUT) ::    v(bd%isd:bd%ied+1,bd%jsd:bd%jed  )
      real ,      intent(INOUT) ::   uc(bd%isd:bd%ied+1,bd%jsd:bd%jed  )
      real ,      intent(INOUT) ::   vc(bd%isd:bd%ied  ,bd%jsd:bd%jed+1)
      real ,      intent(INOUT) ::   ua(bd%isd:bd%ied  ,bd%jsd:bd%jed  )
      real ,      intent(INOUT) ::   va(bd%isd:bd%ied  ,bd%jsd:bd%jed  )
      integer,      intent(IN) :: defOnGrid
      integer,      intent(IN) :: npx, npy
      integer,      intent(IN) :: ng
      integer,      intent(IN) :: ndims
      integer,      intent(IN) :: nregions
      logical,      intent(IN) :: bounded_domain
      type(fv_grid_type), intent(IN), target :: gridstruct
      type(domain2d), intent(INOUT) :: domain
      integer, intent(IN)  :: tile

      real(kind=R_GRID) :: p1(2), p2(2), p3(2), p4(2), pt(2)
      real(kind=R_GRID) :: e1(3), e2(3), ex(3), ey(3)

      real   :: dist, r, r0
      integer :: i,j,k,n
      real :: utmp, vtmp

      real :: psi_b(bd%isd:bd%ied+1,bd%jsd:bd%jed+1), psi(bd%isd:bd%ied,bd%jsd:bd%jed), psi1, psi2
      integer :: is2, ie2, js2, je2

      real(kind=R_GRID), pointer, dimension(:,:,:)   :: agrid, grid
      real, pointer, dimension(:,:)     :: area, rarea, fC, f0
      real(kind=R_GRID), pointer, dimension(:,:,:)   :: ee1, ee2, en1, en2
      real(kind=R_GRID), pointer, dimension(:,:,:,:) :: ew, es
      real, pointer, dimension(:,:)     :: dx,dy, dxa,dya, rdxa, rdya, dxc,dyc

      logical, pointer :: cubed_sphere, latlon

      logical, pointer :: have_south_pole, have_north_pole

      integer, pointer :: ntiles_g
      real,    pointer :: acapN, acapS, globalarea

      integer :: is, ie, js, je
      integer :: isd, ied, jsd, jed

      grid => gridstruct%grid_64
      agrid=> gridstruct%agrid_64

      area  => gridstruct%area
      rarea => gridstruct%rarea

      fC    => gridstruct%fC
      f0    => gridstruct%f0

      ee1   => gridstruct%ee1
      ee2   => gridstruct%ee2
      ew    => gridstruct%ew
      es    => gridstruct%es
      en1   => gridstruct%en1
      en2   => gridstruct%en2

      dx      => gridstruct%dx
      dy      => gridstruct%dy
      dxa     => gridstruct%dxa
      dya     => gridstruct%dya
      rdxa    => gridstruct%rdxa
      rdya    => gridstruct%rdya
      dxc     => gridstruct%dxc
      dyc     => gridstruct%dyc

      cubed_sphere => gridstruct%cubed_sphere
      latlon       => gridstruct%latlon

      have_south_pole               => gridstruct%have_south_pole
      have_north_pole               => gridstruct%have_north_pole

      ntiles_g                      => gridstruct%ntiles_g
      acapN                         => gridstruct%acapN
      acapS                         => gridstruct%acapS
      globalarea                    => gridstruct%globalarea

      is  = bd%is
      ie  = bd%ie
      js  = bd%js
      je  = bd%je
      isd = bd%isd
      ied = bd%ied
      jsd = bd%jsd
      jed = bd%jed

      if (bounded_domain) then

         is2 = is-2
         ie2 = ie+2
         js2 = js-2
         je2 = je+2

      else

         is2 = is
         ie2 = ie
         js2 = js
         je2 = je

      end if

 200  format(i4.4,'x',i4.4,'x',i4.4,' ',e21.14,' ',e21.14,' ',e21.14,' ',e21.14,' ',e21.14,' ',e21.14,' ',e21.14,' ',e21.14)

      psi(:,:) = 1.e25
      psi_b(:,:) = 1.e25
      do j=jsd,jed
         do i=isd,ied
            psi(i,j) = (-1.0 * Ubar * radius *( sin(agrid(i,j,2))                  *cos(alpha) - &
                                            cos(agrid(i,j,1))*cos(agrid(i,j,2))*sin(alpha) ) )
         enddo
      enddo
      call mpp_update_domains( psi, domain )
      do j=jsd,jed+1
         do i=isd,ied+1
            psi_b(i,j) = (-1.0 * Ubar * radius *( sin(grid(i,j,2))                 *cos(alpha) - &
                                              cos(grid(i,j,1))*cos(grid(i,j,2))*sin(alpha) ) )
         enddo
      enddo

      if ( (cubed_sphere) .and. (defOnGrid==0) ) then
         do j=js,je+1
            do i=is,ie
               dist = dx(i,j)
               vc(i,j) = (psi_b(i+1,j)-psi_b(i,j))/dist
               if (dist==0) vc(i,j) = 0.
            enddo
         enddo
         do j=js,je
            do i=is,ie+1
               dist = dy(i,j)
               uc(i,j) = -1.0*(psi_b(i,j+1)-psi_b(i,j))/dist
               if (dist==0) uc(i,j) = 0.
            enddo
         enddo
         call mpp_update_domains( uc, vc, domain, gridtype=CGRID_NE_PARAM)
         call fill_corners(uc, vc, npx, npy, VECTOR=.true., CGRID=.true.)
         do j=js,je
            do i=is,ie+1
               dist = dxc(i,j)
               v(i,j) = (psi(i,j)-psi(i-1,j))/dist
               if (dist==0) v(i,j) = 0.
            enddo
         enddo
         do j=js,je+1
            do i=is,ie
               dist = dyc(i,j)
               u(i,j) = -1.0*(psi(i,j)-psi(i,j-1))/dist
               if (dist==0) u(i,j) = 0.
            enddo
         enddo
         call mp_update_dwinds(u, v, npx, npy, domain, bd)
         do j=js,je
            do i=is,ie
               psi1 = 0.5*(psi(i,j)+psi(i,j-1))
               psi2 = 0.5*(psi(i,j)+psi(i,j+1))
               dist = dya(i,j)
               ua(i,j) = -1.0 * (psi2 - psi1) / (dist)
               if (dist==0) ua(i,j) = 0.
               psi1 = 0.5*(psi(i,j)+psi(i-1,j))
               psi2 = 0.5*(psi(i,j)+psi(i+1,j))
               dist = dxa(i,j)
               va(i,j) = (psi2 - psi1) / (dist)
               if (dist==0) va(i,j) = 0.
            enddo
         enddo

      elseif ( (cubed_sphere) .and. (defOnGrid==1) ) then
         do j=js,je+1
            do i=is,ie
               dist = dx(i,j)
               vc(i,j) = (psi_b(i+1,j)-psi_b(i,j))/dist
               if (dist==0) vc(i,j) = 0.
            enddo
         enddo
         do j=js,je
            do i=is,ie+1
               dist = dy(i,j)
               uc(i,j) = -1.0*(psi_b(i,j+1)-psi_b(i,j))/dist
               if (dist==0) uc(i,j) = 0.
            enddo
         enddo
         call mpp_update_domains( uc, vc, domain, gridtype=CGRID_NE_PARAM)
         call fill_corners(uc, vc, npx, npy, VECTOR=.true., CGRID=.true.)
         call ctoa(uc,vc,ua,va,dx, dy, dxc,dyc,dxa,dya,npx,npy,ng, bd)
         call atod(ua,va,u ,v ,dxa, dya,dxc,dyc,npx,npy,ng, bounded_domain, domain, bd)
        ! call d2a2c(npx,npy,1, is,ie, js,je, ng, u(isd,jsd),v(isd,jsd), &
        !            ua(isd,jsd),va(isd,jsd), uc(isd,jsd),vc(isd,jsd))
      elseif ( (cubed_sphere) .and. (defOnGrid==2) ) then
         do j=js2,je2
            do i=is2,ie2+1
               dist = dxc(i,j)
               v(i,j) = (psi(i,j)-psi(i-1,j))/dist
               if (dist==0) v(i,j) = 0.
            enddo
         enddo
         do j=js2,je2+1
            do i=is2,ie2
               dist = dyc(i,j)
               u(i,j) = -1.0*(psi(i,j)-psi(i,j-1))/dist
               if (dist==0) u(i,j) = 0.
            enddo
         enddo
         call mp_update_dwinds(u, v, npx, npy, domain, bd)
         call dtoa( u, v,ua,va,dx,dy,dxa,dya,dxc,dyc,npx,npy,ng, bd)
         call atoc(ua,va,uc,vc,dx,dy,dxa,dya,npx,npy,ng, bounded_domain, domain, bd)
      elseif ( (cubed_sphere) .and. (defOnGrid==3) ) then
         do j=js,je
            do i=is,ie
               psi1 = 0.5*(psi(i,j)+psi(i,j-1))
               psi2 = 0.5*(psi(i,j)+psi(i,j+1))
               dist = dya(i,j)
               ua(i,j) = -1.0 * (psi2 - psi1) / (dist)
               if (dist==0) ua(i,j) = 0.
               psi1 = 0.5*(psi(i,j)+psi(i-1,j))
               psi2 = 0.5*(psi(i,j)+psi(i+1,j))
               dist = dxa(i,j)
               va(i,j) = (psi2 - psi1) / (dist)
               if (dist==0) va(i,j) = 0.
            enddo
         enddo
         call mpp_update_domains( ua, va, domain, gridtype=AGRID_PARAM)
         call atod(ua,va, u, v,dxa, dya,dxc,dyc,npx,npy,ng, bounded_domain, domain, bd)
         call atoc(ua,va,uc,vc,dx,dy,dxa,dya,npx,npy,ng, bounded_domain,domain, bd)
      elseif ( (latlon) .or. (defOnGrid==4) ) then

         do j=js,je
            do i=is,ie
               ua(i,j) =  Ubar * ( COS(agrid(i,j,2))*COS(alpha) + &
                                     SIN(agrid(i,j,2))*COS(agrid(i,j,1))*SIN(alpha) )
               va(i,j) = -Ubar *   SIN(agrid(i,j,1))*SIN(alpha)
               call mid_pt_sphere(grid(i,j,1:2), grid(i,j+1,1:2), p1)
               call mid_pt_sphere(grid(i,j,1:2), grid(i+1,j,1:2), p2)
               call mid_pt_sphere(grid(i+1,j,1:2), grid(i+1,j+1,1:2), p3)
               call mid_pt_sphere(grid(i,j+1,1:2), grid(i+1,j+1,1:2), p4)
               if (cubed_sphere) call rotate_winds(ua(i,j), va(i,j), p1,p2,p3,p4, agrid(i,j,1:2), 2, 1)

               psi1 = 0.5*(psi(i,j)+psi(i,j-1))
               psi2 = 0.5*(psi(i,j)+psi(i,j+1))
               dist = dya(i,j)
    if ( (tile==1) .and.(i==1) ) print*, ua(i,j), -1.0 * (psi2 - psi1) / (dist)

            enddo
         enddo
         call mpp_update_domains( ua, va, domain, gridtype=AGRID_PARAM)
         call atod(ua,va, u, v,dxa, dya,dxc,dyc,npx,npy,ng, bounded_domain, domain, bd)
         call atoc(ua,va,uc,vc,dx,dy,dxa,dya,npx,npy,ng, bounded_domain, domain, bd)
     elseif ( (latlon) .or. (defOnGrid==5) ) then
! SJL mods:
! v-wind:
         do j=js2,je2
            do i=is2,ie2+1
               p1(:) = grid(i  ,j ,1:2)
               p2(:) = grid(i,j+1 ,1:2)
               call mid_pt_sphere(p1, p2, pt)
               call get_unit_vect2 (p1, p2, e2)
               call get_latlon_vector(pt, ex, ey)
               utmp =  Ubar * ( COS(pt(2))*COS(alpha) + &
                                SIN(pt(2))*COS(pt(1))*SIN(alpha) )
               vtmp = -Ubar *   SIN(pt(1))*SIN(alpha)
               v(i,j) = utmp*inner_prod(e2,ex) + vtmp*inner_prod(e2,ey)
            enddo
         enddo
! D grid u-wind:
         do j=js2,je2+1
            do i=is2,ie2
               p1(:) = grid(i  ,j  ,1:2)
               p2(:) = grid(i+1,j  ,1:2)
               call mid_pt_sphere(p1, p2, pt)
               call get_unit_vect2 (p1, p2, e1)
               call get_latlon_vector(pt, ex, ey)
               utmp =  Ubar * ( COS(pt(2))*COS(alpha) + &
                                SIN(pt(2))*COS(pt(1))*SIN(alpha) )
               vtmp = -Ubar *   SIN(pt(1))*SIN(alpha)
               u(i,j) = utmp*inner_prod(e1,ex) + vtmp*inner_prod(e1,ey)
            enddo
         enddo

         call mp_update_dwinds(u, v, npx, npy, domain, bd)
         call dtoa( u, v,ua,va,dx,dy,dxa,dya,dxc,dyc,npx,npy,ng, bd)
         call atoc(ua,va,uc,vc,dx,dy,dxa,dya,npx,npy,ng, bounded_domain, domain, bd)
     else
         !print*, 'Choose an appropriate grid to define the winds on'
         !stop
     endif

      end subroutine init_winds
'''


def postinit_for_all_sw(state, grid_data):
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

    init_winds(state, grid_data)
        
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
    postinit_for_all_sw(numpy_state, grid_data)

    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    comm.halo_update(state.phis, n_points=NHALO)
    comm.vector_halo_update(state.u, state.v, n_points=NHALO)
    # TODO: anymore comm updates? delp?

    return state
