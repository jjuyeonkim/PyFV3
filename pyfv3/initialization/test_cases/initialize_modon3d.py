import numpy as np

import ndsl.constants as constants
import ndsl.dsl.gt4py_utils as utils
from ndsl import CubedSphereCommunicator, QuantityFactory
from ndsl.dsl.typing import Float
from ndsl.grid import GridData
from ndsl.grid.gnomonic import great_circle_distance_lon_lat
from pyfv3.dycore_state import DycoreState
from pyfv3.initialization import init_utils


# TODO: Why isn't this a class with things like NHALO passed around as member
# variables and _init_background_state as member functions?

PT0 = Float(300.0)  # Potential temperature
P00 = Float(1.0e5)  # TODO: Same as baroclinic... how to consolidate?
NHALO = constants.N_HALO_DEFAULT


def _init_modon_pressure_fields(
    eta,  # TODO: Do I need?
    eta_v,  # TODO: Do I need?
    delp,
    ps,
    pe,
    peln,
    pk,
    pkz,  # TODO: Do I need?
    ak,
    bk,
    ptop,
):
    # TODO: copying from baroclinic --- double-check indices for modon.
    pe[:] = Float(0.0)
    pk[:] = Float(1.0)

    ps[:] = P00  # TODO: This is set above, do I need this?
    delp[:, :, :-1] = init_utils.initialize_delp(ps, ak, bk)
    pe[:] = init_utils.initialize_edge_pressure(delp, ptop)
    peln[:] = init_utils.initialize_log_pressure_interfaces(pe, ptop)

    # NOTE: The modon calculation for pk looks different than baroclinic
    # (init_utils.initialize_kappa_pressures).
    pk[:] = np.zeros(pe.shape)
    pk[:, :, 0] = np.exp(constants.KAPPA * peln[:, :, 0])
    pk[:, :, 1:] = np.exp(constants.KAPPA * peln[:, :, 1:])
    # TODO pz may not be needed?
    # eta[:-1], eta_v[:-1] = init_utils.compute_eta(ak, bk) # TODO: Do I need this?


def _init_modon3d_u_v_wind(
    grid_data: GridData,
    u,  # TODO: type
    v,  # TODO: type
    lon,
    lat,
    nx,  # TODO: type
    ny,  # TODO: type
    nz,  # TODO: type
    p0,  # TODO: type
    is_westerly: bool = True,
):
    """
    TODO desc
    Args:
    """
    # sample_quantity = grid_data.lat
    # shape = (*sample_quantity.data.shape[0:2], grid_data.ak.data.shape[0])
    # nx, ny, nz = init_utils.local_compute_size(shape)

    soliton_umax = Float(50.0)  # TODO: Add to config?
    soliton_size = Float(750.0e3)  # TODO: Add to config?

    ubar = soliton_umax
    r0 = soliton_size

    grid = np.transpose(
        np.stack(  # TODO: Refactor to non-protected _horizontal_data
            # TODO: Is it okay just to use the field data for this part?
            [grid_data._horizontal_data.lon.field, grid_data._horizontal_data.lat.field]
        ),
        [1, 2, 0],
    )

    # V winds
    p1 = grid[:, :-1, :]
    p2 = grid[:, 1:, :]
    muv = init_utils._find_midpoint_unit_vectors(
        p1, p2
    )  # TODO: Refactor to non-protected call
    p3 = muv["midpoint"]
    e2 = muv["unit_dir"]
    ex = muv["exv"]
    ey = muv["eyv"]

    # TODO: Is this great circle distance correct?
    r = great_circle_distance_lon_lat(p3[0], p0[0], p3[1], p0[1], constants.RADIUS, np)[
        :, :, None
    ]
    r3d = np.repeat(r, v.shape[2], axis=2)

    utmp = ubar * np.exp(-((r3d / r0) ** 2))
    # for k in range(0, v.shape[2]): # TODO: iterate over k better than this.
    k = 0
    if is_westerly:
        v[:, :-1, k] = utmp * np.sum(e2 * ex, 2)  # TODO: double-check innerprod?
    else:
        v[:, :-1, k] -= utmp * np.sum(e2 * ex, 2)  # TODO: double-check innerprod?

    # U winds
    p1 = grid[:-1, :, :]
    p2 = grid[1:, :, :]
    muv = init_utils._find_midpoint_unit_vectors(
        p1, p2
    )  # TODO: Refactor to non-protected call

    p3 = muv["midpoint"]
    e2 = muv["unit_dir"]
    ex = muv["exv"]
    ey = muv["eyv"]

    # r = 1 # TODO: Get the actual great circle distance? which one?
    # TODO: Is this great circle distance correct?
    r = great_circle_distance_lon_lat(p3[0], p0[0], p3[1], p0[1], constants.RADIUS, np)[
        :, :, None
    ]
    r3d = np.repeat(r, u.shape[2], axis=2)

    utmp = ubar * np.exp(-((r3d / r0) ** 2))
    # for k in range(0, v.shape[2]): # TODO: iterate over k better than this.
    k = 0
    if is_westerly:
        u[:-1, :, k] = utmp * np.sum(e2 * ex, 2)  # TODO: double-check innerprod?
    else:
        u[:-1, :, k] -= utmp * np.sum(e2 * ex, 2)  # TODO: double-check innerprod?


def _init_modon3d(
    grid_data: GridData,
    u,  # TODO: type
    v,  # TODO: type
    lon,  # TODO: type
    lat,  # TODO: type
    nx,  # TODO: type
    ny,  # TODO: type
    nz,  # TODO: type
    nsolitons: int = 2,  # TODO: add to dycore config?
):
    p0w = (Float(constants.PI * 0.5), Float(0.0))
    p0e = (p0w[0] + constants.PI, Float(0.0))

    # westerly
    _init_modon3d_u_v_wind(
        grid_data, u, v, lon, lat, nx, ny, nz, p0=p0w, is_westerly=True
    )

    # easterly
    if nsolitons > 0:
        # p0(1) = p0(1) + pi # TODO: Not used??
        # p0(2) = 0. # TODO: Not used??
        _init_modon3d_u_v_wind(
            grid_data, v, u, lon, lat, nx, ny, nz, p0=p0e, is_westerly=False
        )


def _convert_back_to_temperature(
    peln,  # TODO: type
    pk,  # TODO: type
    pkz,  # TODO: type
    pt,  # TODO: type
    use_pt=False,  # TODO: type
):
    pkz[:, :, :-1] = (
        pk[:, :, 1:] - pk[:, :, :-1]
    ) / (
        constants.KAPPA * (peln[:, :, 1:] - peln[:, :, :-1])
    )  # TODO: Again, what about the kth? pkz?
    if use_pt:
        pt[:] = PT0 / P00 ** constants.KAPPA
        pt[:] *= pkz[:]
    else:
        pt[:] = PT0
    # TODO: Which Tracers do I set to 0? q(i,j,k,1) = 0.

def _init_non_hydrostatic(
    pt,  # TODO: type
    pe,  # TODO: type
    delz,  # TODO: type
    w,  # TODO: type
):
    delz[:, :, :-1] = (
        constants.RDGAS * pt[:, :, :-1] / constants.GRAV * np.log(pe[:, :, :-1] / pe[:, :, 1:])
    )  # TODO: Does this pe slice work? What about the kth delz?
    w[:] = Float(0.0)


def init_state(
    grid_data: GridData,
    quantity_factory: QuantityFactory,
    comm: CubedSphereCommunicator,
    hydrostatic: bool,
    nsolitons: int = 2,  # TODO: add to config?
) -> DycoreState:
    """
    Create a DycoreState object with quantities initialized for the
    3D Modon Soliton test case applied to the cubed sphere grid.
    """
    sample_quantity = grid_data.lat
    shape = (*sample_quantity.data.shape[0:2], grid_data.ak.data.shape[0])
    numpy_state = init_utils.empty_numpy_dycore_state(shape)

    # Background init for ps, phis, u, v, tracers
    numpy_state.ps[:] = P00
    numpy_state.phis[:] = Float(0.0)
    numpy_state.u[:] = Float(0.0)
    numpy_state.v[:] = Float(0.0)

    # TODO: How do I handle tracers again?
    numpy_state.qvapor[:] = Float(0.0)

    nx, ny, nz = init_utils.local_compute_size(shape)

    # TODO: copying from baroclinic --- double-check for modon.
    numpy_state.delp[:] = Float(1e30)
    numpy_state.delp[:NHALO, :NHALO] = Float(0.0)
    numpy_state.delp[:NHALO, NHALO + ny :] = Float(0.0)
    numpy_state.delp[NHALO + nx :, :NHALO] = Float(0.0)
    numpy_state.delp[NHALO + nx :, NHALO + ny :] = Float(0.0)

    eta = np.zeros(nz)
    eta_v = np.zeros(nz)
    islice, jslice, slice_3d, slice_2d = init_utils.compute_slices(nx, ny)
    # Slices with extra buffer points in the horizontal dimension
    # to accomodate averaging over shifted calculations on the grid
    _, _, slice_3d_buffer, slice_2d_buffer = init_utils.compute_slices(nx + 1, ny + 1)

    _init_modon_pressure_fields(
        eta=eta,
        eta_v=eta_v,
        delp=numpy_state.delp[slice_3d],
        ps=numpy_state.ps[slice_2d],
        pe=numpy_state.pe[slice_3d],
        peln=numpy_state.peln[slice_3d],
        pk=numpy_state.pk[slice_3d],
        pkz=numpy_state.pkz[slice_3d],
        ak=utils.asarray(grid_data.ak.data),
        bk=utils.asarray(grid_data.bk.data),
        ptop=grid_data.ptop,
    )

    # _init_modon3d(
    #     grid_data,
    #     u=numpy_state.u[slice_3d_buffer],
    #     v=numpy_state.v[slice_3d_buffer],
    #     lon=utils.asarray(grid_data.lon.data[slice_2d_buffer]),
    #     lat=utils.asarray(grid_data.lat.data[slice_2d_buffer]),
    #     nx=nx,
    #     ny=ny,
    #     nz=nz,
    #     nsolitons=nsolitons,
    # )

    _convert_back_to_temperature(
        peln=numpy_state.peln[slice_3d],
        pk=numpy_state.pk[slice_3d],
        pkz=numpy_state.pkz[slice_3d],
        pt=numpy_state.pt[slice_3d],
        use_pt=False, # TODO: What is USE_PT?
    )

    # TODO: Can I ignore the NEST_TEST?

    if not hydrostatic:
        _init_non_hydrostatic(
            pt=numpy_state.pt[slice_3d],
            pe=numpy_state.pe[slice_3d],
            delz=numpy_state.delz[slice_3d],
            w=numpy_state.w[slice_3d],
        )

    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    comm.halo_update(state.phis, n_points=NHALO)

    comm.vector_halo_update(state.u, state.v, n_points=NHALO)

    return state


"""
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
"""
