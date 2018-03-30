import numpy as np
from argparse import ArgumentParser

from devito.logger import info
from devito import Constant
from examples.seismic.elastic import ElasticWaveSolver
from examples.seismic import ModelElastic, RickerSource, Receiver


def elastic_setup(shape=(50, 50), spacing=(15.0, 15.0),
                   tn=500., space_order=4, nbpml=10,
                   **kwargs):

    shape = (1601, 401)
    spacing = (7.5, 7.5)
    origin = (0., 0.)

    vp = np.fromfile("/nethome/mlouboutin3/Research/datasets/devito_data/Simple2D/vp_marmousi_bi", dtype=np.float32, sep="")
    vp = np.reshape(vp, shape)
    # Cut the model to make it slightly cheaper
    vp = vp[301:-300, :]
    vs = vp/2
    rho = vp - np.min(vp) + 1.0
    shape = vp.shape
    nrec = shape[0]
    model = ModelElastic(origin, spacing, shape, space_order, vp, vs, rho)

    # Derive timestepping from model spacing
    dt = model.critical_dt
    t0 = 0.0
    nt = int(1 + (tn-t0) / dt)  # Number of timesteps
    time = np.linspace(t0, tn, nt)  # Discretized time axis

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time=time)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, just below surface)
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = ElasticWaveSolver(model, source=src, receiver=rec,
                                space_order=space_order, **kwargs)
    return solver


def run(shape=(50, 50), spacing=(20.0, 20.0), tn=1000.0,
        space_order=4, nbpml=40, full_run=False,
        autotune=False, **kwargs):

    solver = elastic_setup(shape=shape, spacing=spacing, nbpml=nbpml, tn=tn,
                            space_order=space_order, **kwargs)
    info("Applying Forward")
    # Define receiver geometry (spread across x, just below surface)
    rec1, rec2, vx, vz, txx, tzz, txz, summary = solver.forward()
    import matplotlib.pyplot as plt
    from IPython import embed;embed()

if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument('--2d', dest='dim2', default=False, action='store_true',
                        help="Preset to determine the physical problem setup")
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help="Execute all operators and store forward wavefield")
    parser.add_argument('-a', '--autotune', default=False, action='store_true',
                        help="Enable autotuning for block sizes")
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-dse", "-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DSE) mode")
    args = parser.parse_args()

    # 3D preset parameters
    shape = (150, 150)
    spacing = (15.0, 15.0)
    tn = 2000.0

    run(shape=shape, spacing=spacing, nbpml=args.nbpml, tn=tn,
        space_order=args.space_order,
        autotune=args.autotune, dse=args.dse, dle=args.dle, full_run=args.full)
