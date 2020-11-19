#!/usr/bin/env python3

from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime
from functools import partial
from itertools import product
from json import dump
from logging import getLogger, INFO, DEBUG, basicConfig
from multiprocessing import Pool
from sys import version_info
from pathlib import Path
import warnings

from numpy import (
    array,
    c_,
    ndindex,
    genfromtxt,
    isfinite,
    isnan,
    NaN,
    nansum,
    power,
    sum,
    zeros,
    )
from numpy.linalg import norm, inv
from scipy.stats import norm as normdistr
from nibabel.affines import apply_affine
from nibabel import load as nload

if version_info < (3, 6):
    raise ImportError('Python 3.6 or later is required')


VERSION = 1

lg = getLogger('fmri_at_elec')

# 1 sigma = 0.6065306597126334


def calc_fmri_at_elec(
        measure_nii, electrodes_file, output_dir, graymatter=None, metric='gaussian',
        distances=None, parallel=False):
    """
    Calculate the (weighted) average of fMRI values at electrode locations
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'values_{name(measure_nii)}_{name(electrodes_file)}_{metric}.tsv'
    if output_file.exists():
        raise FileExistsError(f'The output file exists already: {output_file}')

    current_time = datetime.now().isoformat(timespec='seconds')
    img = nload(str(measure_nii))
    mri = img.get_fdata()
    mri[mri == 0] = NaN

    labels, elec_pos = read_electrodes(electrodes_file)

    nd = array(list(ndindex(mri.shape)))
    ndi = from_mrifile_to_chan(img, nd)

    if graymatter:
        gm_mri = nload(str(graymatter)).get_data().astype(bool)
        mri[~gm_mri] = NaN

    lg.info(f'Computing fMRI values for {measure_nii.name} at {len(labels)} electrodes and {len(distances)} "{metric}" values')
    fmri_vals = compute_weighted_averages(distances, elec_pos, mri, ndi, metric, parallel)

    write_output(output_dir / output_file, labels, distances, fmri_vals)
    D = {
        'program': 'fmri_at_elec',
        'version': VERSION,
        'date': current_time,
        'fMRI_file': str(measure_nii),
        'electrode_file': str(electrodes_file),
        'graymatter_file': str(graymatter),
        'metric': metric,
        'values': distances,
        }

    json_file = output_file.with_suffix('.json')
    with json_file.open('w') as f:
        dump(D, f, indent=2)


def name(s):
    return s.name.split('.')[0]


def read_electrodes(electrode_file):
    DTYPE = [
        ('name', '<U4096'),
        ('x', '<f8'),
        ('y', '<f8'),
        ('z', '<f8'),
        ]
    elec = genfromtxt(electrode_file, dtype=DTYPE, skip_header=1, delimiter='\t')
    return elec['name'], c_[elec['x'], elec['y'], elec['z']]


def write_output(output_file, labels, distances, fmri_vals):

    with output_file.open('w') as f:
        f.write('channel\t' + '\t'.join(str(one_k) for one_k in distances) + '\n')
        for one_label, val_at_elec in zip(labels, fmri_vals):
            f.write(one_label + '\t' + '\t'.join(str(one_val) for one_val in val_at_elec) + '\n')


def from_chan_to_mrifile(img, xyz):
    return apply_affine(inv(img.affine), xyz).astype(int)


def from_mrifile_to_chan(img, xyz):
    return apply_affine(img.affine, xyz)


def compute_weighted_averages(distances, elec_pos, mri, ndi, metric, parallel=False):
    partial_compute_chan = partial(compute_one_chan_one_distance, ndi=ndi, mri=mri, metric=metric)

    args = product(elec_pos, distances)
    if parallel:
        with Pool() as p:
            fmri_vals = p.starmap(partial_compute_chan, args)
    else:
        fmri_vals = [partial_compute_chan(*arg) for arg in args]

    return array(fmri_vals).reshape(-1, len(distances))


def compute_one_chan_one_distance(pos, distance, ndi, mri, metric):
    dist_chan = norm(ndi - pos, axis=1)

    if metric == 'gaussian':
        m = normdistr.pdf(dist_chan, scale=distance)
        m /= normdistr.pdf(0, scale=distance)  # normalize so that peak is at 1, so that it's easier to count voxels

    elif metric == 'sphere':
        m = zeros(dist_chan.shape)
        m[dist_chan <= distance] = 1

    elif metric == 'inverse':
        m = power(dist_chan, -1 * distance)

    m = m.reshape(mri.shape)
    m[isnan(mri)] = NaN

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m /= sum(m[isfinite(m)])  # normalize so that the sum of the finite numbers is 1

    mq = m * mri
    return nansum(mq)


def main():

    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        prog='fmri_at_elec',
        description="Calculate the weighted average around each electrode")
    parser.add_argument(
        '-f', '--fmri',
        help='path to NIfTI file with the fMRI values',
        required=True,
        )
    parser.add_argument(
        '-e', '--electrodes',
        help='path to electrodes in tsv format (columns: label, x, y, z in mm)',
        required=True,
        )
    parser.add_argument(
        '-o', '--output',
        help='directory with the results',
        required=True,
        )
    parser.add_argument(
        '-g', '--graymatter',
        help='path to NIfTI file with the graymatter (1 -> gray matter, to include, 0 -> to exclude)',
        )
    parser.add_argument(
        '-m', '--metric',
        help='metric to compute weighted mean',
        choices=('gaussian', 'sphere', 'inverse'),
        default='gaussian',
        )
    parser.add_argument(
        '-s', '--sigma',
        help='(only for gaussian) kernel widths',
        nargs='+',
        type=float,
        )
    parser.add_argument(
        '-r', '--radius',
        help='(only for sphere) radius of the sphere',
        nargs='+',
        type=float,
        )
    parser.add_argument(
        '-c', '--coefficient',
        help='(only for inverse) 1 -> inverse of the distance, 2 -> square of the distance, 3 -> cube of the distance',
        nargs='+',
        type=float,
        )
    parser.add_argument(
        '-l', '--log',
        default='info',
        help='Logging level: info (default), debug',
        )
    parser.add_argument(
        '-p', '--parallel',
        action='store_true',
        help='Run each electrode in parallel',
        )

    args = parser.parse_args()

    if args.log[:1].lower() == 'i':
        LEVEL = INFO
        FORMAT = '{asctime:<10}{message}'

    elif args.log[:1].lower() == 'd':
        LEVEL = DEBUG
        FORMAT = '{asctime:<10}{levelname:<10}(l. {lineno: 5d}): {message}'

    DATE_FORMAT = '%H:%M:%S'
    basicConfig(format=FORMAT, datefmt=DATE_FORMAT, style='{', level=LEVEL)
    lg.info(f'Calculate fMRI values at electrodes, version {VERSION: 4d}')

    if args.metric == 'gaussian':
        if args.sigma is None:
            parser.error('You need to specify "--sigma" with "gaussian"')
        distance = args.sigma
    elif args.metric == 'sphere':
        if args.radius is None:
            parser.error('You need to specify "--radius" with "sphere"')
        distance = args.radius
    elif args.metric == 'inverse':
        if args.coefficient is None:
            parser.error('You need to specify "--coefficient" with "inverse"')
        distance = args.coefficient

    graymatter = args.graymatter
    if graymatter is not None:
        graymatter = Path(graymatter).resolve()

    calc_fmri_at_elec(
        measure_nii=Path(args.fmri).resolve(),
        electrodes_file=Path(args.electrodes).resolve(),
        output_dir=Path(args.output).resolve(),
        graymatter=graymatter,
        metric=args.metric,
        distances=distance,
        parallel=args.parallel,
        )


if __name__ == '__main__':
    main()
