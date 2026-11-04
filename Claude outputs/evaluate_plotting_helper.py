# pylint: disable=invalid-name
""" helper file containing plotting functions to evaluate contributions to the
    Fast Calorimeter Challenge 2022.

    by C. Krause

    Modified: every 1D histogram now has a gen/ref ratio panel underneath
    (with a statistical-uncertainty band). The binning and the separation power
    are unchanged from the original code.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


########## common helper: histogram + ratio panel + separation power ##########

def _hist_with_ratio(ref_vals, gen_vals, bins, arg, filename, print_label, file_label,
                     title=None, xlabel=None, xscale='linear', yscale='log', xlim=None,
                     ratio_ylim=(0.5, 1.5)):
    """ Draws reference vs generated histograms (top) and their ratio gen/ref (bottom),
        saves the figure and writes the separation power, depending on arg.mode.

        ref_vals, gen_vals : 1D arrays of one observable (one value per event)
        bins               : bin edges (array) or number of bins (int)
        filename           : name of the png file inside arg.output_dir
        print_label        : text used in "Separation power of ... histogram"
        file_label         : text written to histogram_chi2_{dataset}.txt
    """
    ref_vals = np.asarray(ref_vals).ravel()
    gen_vals = np.asarray(gen_vals).ravel()

    # an integer number of bins is turned into edges from the reference sample,
    # exactly like plt.hist(reference, bins=40) did in the original code
    if np.ndim(bins) == 0:
        bins = np.histogram_bin_edges(ref_vals, bins=bins)
    bins = np.asarray(bins, dtype=float)
    widths = np.diff(bins)

    # raw counts (needed for the statistical error) ...
    n_ref, _ = np.histogram(ref_vals, bins=bins)
    n_gen, _ = np.histogram(gen_vals, bins=bins)
    # ... and the same normalisation as plt.hist(..., density=True)
    counts_ref = n_ref / (n_ref.sum() * widths) if n_ref.sum() > 0 else np.zeros_like(widths)
    counts_data = n_gen / (n_gen.sum() * widths) if n_gen.sum() > 0 else np.zeros_like(widths)

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(6, 7), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})

    # top panel: the two histograms (same style as the original plots)
    ax.stairs(counts_ref, bins, fill=True, alpha=0.2, color='C0', label='reference')
    ax.stairs(counts_data, bins, linewidth=3., color='C1', label='generated')
    if title is not None:
        ax.set_title(title)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(fontsize=20)

    # bottom panel: ratio gen/ref with a relative error sqrt(1/N_gen + 1/N_ref)
    ratio = np.divide(counts_data, counts_ref, out=np.full_like(counts_data, np.nan),
                      where=(counts_ref > 0))
    rel_err = np.full_like(ratio, np.nan)
    ok = (n_ref > 0) & (n_gen > 0)
    rel_err[ok] = np.sqrt(1. / n_ref[ok] + 1. / n_gen[ok])
    band_hi = np.where(ok, ratio * (1. + rel_err), np.nan)
    band_lo = np.where(ok, ratio * (1. - rel_err), np.nan)
    rax.stairs(band_hi, bins, baseline=band_lo, fill=True, alpha=0.3, color='C1')
    rax.stairs(ratio, bins, baseline=None, color='C1', linewidth=1.5)   # no drop to 0 at the edges
    rax.axhline(1.0, color='k', ls='--', lw=1)
    rax.set_ylim(*ratio_ylim)
    rax.set_ylabel('gen / ref')
    if xlabel is not None:
        rax.set_xlabel(xlabel)

    fig.tight_layout()
    if arg.mode in ['all', 'hist-p', 'hist']:
        fig.savefig(os.path.join(arg.output_dir, filename), dpi=300)

    seps = None
    if arg.mode in ['all', 'hist-chi', 'hist']:
        seps = _separation_power(counts_ref, counts_data, bins)
        print("Separation power of {} histogram: {}".format(print_label, seps))
        with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                  'a') as f:
            f.write('{}: \n'.format(file_label))
            f.write(str(seps))
            f.write('\n\n')
    plt.close(fig)
    return seps


########## 2D shower images (unchanged) ##########

def plot_layer_comparison(hlf_class, data, reference_class, reference_data, arg, show=False):
    """ plots showers of of data and reference next to each other, for comparison """
    num_layer = len(reference_class.relevantLayers)
    vmax = np.max(reference_data)
    layer_boundaries = np.unique(reference_class.bin_edges)
    for idx, layer_id in enumerate(reference_class.relevantLayers):
        plt.figure(figsize=(6, 4))
        reference_data_processed = reference_data\
            [:, layer_boundaries[idx]:layer_boundaries[idx+1]]
        reference_class._DrawSingleLayer(reference_data_processed,
                                         idx, filename=None,
                                         title='Reference Layer '+str(layer_id),
                                         fig=plt.gcf(), subplot=(1, 2, 1), vmax=vmax,
                                         colbar='None')
        data_processed = data[:, layer_boundaries[idx]:layer_boundaries[idx+1]]
        hlf_class._DrawSingleLayer(data_processed,
                                   idx, filename=None,
                                   title='Generated Layer '+str(layer_id),
                                   fig=plt.gcf(), subplot=(1, 2, 2), vmax=vmax, colbar='both')

        filename = os.path.join(arg.output_dir,
                                'Average_Layer_{}_dataset_{}.png'.format(layer_id, arg.dataset))
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        plt.close()


########## dataset 1 only, 4x4 grid (unchanged, no ratio panel) ##########

def plot_Etot_Einc_discrete(hlf_class, reference_class, arg):
    """ plots Etot normalized to Einc histograms for each Einc in ds1 """
    # hardcode boundaries?
    bins = np.linspace(0.4, 1.4, 21)
    plt.figure(figsize=(10, 10))
    target_energies = 2**np.linspace(8, 23, 16)
    for i in range(len(target_energies)-1):
        if i > 3 and 'photons' in arg.dataset:
            bins = np.linspace(0.9, 1.1, 21)
        energy = target_energies[i]
        which_showers_ref = ((reference_class.Einc.squeeze() >= target_energies[i]) & \
                             (reference_class.Einc.squeeze() < target_energies[i+1])).squeeze()
        which_showers_hlf = ((hlf_class.Einc.squeeze() >= target_energies[i]) & \
                             (hlf_class.Einc.squeeze() < target_energies[i+1])).squeeze()
        ax = plt.subplot(4, 4, i+1)
        counts_ref, _, _ = ax.hist(reference_class.GetEtot()[which_showers_ref] /\
                                   reference_class.Einc.squeeze()[which_showers_ref],
                                   bins=bins, label='reference', density=True,
                                   histtype='stepfilled', alpha=0.2, linewidth=2.)
        counts_data, _, _ = ax.hist(hlf_class.GetEtot()[which_showers_hlf] /\
                                    hlf_class.Einc.squeeze()[which_showers_hlf], bins=bins,
                                    label='generated', histtype='step', linewidth=3., alpha=1.,
                                    density=True)
        if i in [0, 1, 2]:
            energy_label = 'E = {:.0f} MeV'.format(energy)
        elif i in np.arange(3, 12):
            energy_label = 'E = {:.1f} GeV'.format(energy/1e3)
        else:
            energy_label = 'E = {:.1f} TeV'.format(energy/1e6)
        ax.text(0.95, 0.95, energy_label, ha='right', va='top',
                transform=ax.transAxes)
        ax.set_xlabel(r'$E_{\text{tot}} / E_{\text{inc}}$')
        ax.xaxis.set_label_coords(1., -0.15)
        ax.set_ylabel('counts')
        ax.yaxis.set_ticklabels([])
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        seps = _separation_power(counts_ref, counts_data, bins)
        print("Separation power of Etot / Einc at E = {} histogram: {}".format(energy, seps))
        with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                  'a') as f:
            f.write('Etot / Einc at E = {}: \n'.format(energy))
            f.write(str(seps))
            f.write('\n\n')
        h, l = ax.get_legend_handles_labels()
    ax = plt.subplot(4, 4, 16)
    ax.legend(h, l, loc='center', fontsize=20)
    ax.axis('off')
    filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}_E_i.png'.format(arg.dataset))
    plt.savefig(filename, dpi=300)
    plt.close()


########## 1D histograms, all with ratio panel ##########

def plot_Etot_Einc(hlf_class, reference_class, arg):
    """ plots Etot normalized to Einc histogram """
    _hist_with_ratio(reference_class.GetEtot() / reference_class.Einc.squeeze(),
                     hlf_class.GetEtot() / hlf_class.Einc.squeeze(),
                     np.linspace(0.5, 1.5, 101), arg,
                     filename='Etot_Einc_dataset_{}.png'.format(arg.dataset),
                     print_label='Etot / Einc', file_label='Etot / Einc',
                     xlabel=r'$E_{\text{tot}} / E_{\text{inc}}$',
                     yscale='linear', xlim=(0.5, 1.5))


def plot_E_layers(hlf_class, reference_class, arg):
    """ plots energy deposited in each layer """
    for key in hlf_class.GetElayers().keys():
        if arg.x_scale == 'log':
            bins = np.logspace(np.log10(arg.min_energy),
                               np.log10(reference_class.GetElayers()[key].max()),
                               40)
        else:
            bins = 40
        _hist_with_ratio(reference_class.GetElayers()[key], hlf_class.GetElayers()[key],
                         bins, arg,
                         filename='E_layer_{}_dataset_{}.png'.format(key, arg.dataset),
                         print_label='E layer {}'.format(key),
                         file_label='E layer {}'.format(key),
                         title="Energy deposited in layer {}".format(key),
                         xlabel=r'$E$ [MeV]',
                         xscale='log' if arg.x_scale == 'log' else 'linear')


def _center_lim(arg, key):
    """ x-range of the center-of-energy histograms (eta, phi) """
    if arg.dataset in ['2', '3']:
        return (-30., 30.)
    if key in [12, 13]:
        return (-500., 500.)
    return (-100., 100.)


def _width_lim(arg, key):
    """ x-range of the width histograms (eta, phi) """
    if arg.dataset in ['2', '3']:
        return (0., 30.)
    if key in [12, 13]:
        return (0., 400.)
    return (0., 100.)


def plot_ECEtas(hlf_class, reference_class, arg):
    """ plots center of energy in eta """
    for key in hlf_class.GetECEtas().keys():
        lim = _center_lim(arg, key)
        _hist_with_ratio(reference_class.GetECEtas()[key], hlf_class.GetECEtas()[key],
                         np.linspace(*lim, 101), arg,
                         filename='ECEta_layer_{}_dataset_{}.png'.format(key, arg.dataset),
                         print_label='EC Eta layer {}'.format(key),
                         file_label='EC Eta layer {}'.format(key),
                         title=r"Center of Energy in $\Delta\eta$ in layer {}".format(key),
                         xlabel=r'[mm]', xlim=lim)


def plot_ECPhis(hlf_class, reference_class, arg):
    """ plots center of energy in phi """
    for key in hlf_class.GetECPhis().keys():
        lim = _center_lim(arg, key)
        _hist_with_ratio(reference_class.GetECPhis()[key], hlf_class.GetECPhis()[key],
                         np.linspace(*lim, 101), arg,
                         filename='ECPhi_layer_{}_dataset_{}.png'.format(key, arg.dataset),
                         print_label='EC Phi layer {}'.format(key),
                         file_label='EC Phi layer {}'.format(key),
                         title=r"Center of Energy in $\Delta\phi$ in layer {}".format(key),
                         xlabel=r'[mm]', xlim=lim)


def plot_ECWidthEtas(hlf_class, reference_class, arg):
    """ plots width of center of energy in eta """
    for key in hlf_class.GetWidthEtas().keys():
        lim = _width_lim(arg, key)
        _hist_with_ratio(reference_class.GetWidthEtas()[key], hlf_class.GetWidthEtas()[key],
                         np.linspace(*lim, 101), arg,
                         filename='WidthEta_layer_{}_dataset_{}.png'.format(key, arg.dataset),
                         print_label='Width Eta layer {}'.format(key),
                         file_label='Width Eta layer {}'.format(key),
                         title=r"Width of Center of Energy in $\Delta\eta$ in layer {}".format(key),
                         xlabel=r'[mm]', xlim=lim)


def plot_ECWidthPhis(hlf_class, reference_class, arg):
    """ plots width of center of energy in phi """
    for key in hlf_class.GetWidthPhis().keys():
        lim = _width_lim(arg, key)
        _hist_with_ratio(reference_class.GetWidthPhis()[key], hlf_class.GetWidthPhis()[key],
                         np.linspace(*lim, 101), arg,
                         filename='WidthPhi_layer_{}_dataset_{}.png'.format(key, arg.dataset),
                         print_label='Width Phi layer {}'.format(key),
                         file_label='Width Phi layer {}'.format(key),
                         title=r"Width of Center of Energy in $\Delta\phi$ in layer {}".format(key),
                         xlabel=r'[mm]', xlim=lim)


def plot_cell_dist(shower_arr, ref_shower_arr, arg):
    """ plots voxel energies across all layers """
    if arg.x_scale == 'log':
        bins = np.logspace(np.log10(arg.min_energy),
                           np.log10(ref_shower_arr.max()),
                           50)
    else:
        bins = 50
    _hist_with_ratio(ref_shower_arr.flatten(), shower_arr.flatten(), bins, arg,
                     filename='voxel_energy_dataset_{}.png'.format(arg.dataset),
                     print_label='voxel distribution', file_label='Voxel distribution',
                     title=r"Voxel energy distribution", xlabel=r'$E$ [MeV]',
                     xscale='log' if arg.x_scale == 'log' else 'linear')


def plot_sparsity(hlf_class, reference_class, arg):
    """ plots sparsity in each layer """
    for key in hlf_class.GetSparsity().keys():
        _hist_with_ratio(reference_class.GetSparsity()[key], hlf_class.GetSparsity()[key],
                         np.linspace(0., 1., 21), arg,
                         filename='Sparsity_layer_{}_dataset_{}.png'.format(key, arg.dataset),
                         print_label='Sparsity layer {}'.format(key),
                         file_label='Sparsity layer {}'.format(key),
                         title="Sparsity in layer {}".format(key),
                         xlabel=r'fraction of voxels with 0 energy deposition',
                         yscale='linear')


def plot_ECR(hlf_class, reference_class, arg):
    """ plots center of energy in r """
    for key in hlf_class.GetECR().keys():
        if arg.dataset in ['2', '3']:
            lim = (0., 42.)
        elif key in [12, 13, 14]:
            lim = (0., 1500.)
        else:
            lim = (0., 500.)
        _hist_with_ratio(reference_class.GetECR()[key], hlf_class.GetECR()[key],
                         np.linspace(*lim, 101), arg,
                         filename='ECR_layer_{}_dataset_{}.png'.format(key, arg.dataset),
                         print_label='EC r layer {}'.format(key),
                         file_label='EC r layer {}'.format(key),
                         title=r"Center of Energy in $r$ in layer {}".format(key),
                         xlabel=r'[mm]', xlim=lim)


def plot_ECWidthR(hlf_class, reference_class, arg):
    """ plots width of center of energy in r """
    for key in hlf_class.GetWidthR().keys():
        if arg.dataset in ['2', '3']:
            lim = (0., 20.)
        elif key in [12, 13, 14]:
            lim = (0., 1500.)
        else:
            lim = (0., 500.)
        _hist_with_ratio(reference_class.GetWidthR()[key], hlf_class.GetWidthR()[key],
                         np.linspace(*lim, 101), arg,
                         filename='WidthR_layer_{}_dataset_{}.png'.format(key, arg.dataset),
                         print_label='Width r layer {}'.format(key),
                         file_label='Width r layer {}'.format(key),
                         title=r"Width of Center of Energy in $r$ in layer {}".format(key),
                         xlabel=r'[mm]', xlim=lim)


def _separation_power(hist1, hist2, bins):
    """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
    ret = (hist1 - hist2)**2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()
