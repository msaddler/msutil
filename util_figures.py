import sys
import os
import numpy as np
import copy
import matplotlib.pyplot
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors


def get_color_list(num_colors, cmap_name='Accent'):
    '''
    Helper function returns list of colors for plotting.
    '''
    if isinstance(cmap_name, list):
        return cmap_name
    cmap = matplotlib.pyplot.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=num_colors)
    scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_list = [scalar_map.to_rgba(x) for x in range(num_colors)]
    return color_list


def format_axes(ax,
                str_title=None,
                str_xlabel=None,
                str_ylabel=None,
                fontsize_title=12,
                fontsize_labels=12,
                fontsize_ticks=12,
                fontweight_title=None,
                fontweight_labels=None,
                xscale='linear',
                yscale='linear',
                xlimits=None,
                ylimits=None,
                xticks=None,
                yticks=None,
                xticks_minor=None,
                yticks_minor=None,
                xticklabels=None,
                yticklabels=None,
                spines_to_hide=[],
                major_tick_params_kwargs_update={},
                minor_tick_params_kwargs_update={}):
    '''
    Helper function for setting axes-related formatting parameters.
    '''
    ax.set_title(str_title, fontsize=fontsize_title, fontweight=fontweight_title)
    ax.set_xlabel(str_xlabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_ylabel(str_ylabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    
    if xticks_minor is not None:
        ax.set_xticks(xticks_minor, minor=True)
    if yticks_minor is not None:
        ax.set_yticks(yticks_minor, minor=True)
    if xticks is not None:
        ax.set_xticks(xticks, minor=False)
    if yticks is not None:
        ax.set_yticks(yticks, minor=False)
    if xticklabels is not None:
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(xticklabels, minor=False)
    if yticklabels is not None:
        ax.set_yticklabels([], minor=True)
        ax.set_yticklabels(yticklabels, minor=False)
    
    major_tick_params_kwargs = {
        'axis': 'both',
        'which': 'major',
        'labelsize': fontsize_ticks,
        'length': fontsize_ticks/2,
        'direction': 'out',
    }
    major_tick_params_kwargs.update(major_tick_params_kwargs_update)
    ax.tick_params(**major_tick_params_kwargs)
    
    minor_tick_params_kwargs = {
        'axis': 'both',
        'which': 'minor',
        'labelsize': fontsize_ticks,
        'length': fontsize_ticks/4,
        'direction': 'out',
    }
    minor_tick_params_kwargs.update(minor_tick_params_kwargs_update)
    ax.tick_params(**minor_tick_params_kwargs)
    
    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)
    
    return ax


def make_line_plot(ax,
                   x,
                   y,
                   legend_on=True,
                   kwargs_plot={},
                   kwargs_legend={},
                   **kwargs_format_axes):
    '''
    Helper function for basic line plot with optional legend.
    '''
    kwargs_plot_tmp = {
        'marker': '',
        'ls': '-',
        'color': [0, 0, 0],
        'lw': 1,
    }
    kwargs_plot_tmp.update(kwargs_plot)
    ax.plot(x, y, **kwargs_plot_tmp)
    ax = format_axes(ax, **kwargs_format_axes)
    if legend_on:
        kwargs_legend_tmp = {
            'loc': 'lower right',
            'frameon': False,
            'handlelength': 1.0,
            'markerscale': 1.0,
            'fontsize': 12,
        }
        kwargs_legend_tmp.update(kwargs_legend)
        ax.legend(**kwargs_legend_tmp)
    return ax


def make_nervegram_plot(ax,
                        nervegram,
                        sr=20000,
                        cfs=None,
                        cmap=matplotlib.cm.gray,
                        fontsize_title=12,
                        fontsize_labels=12,
                        fontsize_legend=12,
                        fontsize_ticks=12,
                        fontweight_title=None,
                        fontweight_labels=None,
                        nxticks=6,
                        nyticks=5,
                        tmin=None,
                        tmax=None,
                        treset=True,
                        vmin=None,
                        vmax=None,
                        vticks=None,
                        str_title=None,
                        str_xlabel='Time (ms)',
                        str_ylabel='Characteristic Frequency (Hz)',
                        str_clabel=None):
    '''
    Helper function for visualizing auditory nervegram (or similar) representation.
    '''
    nervegram = np.squeeze(nervegram)
    assert len(nervegram.shape) == 2, "nervegram must be 2D array"
    t = np.arange(0, nervegram.shape[1]) / sr
    if (tmin is not None) and (tmax is not None):
        t_IDX = np.logical_and(t >= tmin, t < tmax)
        t = t[t_IDX]
        nervegram = nervegram[:, t_IDX]
    if treset:
        t = t - t[0]
    time_idx = np.linspace(0, t.shape[0]-1, nxticks, dtype=int)
    time_labels = ['{:.0f}'.format(1e3 * t[itr0]) for itr0 in time_idx]
    if cfs is None:
        cfs = np.arange(0, nervegram.shape[0])
    else:
        cfs = np.array(cfs)
        assert cfs.shape[0] == nervegram.shape[0], "cfs.shape[0] must match nervegram.shape[0]"
    freq_idx = np.linspace(0, cfs.shape[0]-1, nyticks, dtype=int)
    freq_labels = ['{:.0f}'.format(cfs[itr0]) for itr0 in freq_idx]
    
    im_nervegram = ax.imshow(nervegram,
                             origin='lower',
                             aspect='auto',
                             extent=[0, nervegram.shape[1], 0, nervegram.shape[0]],
                             cmap=cmap,
                             vmin=vmin,
                             vmax=vmax)
    
    if str_clabel is not None:
        cbar = matplotlib.pyplot.colorbar(im_nervegram, ax=ax, pad=0.02)
        cbar.ax.set_ylabel(str_clabel, fontsize=fontsize_labels)
        if vticks is not None:
            cbar.set_ticks(vticks)
        else:
            cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nyticks, integer=True))
        cbar.ax.tick_params(direction='out',
                            axis='both',
                            which='both',
                            labelsize=fontsize_ticks,
                            length=fontsize_ticks/2)
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%03d'))
    
    ax = format_axes(ax,
                     str_title=str_title,
                     str_xlabel=str_xlabel,
                     str_ylabel=str_ylabel,
                     fontsize_title=fontsize_title,
                     fontsize_labels=fontsize_labels,
                     fontsize_ticks=fontsize_ticks,
                     fontweight_title=fontweight_title,
                     fontweight_labels=fontweight_labels,
                     xticks=time_idx,
                     yticks=freq_idx,
                     xticklabels=time_labels,
                     yticklabels=freq_labels)
    
    return ax
