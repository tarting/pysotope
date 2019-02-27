'''
pysotope - a package for inverting double spike isotope analysis data
This module produces summary diagrams from reduced data
'''


# pysotope - a package for inverting double spike isotope analysis data
#
#     Copyright (C) 2018 Trygvi Bech Arting
#
#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 2 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License along
#     with this program; if not, write to the Free Software Foundation, Inc.,
#     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from pysotope.invert import gen_filter_function



def generate_summaryplot(summaries_df, spec, figfolder='./GFX', **kwargs):
    plot_vars = spec['plot_vars']['summary']
    n_diagrams = len(plot_vars)
    fig, axes = plt.subplots(1,n_diagrams,figsize=(12,8), sharey=True)

    summaries_df_filtered = summaries_df[summaries_df.ignore.apply(lambda i: not i)]
    if '2sigma_filter' in spec['plot_vars']:
        sigmax2_cutoff = spec['plot_vars']['2sigma_filter']
        summaries_df_filtered = summaries_df_filtered[
            summaries_df_filtered[plot_vars[0]+'_2SE'] < sigmax2_cutoff]
    if 'intens_min' in spec['plot_vars']:
        intens_min = spec['plot_vars']['intens_min']
        summaries_df_filtered = summaries_df_filtered[
            summaries_df_filtered[plot_vars[1]+ '_min' ] > intens_min]
    if 'min_cycles' in spec['plot_vars']:
        min_cycles = spec['plot_vars']['min_cycles']
        summaries_df_filtered = summaries_df_filtered[
            summaries_df_filtered[plot_vars[1]+ '_N' ] > min_cycles]
        
    
    yticklabels = []
    yticks = []
    for i, (bead, df) in enumerate(summaries_df_filtered.groupby('bead_id')):
        yticklabels.append(bead)
        yticks.append(i)
        ys = np.linspace(i-.3, i+.3, len(df))

        for j, (ax, var) in enumerate(zip(axes, plot_vars)):
            if j == 0:
                ax.errorbar(x=df[var], xerr=df[var+'_2SE'], y = ys,
                            marker='.', mfc='w', ls='', elinewidth=1, capsize=3)
                ax.set_xlabel(var + ' $\pm 2ϵ$')
            elif j == 1:
                ax.errorbar(x=df[var]*1000, y=ys, xerr=[df[var]*1000-df[var+'_min']*1000, df[var+'_max']*1000-df[var]*1000], marker='.', mfc='w', ls='', elinewidth=1, capsize=3)
                ax.set_xlabel(var + ' (min, max) [mV]')
            elif j == 2:
                ax.errorbar(x=df[var], y=ys, xerr=df[var+'_2STD'], marker='.', mfc='w', ls='', elinewidth=1, capsize=3)
                ax.set_xlabel(var)
            elif type(var) is list:
                numer, denom = var
                ax.errorbar(x=df[numer]/df[denom], y=ys, marker='.', mfc='w', ls='')
                ax.set_xlabel(numer + '/' + denom)
                ax.set_xscale('log')

            else: 
                ax.errorbar(x=df[var], y=ys, marker='.', mfc='w', ls='')
                ax.set_xlabel(var)


    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(yticklabels)
    axes[0].set_ylim(*axes[0].get_ylim())
    axes[0].plot([0,0], axes[0].get_ylim(), '-k', lw=1, zorder=-1)
    fig.tight_layout()
    fig.savefig(os.path.join(figfolder,'0_{}_overview.png'.format(spec['element'])), **kwargs)
    plt.close(fig)



def generate_cycleplots(cycle_df, spec, figfolder='./GFX', **kwargs):
    plot_vars = spec['plot_vars']['cycles']
    
    filter_function = gen_filter_function(**spec['outlier_rejection'])
    
    
    new_plot_vars = []
    for i,var in enumerate(plot_vars):
        if type(var) is list:
            numer, denom = var
            new_var = numer+'/'+denom
            cycle_df[new_var] = cycle_df[numer]/cycle_df[denom]
            new_plot_vars.append(new_var)
        else:
            new_plot_vars.append(var)
    
    
    n_diagrams = len(plot_vars)
    for bead_id, bead_df in tqdm(cycle_df.groupby('bead_id')):
        height = math.ceil(n_diagrams/2)
        fig, axes = plt.subplots(height,2,figsize=(16,3*height), sharex=True)
        axes=axes.T.ravel()


        # Enforce sorting of run numbers rather than using groupby
        run_numbers = sorted(list(bead_df['run_no'].unique()))

        for ax, ylab in zip(axes,
                            new_plot_vars):
            xs = [-1]
            bead_xs = []
            bead_ys = []
            
            rej_xs = []
            rej_ys = []
            
            for run_no in run_numbers:
                
                run_df = bead_df[bead_df.run_no == run_no].copy()
                if len(run_df) <= 1:
                    continue
                ys = run_df[ylab]
                xs = xs[-1] + np.array(range(len(run_df))) + 1
                
                l = ax.plot(xs, ys, '.')[0]
                root_n = len(ys)**0.5
                ax.fill_between([min(xs), max(xs)], 2*[np.mean(ys) - np.std(ys)*2], 2*[np.mean(ys) + np.std(ys)*2], alpha=0.1, color = l.get_color())
                ax.fill_between([min(xs), max(xs)], 2*[np.mean(ys) - (np.std(ys)*2)/root_n], 2*[np.mean(ys) +( np.std(ys)*2)/root_n], alpha=0.2, color = l.get_color())
                ax.plot([min(xs), max(xs)], 2*[np.mean(ys)], '-', color = l.get_color())
                
                # Get coordinates for rejected points
                mean = np.median(ys)
                filterdf = pd.DataFrame.from_dict({'ys':ys,'xs':xs})
                filterdf['distance'] = abs(filterdf['ys']-np.mean(ys))
                filterdf = filterdf.sort_values('distance', ascending=False)
                filter_items = filter_function(ys, True)
                if len(filter_items) > 0:
                    rejects = max(filter_items)
                    rejected = filterdf.iloc[:rejects]
                    rej_xs += list(rejected['xs'])
                    rej_ys += list(rejected['ys'])

                bead_xs += list(xs)
                bead_ys += list(ys)
                
            ax.set_xlim(*ax.get_xlim())
            ax.plot(ax.get_xlim(), 2*[np.mean(bead_ys)], ':k', zorder=-1)
            root_n = len(bead_ys)**0.5
            ax.fill_between(ax.get_xlim(), 2*[np.mean(bead_ys) - (np.std(bead_ys)*2)/root_n], 2*[np.mean(bead_ys) +( np.std(bead_ys)*2)/root_n], alpha=0.4, zorder=-1, color='k')
            ax.plot(rej_xs, rej_ys, 'xr')
            ax.set_title(ylab)
        fig.suptitle(bead_id)
        fig.tight_layout(rect=[0,0,1,.95])
        fig.savefig(os.path.join(figfolder,'1_{}_{}.png'.format(spec['element'],bead_id)), **kwargs)
        plt.close(fig)
