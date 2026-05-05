# core
import gzip
import json
import time
import warnings

# external
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cdlib
from cdlib import readwrite
import seaborn as sns

# internal
from clustering import perform_overlapping_clustering, plot_overlapping_clusters, beautify_name, build_graph, compute_metrics, compute_fit


RESULTS_PATH = "results/"


def cluster_and_plot(ABC_data, optimal_pars, cdkwargs, bnames, **plot_kwargs):
    _, coms = perform_overlapping_clustering(ABC_data=ABC_data, pars=optimal_pars, verbose=False, **cdkwargs)
    kwargs = plot_overlapping_clusters(ABC_data, coms, bnames, thresh=optimal_pars["thresh"], edge_width=None, **plot_kwargs)
    return coms, kwargs


def plot_index_conditions(ABC_data, optimal_pars, cdkwargs, labels, bnames, hops=1,
                          LTCs=("Type 1 Diabetes Mellitus", "Type 2 Diabetes Mellitus", "CHD", "Multiple Sclerosis"), **plot_kwargs):
    side = 6
    figs = []
    for LTC in LTCs:
        for sex in range(2):
            fig, axes = plt.subplots(2, 3, figsize=(3*side, 2*side))
            for ind in range(2+sex, 14, 2):
                # print(labels[ind], "\n=====================")
                ax = axes[int((ind-2)/6), int((ind-2)/2) % 3]
                cluster_and_plot(ABC_data[ind], optimal_pars[ind], cdkwargs, bnames,
                                 LTC=LTC, ego_clusters=True, hops=hops,
                                 mplfig=ax, ink_scale=.01, fit_view=2.1, **plot_kwargs)
                # adjust_aspect=False, # fit_view_ink=False, # fit_view=False,
                                          #vertex_font_size=.5, edge_width=0.4, vertex_max_size=.0000004);
                ax.set(title=labels[ind])
                ax.yaxis.set_visible(False), ax.xaxis.set_visible(False)
                [spine.set_visible(False) for spine in ax.spines.values()]
            fig.suptitle(LTC if type(LTC) != list else ", ".join(LTC))
            fig.tight_layout()
            figs.append(fig)
    return figs


def plot_body_systems(ABC_data, optimal_pars, cdkwargs, labels, bnames, groups_dict, **plot_kwargs):
    body_systems = {"Circulatory":[
        'Abdominal Aortic Aneurysm', 'Atrial Fibrillation',
        'Cardiac conduction disorder', 'Cardiomyopathy',
        'CKD', 'CHD',
        'Heart Failure', 'Heart Valve Disorder',
        'Hypertension',
        'Peripheral Arterial Disease', 'Primary Pulmonary Hypertension',
        'Stroke/TIA',
        'Subarachnoid Haemorrhage', 'Supraventricular Tachycardia',], # 'Venous thromboembolic disease']
               "Mental and behavioural": groups_dict["Mental and behavioural"]}

    for body_sys in ("Mental and behavioural", "Circulatory"):
        print("\nXXXXXXXXXXXXXXXXXXXXXXXXXXX\n", body_sys, "\nXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")
        for ind in range(2, 14):
            print(labels[ind], "\n=====================")
            cluster_and_plot(ABC_data[ind], optimal_pars[ind], cdkwargs, bnames,
                                        LTC=body_systems[body_sys], ego_clusters=True, **plot_kwargs)
            

par_dict = {"thresh": r"Backboning parameter, $\delta$",
            "d_threshold": r"Density threshold, $d_{thresh}$",
            "cp_threshold": r"Cluster-Property threshold, $cp_{thresh}$",
           "t_in": r"$T_{in}$"}
par_dict_short = {"thresh": r"$\delta$",
            "d_threshold": r"$d_{thresh}$",
            "cp_threshold": r"$cp_{thresh}$",
           "t_in": r"$T_{in}$"}
stats_name_dict = {"n_clusters": r"$C$",  # "N clusters",
                    "n_overlap": r"$O$",  # "N overlaps",
                    "unclustered": r"$U$",  # "Unclustered",
                    "max_size": r"$S$",  # "Max size",
                    "size": r"$N$",  # "N nodes",
                    "edges_per_node": r"$E$",  # "Edges/node",
                    "fit": r"$MCS$",  # "Fit",
                    "distance_min5": r"$R$"}  # "Stability"}

stats_long_name_dict = {"n_clusters": r"N of clusters ($C$)",
                    "n_overlap": r"N of multiple-membership conditions ($O$)",
                    "unclustered": r"Number of unclustered conditions ($U$)",
                    "max_size": r"Maximum cluster size ($S$)",
                    "size": r"N of conditions in the network ($N$)",
                    "edges_per_node": r"Nu of associations per node ($E$, after filtering)",
                    "fit": r"Multimorbidity Clustering Score ($MCS$)",
                    "distance_min5": r"Stability of solution ($R$)"}


class GridResults:
    """
    Class to store the results of a grid search.
    """

    def __init__(self, pars, resolution, gkwargs, cdkwargs):
        """
        Initialize the grid search with the parameters and resolution.
        """
        self.gkwargs = gkwargs
        self.cdkwargs = cdkwargs
        
        self.pars = pars
        self.resolution = resolution
        self.axes = {par: list(np.linspace(prange[0], prange[1], resolution)) for par, prange in pars.items()}
        self.shape = tuple([len(ax) for ax in self.axes.values()])
        self.indices = list(np.ndindex(self.shape))

    def find_index(self, pars):
        """
        Find the index of the value in the parameter sweep results.
        """
        return tuple([self.axes[par].index(pars[par]) if par in pars else slice(self.resolution) for par in self.pars])

    def compute_graphs(self, ABC_data):
        """
        Compute the graphs for each parameter combination.
        """
        gkwargs = dict(ABC_data=ABC_data, sig=True, backboning=True, verbose=False, gtype="ig", **self.gkwargs)
        return [build_graph(thresh=thresh, **gkwargs) for thresh in self.axes["thresh"]] if "thresh" in self.pars else [build_graph(**gkwargs)]

    def grid_search(self, ABC_data):
        graphs = self.compute_graphs(ABC_data)
        coms = np.full((self.resolution,) * len(self.pars), None)
        for indices in self.indices:
            pars_instance = {par: self.axes[par][indices[i]] for i, par in enumerate(self.pars)}
            _, coms[indices] = perform_overlapping_clustering(pars=pars_instance, verbose="thresh" not in self.pars, return_clustering_object=True,
                                                        graph=graphs[indices[list(self.pars).index("thresh")] if "thresh" in self.pars else 0],  **self.cdkwargs)
                
        return coms


    def compute_metrics(self, coms, ABC_data, include_distances: bool = True, window_size: int = 3):
        """
        Compute the fit for each parameter combination.
        """
        graphs = self.compute_graphs(ABC_data)
        distances = compute_distances(coms) if include_distances else False

        stats = []
        thresh_order = list(self.pars).index("thresh") if "thresh" in self.pars else None
        for inds in self.indices:
            pars_instance = {par: self.axes[par][inds[i]] for i, par in enumerate(self.pars)}
            
            thresh_ind = inds[thresh_order] if thresh_order is not None else 0
            row = {**pars_instance, **compute_metrics(coms[inds], graph=graphs[thresh_ind])}

            if include_distances:
                # TODO: Note that this just creates windows with direct lines along the axes, but not squares / cubes
                def get_window(w_size):
                    window = []
                    for w in range(1, w_size + 1):
                        for i in range(len(inds)):
                            window.append(distances[i][tuple([ind if m != i else ((ind - w) if ind >= w else 0) for m, ind in enumerate(inds)])])
                            window.append(distances[i][tuple([ind if m != i else (ind if ind < (len(distances) - w) else -1) for m, ind in enumerate(inds)])])
                    return window
                    
                for j in range(1, window_size + 1):
                    windw = get_window(j)
                    row[f"distance_avg{j}"] = sum(windw) / (2 * j * len(inds))
                    row[f"distance_min{j}"] = min(windw)
            
            stats.append(row)
        stats = pd.DataFrame(stats)
        return stats

    def df_to_tensor(self, df, col):
        """
        Convert a dataframe to a matrix with the values of the parameters.
        """
        tensor = np.full((self.resolution,) * len(self.pars), None)
        for _, row in df.iterrows():
            idx = tuple([self.axes[par].index(row[par]) for par in self.pars])
            tensor[idx] = row[col]
        return tensor
    
    def df_to_matrix(self, df, col, fixed_par=None):
        """
        Convert a dataframe to a matrix with the values of the parameters.
        """
        if fixed_par is not None:
            df = df[df[fixed_par[0]] == fixed_par[1]]
        axes_pars = [par for par in self.pars if fixed_par is None or par != fixed_par[0]]
        return pd.pivot_table(df, index=axes_pars[0], columns=axes_pars[1], values=col)
    
    def plot_heatmap(self, df, col, fixed_par=None, ax=None, title="", fit_kwargs=None, alpha_line=0.8, lw_line=1.2, cbar=True,
                     **hm_kwargs):
        """
        Plot a heatmap of the results.
        """
        if col == "fit" and fit_kwargs is not None:
            compute_fit(df, **fit_kwargs)

        matrix = self.df_to_matrix(df, col, fixed_par)
        matrix.index = pd.Float64Index(np.array([np.round(el, decimals=10) for el in matrix.index]), name=matrix.index.name)
        matrix.columns = pd.Float64Index(np.array([np.round(el, decimals=10) for el in matrix.columns]), name=matrix.columns.name)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        hmap = sns.heatmap(matrix, ax=ax, vmin=0, cbar=cbar, **hm_kwargs)
        if cbar:
            cbar = hmap.collections[0].colorbar
            cbar.set_label(stats_long_name_dict[col] if col[:4] != "dist" else stats_long_name_dict["distance_min5"])
        ax.set(xlabel=par_dict[matrix.columns.name], ylabel=par_dict[matrix.index.name], title=title)

        if col == "fit":
            index_of_max = np.unravel_index(np.argmax(matrix.values), matrix.values.shape)
            # line_kwargs = dict(c="yellow", ls="--", alpha=alpha_line, lw=lw_line)
            # ax.axhline(index_of_max[0] + .5, **line_kwargs)
            # ax.axvline(index_of_max[1] + .5, **line_kwargs)
            ax.plot(index_of_max[1] + .5, index_of_max[0] + .5, marker="o", c="yellow", markersize=3)

        return ax
    
    def plot_parameter_sweep(self, stats, par, title="", ax=None, fixed_pars=None, communities=None, distance_metric=None,
                             legend_args={}, twinx_visibility=True, verbose=False, sub_scores=None, **fit_kwargs):
        compute_fit(stats, distance_metric=distance_metric if communities is None else None, **fit_kwargs)
        if fixed_pars is not None:
            for key, val in fixed_pars.items():
                if key != par:
                    stats = stats[stats[key] == val]

        # ls = ('-', '--', '-.', ':', (6,2), (2,1), (0.5,0.5), (4,1,2,1))
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        for var in (stats_name_dict if sub_scores is None else sub_scores):
            label = stats_name_dict[var]
            if var == "n_overlap" and "n_overlap" not in stats:
                continue
            if var != "distance_min5":
                i = list(stats_name_dict.keys()).index(var)
                sns.lineplot(data=stats, x=par, y=var, ax=ax, label=label,  # f"{i+1} {label}",
                             errorbar=("pi", 50), color=f"C{i}")  # , ls=ls[i])
            else:
                ax.plot(0, 0, label=label, color="grey", alpha=0.5)  # label=f"{i+1} {label}"
        ax.set(xlabel=par_dict[par], ylabel="Sub-scores", ylim=(0, 80), xlim=self.pars[par], title=title)
        opt = stats[stats["fit"] == stats["fit"].max()][par].iloc[0]
        print(f"Optimal value: {opt} with {stats_name_dict['fit']}: {stats['fit'].max()}") if verbose else None
        ax.axvline(opt, color="k", ls=":", label="opt")  # f"{i+2} opt: {opt:.3f}")
        ax.legend(**legend_args) if legend_args is not None else None
        ax.grid()

        if (communities is not None or distance_metric is not None) and (sub_scores is None or 'distance_min5' in sub_scores):
            ax2 = ax.twinx()
            if distance_metric is not None:
                assert distance_metric in stats
                sns.lineplot(data=stats, x=par, y=distance_metric, ax=ax2, errorbar=("pi", 50), color="grey", alpha=0.5)  # label=distance_metric
            else:
                if fixed_pars is None:
                    assert communities.ndim == 1
                elif communities.ndim > 1:
                    idx = [self.find_index({par2: fixed_pars[par2]})[0] if par2 != par else slice(len(communities)) for par2 in self.pars if par2 in fixed_pars or par2 == par]
                    communities = communities[tuple(idx)]
                distances = compute_distances(communities)[0]
            
                prange = self.axes[par]
                ax2.plot([(prange[i] + prange[i+1])/2 for i in range(len(prange)-1)], distances, color="grey", alpha=0.5)

            ax.plot([0], [0], color="grey", alpha=0.5, label="Stability")  # Just for the legend
            ax2.set(ylim=(0, 1))
            ax2.spines['right'].set_color('grey')
            ax2.tick_params(axis='y', colors='grey')
            for label in ax2.get_yticklabels():
                label.set_color('grey')
            ax2.set_ylabel(r'Stability ($R$)' if twinx_visibility else '', color='grey')
            if not twinx_visibility:
                ax2.yaxis.set_visible(False)
                ax2.spines['right'].set_visible(False)

        return ax


def find_index(stats, par, val):
    """
    Find the index of the value in the parameter sweep results.
    """
    if par not in stats.columns:
        raise ValueError(f"Parameter '{par}' not found in the results.")
    if val not in stats[par].unique():
        raise ValueError(f"Value '{val}' not found in the parameter '{par}' results.")
    return list(stats[par].unique()).index(val)


def find_best_pars(stats, pars, warn=True, fit_kwargs=None):
    """
    Find the best parameters from the grid search results.
    """
    if "fit" not in stats:
        assert fit_kwargs is not None, "Fit needs to be computed. Provide a 'fit_kwargs' argument."
        compute_fit(stats, **fit_kwargs)
    best_fit = stats["fit"].max()
    best_results = stats[stats["fit"] == best_fit]
    if warn:
        warnings.simplefilter("default")
        warnings.warn(f"Multiple best results ({len(best_results)}) with the same fit: {best_fit}.") if len(best_results) > 1 else None
        warnings.simplefilter("ignore")
    best_results = best_results.iloc[-1]  # take the last one (arbitrary, but ensures d_threshold and cp_threshold are near 1.0)
    best_pars = {par: best_results[par] for par in pars}
    pars_idx = [find_index(stats, par, best_pars[par]) for par in pars]
    return best_pars, best_fit, pars_idx


def plot_statistics(stats, pars, age_labels, sex_labels, fit_kwargs):
    """
    Plot the statistics of the clustering results.
    """
    [compute_fit(stat, **fit_kwargs) for stat in stats.values()]
    
    results_df, edges, optimal_pars = [], [], {}
    for ind, stat in stats.items():
        optimal_pars[ind], _, _ = find_best_pars(stat, pars)
        row = stat.query(" and ".join([f"{par} == @optimal_pars[{ind}]['{par}']" for par in pars])).iloc[0].to_dict()
        [results_df.append({"stat": stats_name_dict[key], "val": val if key[:4] != "dist" else val * row["size"],
                            "age_group": age_labels[int(ind/2)], "sex": sex_labels[ind%2]}) for key, val in row.items() if key in stats_name_dict]
        edges.append(row["edges"])
    results_df = pd.DataFrame(results_df)

    g = sns.relplot(data=results_df, x="age_group", y="val", hue="stat", style="stat", col="sex",
                    kind="line", markers=True, hue_order=stats_name_dict.values(),)
    g.set_titles(col_template="{col_name}")  # "Sex: "
    for ax in g.axes.flat:
        ax.grid(True)
        ax.set(xlabel="Age group")  # , xlim=[0,len(results_df["age_group"].unique()) - 1])
    # for i, t in enumerate(g._legend.texts):
    #     label = t.get_text()
    #     t.set_text(f"{i+1} {label}")
    g.axes.flat[0].set(ylabel="Sub-scores")

    n_plots = len(pars)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3*n_plots), sharex=True)
    for k, par in enumerate(pars):  # take the keys of the first cohort
        if type(axes) == np.ndarray:
            ax = axes[k]
        else:
            ax = axes
            
        for i, style in enumerate(("solid", "--")):
            sns.lineplot(x=age_labels[1:], y=[optimal_pars[ind][par] for ind in range(2+i, 14, 2)],
                        label=sex_labels[i], ax=ax, marker="o", ls=style)
        if par == "thresh":
            ax.axhline(1.64, c="lightgrey", ls="-.")
        else:
            ax.set(ylim=(0, 1))
        ax.set(ylabel=par_dict[par], xlabel="Age group")
        ax.grid()
    fig.tight_layout()
    return fig, optimal_pars


def compute_distances(communities, dist=cdlib.evaluation.overlapping_normalized_mutual_information_LFK):
    if communities.ndim == 1:
        return (np.array([dist(communities[i], communities[i+1]).score for i in range(len(communities) - 1)]),)
    elif communities.ndim == 2:
        return (np.array([[dist(communities[i,j], communities[i+1,j]).score for j in range(len(communities))] for i in range(len(communities) - 1)]),
                np.array([[dist(communities[i,j], communities[i,j+1]).score for j in range(len(communities) - 1)] for i in range(len(communities))]))
    elif communities.ndim == 3:
        return (np.array([[[dist(communities[i,j,k], communities[i+1,j,k]).score for k in range(len(communities))] for j in range(len(communities))] for i in range(len(communities) - 1)]),
                np.array([[[dist(communities[i,j,k], communities[i,j+1,k]).score for k in range(len(communities))] for j in range(len(communities) - 1)] for i in range(len(communities))]),
                np.array([[[dist(communities[i,j,k], communities[i,j,k+1]).score for k in range(len(communities) - 1)] for j in range(len(communities))] for i in range(len(communities))]))
    

def save_communities(coms, fname="communities"):
    for ind, coms_k in coms.items():
        js_strings = [json.dumps({
                "communities": communities.communities,
                "algorithm": communities.method_name,
                "params": communities.method_parameters,
                "overlap": communities.overlap,
                "coverage": communities.node_coverage,
                }) for communities in coms_k.flatten()]
        
        with gzip.open(f"{RESULTS_PATH}{fname}{ind}.gzipjson", "wt") as f:
            f.write("\n".join(js_strings))


def load_communities(shape, fname="communities"):
    coms = {}
    for ind in range(2, 14):
        with gzip.open(f"{RESULTS_PATH}{fname}{ind}.gzipjson", "rt") as f:
            json_strings = f.read().split('\n')
        
        coms[ind] = np.array([readwrite.read_community_from_json_string(com) for com in json_strings]).reshape(shape)
    return coms
