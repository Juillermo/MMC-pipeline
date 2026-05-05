# core python modules
import warnings
import time
import itertools as it

# external
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import graph_tool.all as gt
import igraph as ig
import cdlib
import leidenalg as la

# internal
from data import beautify_name
from network_utils.backboning import perform_backboning


def build_graph(ABC_data, sig, thresh, backboning=False, calculate_p_value=False, names=None, remove_disconnected=True, verbose=True, gtype="gt"):
    assert gtype in ("gt", "ig"), f"Graph type '{gtype}' not understood."
    names = names if names is not None else list(np.append(np.unique(ABC_data.namei.values), ABC_data.namej.values[-1]))
    table = ABC_data if not backboning else perform_backboning(ABC_data, thresh, sig, calculate_p_value=calculate_p_value)

    g = gt.Graph(directed=False)
    g.vp.name = g.new_vp("string")
    g.vp.prev = g.new_vp("float")

    g.ep.weight = g.new_ep("float")
    g.ep.logweight = g.new_ep("float")
    g.ep.abslog_weight = g.new_ep("float")

    g.add_vertex(n=len(names))
    for v in g.get_vertices():
        g.vp.name[v] = beautify_name(names[v], short=True)

    for _, row in table.iterrows():
        if (backboning or not sig or row["a_sig"]) and (thresh is None or backboning or row["ABC"] >= thresh):
            e = g.add_edge(names.index(row["namei"]), names.index(row["namej"]))
            g.ep.weight[e] = row["ABC"]
            g.ep.logweight[e] = np.log(row["ABC"])
            g.ep.abslog_weight[e] = np.abs(np.log(row["ABC"]))
            
            g.vp.prev[e.source()] = row["Pi"]
            g.vp.prev[e.target()] = row["Pj"]

    if remove_disconnected:
        disconnected = [v for v in gt.find_vertex(g, "out", 0)]
        print("Disconnected conditions:", [g.vp.name[v] for v in disconnected]) if verbose else None
        g.remove_vertex(disconnected)
        revised_names = g.new_vp("string")
        for v in g.vertices():
            revised_names[v] = g.vp.name[v]
        g.vp.name = revised_names
        names = revised_names

    print(g) if verbose else None
    print(f"{g.num_edges()} edges out of {int((n := len(names)) * (n - 1) / 2)} possible edges.") if verbose else None
    return g if gtype == "gt" else ig.Graph.from_graph_tool(g)


def extract_blocks(state):
    """Returns a list of the same size of nodes with their assigned cluster labels."""
    if type(state) == cdlib.classes.node_clustering.NodeClustering:
        assert not state.overlap, "This formatting does not work with overlapping clusters."
        assert state.node_coverage == 1, str(state.node_coverage)
        blocks = np.NaN * np.ones(sum(len(comm) for comm in state.communities))
        for cluster_label, community in enumerate(state.communities):
            for LTC_id in community:
                blocks[LTC_id] = cluster_label
        return blocks
    elif type(state) in (la.ModularityVertexPartition, la.CPMVertexPartition):
        return state.membership
    elif (type(state) == list and type(state[0]) == frozenset) or type(state) == ig.VertexClustering:
        blocks = np.NaN * np.ones(sum(len(comm) for comm in state))
        for cluster_label, community in enumerate(state):
            for LTC_id in community:
                blocks[LTC_id] = cluster_label
        return blocks
    elif type(state) == gt.NestedBlockState:
        return state.get_bs()[0]
    else:
        return state.b.a


def extract_clusters(state, names=True):
    if type(state) == cdlib.classes.node_clustering.NodeClustering:
        clusters = {i: community for i, community in enumerate(state.communities)}
    else:
        blocks = extract_blocks(state)
        cluster_memberships = {i: el for i, el in enumerate(blocks)}
        clusters = {
            cluster_key: [state.g.vp.name[LTC] if names else LTC for LTC, clus_key in cluster_memberships.items() if
                          clus_key == cluster_key] for cluster_key in np.unique(list(cluster_memberships.values()))}
    return clusters


def detect_communities(g, n_parallel=10, state_type=gt.NestedBlockState, logweight=True, **ckwargs):
    if state_type in ("Leiden", "Leiden_CPM", "Louvain", "Walktrap", "Infomap"):
        ig_graph = ig.Graph.from_graph_tool(g)
        if logweight:
            # Leiden and Louvain don't accept negative weights
            ig_graph.es["logweight"] = [el if el >=0 else 0 for el in ig_graph.es["logweight"]]

    # Run the algorithm 'n_parallel' times and keep the best
    scores = []
    partitions = []
    t_ini = time.time()
    t_iters = []
    for i_n_init in range(n_parallel):
        if state_type in ("Leiden", "Louvain", "Leiden_CPM", "Walktrap", "Infomap"):
            weight_var = ("weight" if not logweight else "logweight") if logweight is not None else None
            if state_type in ("Leiden", "Leiden_CPM"):
                partition = la.find_partition(ig_graph, weights=weight_var, n_iterations=-1, **ckwargs,
                                              partition_type=la.CPMVertexPartition if state_type == "Leiden_CPM" else la.ModularityVertexPartition)
            elif state_type == "Louvain":
                partition = cdlib.algorithms.louvain(ig_graph, weight=weight_var)
            elif state_type == "Walktrap":
                partition = ig_graph.community_walktrap(weights=weight_var).as_clustering()
            elif state_type == "Infomap":
                partition = ig_graph.community_infomap(edge_weights=weight_var)
            else:
                raise Exception(f"Community detection algorithm '{state_type}' not understood.")

            scores.append(ig_graph.modularity(partition, weights=weight_var, directed=False))
            partitions.append(extract_blocks(partition))
            if scores[-1] == max(scores):
                max_partition = partition
            if not i_n_init % int(n_parallel / 10):
                print(i_n_init, end=",")

        else:
            t_ini_iter = time.time()
            if state_type == gt.NestedBlockState:
                state_args = dict(recs=[g.ep.log_weight], rec_types=["real-normal"])
            elif state_type == gt.ModularityState:
                assert logweight is not None, "Unweighted method not implemented."  # Albeit quite simple to implement
                state_args = {"eweight": g.ep.weight if not logweight else g.ep.log_weight}
            else:
                state_args = {}
            func = gt.minimize_nested_blockmodel_dl if state_type == gt.NestedBlockState else gt.minimize_blockmodel_dl
            state_tmp = func(g, state=state_type, state_args=state_args)

            if state_type == gt.NestedBlockState:
                L = 0
                for s in state_tmp.levels:
                    L += 1
                    if s.get_nonempty_B() == 2:
                        break
                state_tmp = state_tmp.copy(bs=state_tmp.get_bs()[:L] + [np.zeros(1)])

            mdl_tmp = state_tmp.entropy() if state_type != gt.ModularityState else -gt.modularity(g, state_tmp.b,weight=g.ep.weight)
            scores.append(mdl_tmp)
            if mdl_tmp == min(scores):
                state = state_tmp.copy()

            t_iters.append(time.time() - t_ini_iter)
            if state_type in (gt.NestedBlockState, gt.PPBlockState, gt.OverlapBlockState):
                print(
                    f"{i_n_init}, {mdl_tmp:.1f}, iter time: {t_iters[-1]:.1f}s (expected remaining {(n_parallel - i_n_init - 1) * np.mean(t_iters):.1f}s)")

    plt.hist(scores, bins=100)
    scores = np.array(scores)

    if state_type in ("Leiden", "Louvain", "Leiden_CPM", "Walktrap", "Infomap"):
        state = gt.ModularityState(g, b=g.new_vp("int", vals=extract_blocks(max_partition)), eweight=g.ep.weight)
        print(f"\nHighest modularity: {max(scores)} ({(scores == scores.max()).sum()} times)")
    else:
        if min(scores) == np.inf:
            state = state_tmp.copy()
        print(f"\nBest mdl: {min(scores)} ({(scores == scores.min()).sum()} times)")

    print(f"Time elapsed: {(time.time() - t_ini):.0f}s.")  # Time/iter ~ {(time.time() - t_ini)/n_parallel:.0f}s")
    return state, scores, partitions


def refine_clusters(state, n_iter=1000, i=0):
    dS, nattempts, nmoves = [], [], []
    print("Refining clusters solution...")
    for i in range(i, i + n_iter):
        print(i, end=", ") if not i % 100 else None
        ds, nat, nmov = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)
        dS.append(ds), nattempts.append(nat), nmoves.append(nmov)

    _, ax = plt.subplots(figsize=(10, 3))
    ax.plot(-np.array(dS));
    ax.set(yscale="log");


def create_cluster_name(list_of_names, max_length=999, n_lines=999, min_len=1, line_separator=",\n",
                        order_by_clusters=None):
    if order_by_clusters is not None:
        new_order = []
        for cluster in order_by_clusters:
            for name in list_of_names:
                if name in cluster:
                    new_order.append(name)
        list_of_names = new_order
    if len(list_of_names) <= min_len:
        return ", ".join([name[:max_length] for name in list_of_names])
    else:
        return line_separator.join([", ".join([name[:max_length] for name in line]) for line in (
            np.array_split(list_of_names, n_lines) if len(list_of_names) > n_lines else [[el] for el in
                                                                                         list_of_names])])


def compute_metrics(clustering_object: cdlib.classes.node_clustering.NodeClustering, graph):
    coms = clustering_object.communities
    metrics = {"n_clusters": (C := len(coms)),
            "size": (N := graph.vcount()),
            "edges": (E := graph.ecount()),
            "edges_per_node": 2 * E / N,
            "unclustered": np.round((1 - clustering_object.node_coverage) * N),
            "max_size": (max(len(com) for com in coms) if C > 0 else 0),}
    if clustering_object.overlap:
        clustered_LTCs = set().union(*coms)
        metrics["n_overlap"] = len([name for name in clustered_LTCs if sum(name in com for com in coms) > 1])
    return metrics


def compute_fit(stats, distance_metric=None, overlapping=True, linear=False):
    N = stats['size']
    U = stats['unclustered']
    S = stats['max_size']
    C = stats['n_clusters']
    E = stats['edges_per_node']
    fit = (N * (N - U) * (N - S) * C * (N - 1 - E)) if not linear else (N + (N - U) + (N - S) + C + (N - 1 - E))
    count = 5

    if overlapping:
        assert "n_overlap" in stats, f"'n_overlap' not in {stats.columns}."
        fit = (fit * stats['n_overlap']) if not linear else (fit + stats['n_overlap'])
        count += 1

    if distance_metric is not None:
        assert distance_metric in stats, f"'{distance_metric}' not in {stats.columns}."
        # TODO: This should actually be 'N_max', i.e. number of conditions in the dataset for that sex
        fit = (fit * stats[distance_metric] * N) if not linear else (fit + stats[distance_metric] * N)
        count += 1

    stats['fit'] = (fit ** (1/count)) if not linear else (fit / count)


def plot_overlapping_clusters(ABC_data, communities, bnames, sig=True, backboning=True, thresh=1.64, LTC=None, hops=None, ego_clusters=False, **plot_kwargs):
    """
    Plot overlapping clusters highlighting the LTCs provided.
    """
    if len(communities) > 20 and 'cmap' not in plot_kwargs:
        warnings.simplefilter("default")
        warnings.warn(f"{len(communities)} communities present but the palette only has 20 colours.")
        warnings.simplefilter("ignore")
    
    gt_graph = build_graph(ABC_data=ABC_data, sig=sig, backboning=backboning, thresh=thresh, verbose=False)
    ig_graph = ig.Graph.from_graph_tool(gt_graph)
    state = gt.ModularityState(gt_graph)
    
    if LTC is not None:
        LTCs = [LTC] if type(LTC) != list else LTC
        for i, LTCi in enumerate(LTCs):
            if LTCi not in bnames:
                assert beautify_name(LTCi, short=True) in bnames, f"'{LTCi}' LTC not understood."
                LTCs[i] = beautify_name(LTCi, short=True)
    else:
        LTCs = None
    
    if LTCs is None or any(LTCi in ig_graph.vs["name"] for LTCi in LTCs):
        overlaps = pd.Series(index=bnames, data=[sum(name in com for com in communities) for name in bnames])
        pmarginals = [[1/overlaps[v['name']] if v['name'] in com else 0.0 for com in communities] for v in ig_graph.vs]
        plkwargs = dict(state=state, LTC=LTCs, hops=hops, communities=None if not ego_clusters else communities,
                        pmarginals=gt_graph.new_vp('vector<double>', vals=pmarginals), edge_colour=False, **plot_kwargs)
        return plot_ego(**plkwargs)


def plot_ego(state, LTC=None, communities=None, hops=None, **kwargs):
    """
    Plot subnetwork containing the condition and the neighbouring nodes.
    """
    if LTC is not None:
        LTCs = [LTC] if type(LTC) != list else LTC
        state = state.copy()
        selected = state.g.new_vp("bool")  # all nodes in the subnetwork
        core = state.g.new_vp("bool")  # main nodes
        assert hops in (1, 2) or hops is None, str(hops)

        if communities is not None:
            ego_clusters_idx = [i for i, com in enumerate(communities) if any(LTC in com for LTC in LTCs)]

        for v in state.g.vertices():
            if state.g.vp['name'][v] in LTCs:
                selected[v] = True  # Mark node
                core[v] = True
                if hops is not None:
                    for n in v.all_neighbors():  # Include neighbors
                        selected[n] = True
                        if hops == 2:
                            for m in n.all_neighbors():  # Include neighbors of neighbors
                                selected[m] = True

            if communities is not None and any([state.g.vp['name'][v] in communities[cluster_ind] for cluster_ind in ego_clusters_idx]):
                selected[v] = True  # Mark nodes in the same communities
        
        if hops is not None or communities is not None:
            # N = state.g.num_vertices()
            state.g = gt.GraphView(state.g, vfilt=selected)
            # print(f"Subnetwork with {state.g.num_vertices()} nodes out of {N}.")
            if len(ego_clusters_idx) == 0 and hops is None:
                print("No communities found for the selected LTCs.")
                return None

        # Highlight the core nodes
        vertex_color = state.g.new_vp('vector<double>')
        vertex_pen_width = state.g.new_vp('float')
        for v in state.g.vertices():
            if core[v]:
                vertex_color[v] = [1., 0, 0, 1.]  # RGBA for red
                vertex_pen_width[v] = 2.5       # Thicker border for selected
            else:
                vertex_color[v] = [0., 0., 0., 0.]  # Transparent border for others
                vertex_pen_width[v] = 1.0             # Normal border width
        kwargs = {**kwargs,
            "vertex_color": vertex_color,
            "vertex_pen_width": vertex_pen_width,
            "vertex_halo": core,
            "vertex_halo_color": [1, 1, 0, .3],  # RGBA for yellow
            "vertex_halo_size": 1.5,}
    return plot_clustering(state, **kwargs)


def plot_cluster(cluster_ind, state, **kwargs):
    """
    Plot subnetwork containing the cluster and the neighbouring nodes.
    """
    blocks = extract_blocks(state)
    n_blocks = max(blocks) + 1
    assert n_blocks > cluster_ind

    # Colour the desired cluster with the special pink colour
    colours = [(234, 67, 136), (181, 26, 21), (204, 105, 4), (207, 170, 39), (88, 138, 147), (62, 62, 240), (149, 107, 250)]
    assert len(colours) >= n_blocks - 1
    colours = colours[:n_blocks - 1]
    colours.insert(cluster_ind, (248, 132, 247))
    cmap = matplotlib.colors.ListedColormap(colours)

    vertex_fill_color = state.g.new_vp('vector<double>')
    for v in state.g.vertices():
        vertex_fill_color[v] = np.array(cmap(blocks[int(v)] / max(blocks))[:3]) / 255  # drop alpha channel, if present

    # Create a subgraph with the desired cluster and its neighbours
    state = state.copy()
    selected = state.g.new_vp("bool")
    for v in state.g.vertices():
        if blocks[int(v)] == cluster_ind:
            selected[v] = True  # Mark node
            for n in v.all_neighbors():  # Include neighbors
                selected[n] = True
    state.g = gt.GraphView(state.g, vfilt=selected)

    # Plot the subgraph
    return plot_clustering(state, vertex_fill_color=vertex_fill_color, **kwargs)


def plot_clustering(state, sfdp=True, pos=None, edge_colour=True, pmarginals=None, b=None, vertex_font_size=10, edge_width=4, vertex_max_size=40,
                    output_size=(800, 800), cmap=None, alpha=1.0,
                    node_names=None, vertex_text_position=-2, vertex_fill_color=None, vertex_size=None, **kwargs):
    state = state.copy()
    if b is not None:
        state.b = b
    if sfdp:
        if type(state) == gt.NestedBlockState:
            state = state.get_levels()[0]
        # Filter out nodes with no edges
        state.g = gt.GraphView(state.g, vfilt=lambda v: v.in_degree() > 0 or v.out_degree() > 0)

    # Node text
    kwargs = {"vertex_text_position": vertex_text_position if type(state) != gt.NestedBlockState else "centered",
              ## `-1`: node size to fit text; *positive*: radians, outside;  *negative*: inside; "centered", outside and rotated
              "vertex_font_size": vertex_font_size, "vertex_text_color": "k", **kwargs}
    if node_names is None:
        kwargs["vertex_text"] = state.g.vp.name if type(state) != gt.OverlapBlockState else ""
    else:
        if node_names == "word2line":
            node_names = [name.replace(" ", "\n") for name in state.g.vp.name]
        kwargs["vertex_text"] = state.g.new_vp("string", vals=node_names)

    if pmarginals is not None:
        kwargs["vertex_shape"] = "pie"
        kwargs["vertex_pie_fractions"] = pmarginals

    # Colour nodes
    cmap = plt.get_cmap('tab20' if cmap is None else cmap)  # Set2')  # Set1')
    if pmarginals is None:
        if vertex_fill_color is None:
            blocks = extract_blocks(state)
            norm = plt.Normalize(vmin=0, vmax=blocks.max())
            vertex_fill_color = state.g.new_vp('vector<double>')
            for v in state.g.vertices():
                colour = cmap(norm(blocks[int(v)]))  # [:3]  # Exclude the alpha channel as it's not used by default
                colour = list(colour)
                colour[-1] = alpha
                vertex_fill_color[v] = colour
        kwargs["vertex_fill_color"] = vertex_fill_color
    else:
        n_colours = len(pmarginals[0])
        norm = plt.Normalize(vmin=0, vmax=n_colours-1)  # 19)
        colours = []
        for v in range(n_colours):
            colour = cmap(norm(v))
            colour = list(colour)
            colour[-1] = alpha
            colours.append(colour)
        kwargs["vertex_pie_colors"] = colours
    if "vertex_color" not in kwargs:
        kwargs["vertex_color"] = [0., 0., 0., 0.]  # For the border of the nodes
    if "vertex_pen_width" not in kwargs:
        kwargs["vertex_pen_width"] = 1

    # Node position
    if pos is not None:
        kwargs["pos"] = pos
    elif sfdp:
        kwargs["pos"] = gt.sfdp_layout(state.g, eweight=state.g.ep.weight)  # , kc=1),}
        # pos=gt.fruchterman_reingold_layout(state.g, weight=state.g.ep.weight, circular=True),
        # pos=gt.arf_layout(state.g, weight=state.g.ep.weight),

    # Node size
    assert vertex_size is None or type(vertex_size) == str, f"Size parameter '{vertex_size}' not understood."
    if type(vertex_size) == str:
        if vertex_size in ("edge_linear", "edge_log"):
            sizes = np.array([max(state.g.ep.weight[e] for e in v.all_edges()) if len(
                state.g.get_all_edges(v)) > 0 else 0 for v in state.g.vertices()])

            if vertex_size == "edge_log":
                sizes[np.isclose(sizes, 0)] = sizes[sizes > 0].min() / 100
                sizes = np.log(sizes)
                sizes += sizes.min()

        elif vertex_size == "degree_linear":
            sizes = state.g.get_all_degrees(state.g.get_vertices(), eweight=state.g.ep.weight)

        elif vertex_size == "prevalence":
            sizes = np.log(state.g.vp.prev.a)
            sizes -= sizes.min() - 1  # To offset to positive values with a minimum size of 1/(max(log(prev)))
        else:
            raise Exception(f"Size parameter '{vertex_size}' not understood.")

        # TEXT_FACTOR = 0.2
        ss = state.g.new_vp("float", vals=(sizes / sizes.max()) * vertex_max_size)
        kwargs.update({"vertex_size": ss,})  # "font_size": 10})  # state.g.new_vp("float", vals=ss.a * TEXT_FACTOR)})

    # Edges
    if type(state) != gt.OverlapBlockState:
        # edge_pen_width = gt.prop_to_size(state.g.ep.abslog_weight, 0, 4, power=1, log=True)
        # gt.prop_to_size(state.g.ep.abslog_weight, 0, 4, power=1, log=True),  # gt.prop_to_size(g.ep.abslog_weight, 0, 4, power=1, log=False),
        if edge_width is not None:
            edge_pen_width = state.g.ep.abslog_weight.copy()
            vals = edge_pen_width.fa  # vals = np.log(prop.fa)
            delta = vals.max() - 0  # vals.min()
            edge_pen_width.fa = 0 + (edge_width - 0) * ((vals - 0) / delta) ** 1 
            kwargs["edge_pen_width"] = edge_pen_width

        kwargs["edge_gradient"] = []  # This prevents edges to be coloured by node membership
        if edge_colour:
            kwargs = {**kwargs,
                "edge_color": state.g.ep.weight,  # gt.prop_to_size(g.ep.weight, power=1, log=True)
                "eorder": state.g.ep.abslog_weight,  # g.ep.abslog_weight,
                "ecmap": (matplotlib.cm.inferno, 1),  # (matplotlib.cm.inferno, .6)
                "ecnorm": matplotlib.colors.SymLogNorm(linthresh=1.0)}

    state.draw(output_size=output_size, **kwargs);  # output="moreno-train-wsbm.pdf")
    return kwargs


def perform_clustering_experiments(ABC_data, sig=False, thresh=None, backboning=False, calculate_p_value=False,
                                   n_parallel=10, state_type=gt.NestedBlockState, logweight=True, sfdp=False,
                                   find_consensus=False, plot_results=True, **ckwargs):
    """
    If 'find_consensus' is True, then the partition is the one from the maximum marginals instead of the maximum
    partition.
    :param ABC_data:
    :param sig:
    :param thresh:
    :param backboning:
    :param calculate_p_value:
    :param n_parallel:
    :param state_type:
    :param logweight:
    :param sfdp: 
    :param find_consensus:
    :return:
    """
    g = build_graph(ABC_data, sig, thresh, backboning, calculate_p_value=calculate_p_value)

    state, _, partitions = detect_communities(g, state_type=state_type, logweight=logweight, n_parallel=n_parallel, **ckwargs)
    if state_type in (gt.NestedBlockState, gt.PPBlockState):
        refine_clusters(state)

    if find_consensus:
        pmode = gt.PartitionModeState(
            partitions[::int(len(partitions) / 10000)] if len(partitions) > 100000 else partitions, converge=True)
        state.b = pmode.get_max(g)
    if plot_results:
        plot_clustering(state, sfdp=sfdp, pmarginals=pmode.get_marginal(g) if find_consensus else None, edge_colour=False if logweight else True)

        kwargs = dict(boundaries=[el for el in (1, 2, 4, 8, 16) if el >= thresh],
                    sym=False) if thresh is not None and not backboning else {}
        clusters, clusters_mat = analyse_clusters(state, ABC_data, sig=sig, thresh=thresh, **kwargs)
    else:
        clusters, clusters_mat = None, None

    return state, clusters, clusters_mat, partitions if not find_consensus else pmode


def perform_overlapping_clustering(ABC_data=None, pars={}, cdfunc=cdlib.algorithms.dpclus, backboning=True, sig=True, thresh=1.64, verbose=True,
                                   graph=None, return_clustering_object=False, **cdkwargs):
    if graph is None:
        assert ABC_data is not None, "Graph not provided and no ABC data to build it."
        buildg_kwargs = dict(ABC_data=ABC_data, sig=sig, backboning=backboning, verbose=verbose, gtype="ig",
                            thresh=pars.get("thresh", thresh) if backboning else None)
        graph = build_graph(**buildg_kwargs)
    coms = cdfunc(g_original=graph, **{par: val for par, val in pars.items() if par != "thresh"}, **cdkwargs)
    return graph, coms.communities if not return_clustering_object else coms


def dpclus():  # just to comment and analyse the code, implementation is done by cdlib (https://github.com/GiulioRossetti/cdlib/blob/master/cdlib/algorithms/internal/DPCLUS.py#L16)
    data = defaultdict(set)  # node id => neighboring node ids
    unvisited = set(data)
    
    while unvisited:  # go through all nodes
        seed = max(unvisited, key=lambda k: (len(data[k]&unvisited),node_index[k]))  # get highest degree node of those not visited, where degree also accounts for UNVISITED neighbours
        frontier = data[seed] & unvisited  # frontier is the neighbours of the seed node that also are unvisited
        if not frontier: break  # no connections left to analyze

        # Re-compute all weights between UNVISITED nodes
        edges,weights = defaultdict(zerodict), defaultdict(int)
        for a,b in combinations(unvisited, 2):
            if b not in data[a]: continue  # they need to be neighbours themselves
            shared = len(data[a] & data[b] & unvisited)  # n of common neighbours
            edges[a][b],edges[b][a] = shared, shared
            weights[a] += shared
            weights[b] += shared

        max_w,_,node = max((w,node_index[n],n) for n,w in weights.iteritems())  # node with max weight is picked (so the code above is the 'else', but it was there to terminate the algorithm if the highest degree is zero)
        if max_w > 0:
            seed = node
            frontier = data[seed] & unvisited  # frontier is the neighbours of the seed node that also are unvisited

        cluster = set((seed,))
        cluster_degrees = {seed: 0}
        nn,ne = 1, 0  # number of nodes, edges in cluster


    while frontier:  # frontier is the nodes to check to add to the cluster next, they start with the unvisited neighbours of the seed node and then expand
        # find higest priority node:
        e_nk,_,_,p = max((len(data[n] & cluster),  # 0. number of edges between node and cluster nodes (e_nk)
                          sum(edges[n][c] for c in cluster),  # 1. sum of edge weights between node and cluster nodes
                          node_index[n],  # 2. the node's index
                          n,  # 3. the node itself (what's the difference of this and the previous one?
                         ) for n in frontier)

        density = 2. * (ne + e_nk) / (nn * (nn+1))
        if density < D_THRESHOLD: break  # adding the node gives too low density; cluster is finished

        # if all nodes only have ONE connecting edge in cluster, use "fine-tuning" (not relevant when d_threshold=1?)
        # this orders by the number of neighbors in the frontier, minus the connectedness of the attached (cluster) node within the cluster
        cp = CP_THRESHOLD
        if e_nk == 1 and len(cluster) > 1:  # this excludes the case of adding the second node to the cluster (first beyond the seed)
            n_degree = dict() # node => fine-tuning parameter
            for n in frontier:
                for c in cluster:
                    if n in edges[c]: break  # find adjacent cluster node
                n_degree[n] = len(data[n] & frontier) - cluster_degrees[c]
            p = max(frontier, key=lambda k: (n_degree[k],node_index[k]))
            if n_degree[p] > 0:
                cp /= 2.  # "when fine-tuning is used [...], we use half the value of cpin for periphery checking and thus help to form some sparse clusters
        
        
        if (e_nk / density / (nn+1)) < cp: break  # no good node found; cluster is finished

      
        # otherwise, add the node to the cluster
        cluster.add(p)
        nn,ne = (nn+1), (ne+e_nk)
      
        cluster_degrees[p] = e_nk
        for n in data[p] & cluster:
            cluster_degrees[n] += 1

        frontier = set.union(*((data[n] - cluster) & unvisited for n in cluster))  # frontier is updated with all unvisited nodes neighbouring the cluster

    # Overlapping nodes are added AFTER all unvisited nodes are first exhausted
    frontier_nodes = (set.union(*(data[c] for c in cluster)) - cluster)  # The frontier now includes all neighbours of cluster, regardless of whether or not they've been visited
    frontier = sorted([len(data[n] & cluster),
                       sum(edges[n][c] for c in cluster),
                       0,  # frontier[2] stores the fine-tuning parameter
                       node_index[n], 
                       n]
      for n in frontier_nodes)
    
    # "fine-tuning"
    fine_tuning = False
    if frontier and frontier[-1][0] == 1:  # There are nodes adjacent to the cluster and not belonging to the cluster AND the node with the largest number of connections to the cluster only has ONE connection to it
        # compute fine-tuning parameter for each node in the frontier
        for n in frontier:
            # find adjacent cluster node
            for c in cluster:
                if n[4] in edges[c]: break
                
            # compute fine-tuning parameter
            n[2] = len(data[n[4]] & frontier_nodes) - cluster_degrees[c]
            
        # sort frontier by fine-tuning parameter
        frontier.sort(key=lambda k: (k[2],k[3]))
        fine_tuning = True
    
    # iterate through visited neighbors and update accordingly
    while frontier:
        e_nk,_,w,_,p = frontier.pop()
        cp = CP_THRESHOLD  / 2. if fine_tuning and w > 0 else CP_THRESHOLD  # "when fine-tuning is used [...], we use half the value of cpin for periphery checking and thus help to form some sparse clusters

        density = 2. * (ne + e_nk) / (nn * (nn+1))
        if density < D_THRESHOLD or (e_nk / density / (nn+1)) < cp: continue

        # add node to the cluster
        cluster.add(p)
        nn,ne = (nn+1), (ne+e_nk)

        cluster_degrees[p] = e_nk
        for n in data[p] & cluster:
            cluster_degrees[n] += 1

        # update E_nk for the other nodes on the frontier
        for n in frontier:
            if p in edges[n[4]]:
                n[0] += 1

    unvisited -= cluster

    num_clusters += 1