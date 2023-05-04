import glob
import itertools
import json
import logging
import os
import time
import warnings
from collections import OrderedDict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brian2 import Hz, PopulationRateMonitor, SpikeMonitor, StateMonitor, nS, pA, second

import core.sim as sim
from style.figure import plot_save
from utils.hashable import hashable

logger = logging.getLogger(__name__)
time_unit = second


class LRDFigure(object):
    """Parent Class for figures.
    A figure consists of multiple plots with data from runs.
    The 'run' method may be "polymorphed" and called multiple times.
    Data needs to be stored after each run as new runs overwrite variables.
    """

    fig_name = "lrd_figure"

    def __init__(self, **kwargs):
        self.setup_kwargs = None
        self.results = None
        self.s_labels = []
        self.s_labels_dict = {}
        self.setup(**kwargs)

    def setup(self, **kwargs):
        self.setup_kwargs = kwargs
        self.setup_kwargs.setdefault("__device_directory", f".cpp_{self.fig_name}")
        return self

    def run(self, cache=True, **kwargs):
        """Basic `single_run` simulation and followed by assignment of results to class object variables"""
        kwargs = {**self.setup_kwargs, **kwargs}
        self.sim_name = sim.get_sim_name(**kwargs)
        if cache and self.load_variables():
            self._internalise_objects()
            return self
        network, results = sim.single_run(**kwargs)
        self._internalise_objects(results)
        self.results = results
        if cache:
            if type(cache) is dict:
                self.save_vars(**cache)
            else:
                self.save_vars()
        return self

    def save_vars(self, path=None, monitors=True):
        if path is None:
            path = os.path.join("temp", self.sim_name + ".h5")
        from brian2 import CodeRunner, VariableOwner

        self.var_df = pd.Series()
        print("\n")
        to_save = {}

        def __value_recorded(__value, __monitors) -> bool:
            if type(__monitors) is bool and __monitors:
                return True
            if type(__monitors) is dict:
                if "synapse_mon" in __value:
                    __monitor_key = "synapse_mon"
                elif "mg2_mon" in __value:
                    __monitor_key = "Mg2"
                else:
                    __monitor_key = __value
                return __monitor_key in __monitors and __monitors[__monitor_key]

        for var_name, value in self.results.items():
            if (
                not np.iterable(value)
                and not isinstance(value, VariableOwner)
                and not isinstance(value, CodeRunner)
            ):
                logger.debug(f"{var_name} = {value}")
                self.var_df.loc[var_name] = value
            elif isinstance(value, StateMonitor) and __value_recorded(
                var_name, monitors
            ):
                states = value.get_states()
                keys = np.array(list(states.keys()))
                keys = keys[keys != "N"]
                nrn_ind = value.record
                columns = pd.MultiIndex.from_product([keys, nrn_ind])
                tmp = pd.DataFrame(columns=columns, index=states["t"])
                for key in keys:
                    try:
                        tmp[key] = states[key]
                    except Exception as err:
                        logger.error(f"{var_name} -> {key} \t {err}")
                        logger.debug(f"shape = {states[key].shape}")
                        logger.debug(f"states[key] = {states[key]}")
                        logger.debug(f"value = {value}")
                tmp.to_hdf(path, key=var_name, complevel=7, complib="blosc:zstd")
            elif isinstance(value, SpikeMonitor) and __value_recorded(
                var_name, monitors
            ):
                spkmon = SpkMon(
                    **{k: v for k, v in value.get_states().items() if k != "N"}
                )
                # to_save[var_name + "_SpkMon"] = spkmon
                spkmon.save(path.replace(".h5", f"{var_name}_SpkMon.npz"))
            elif isinstance(value, PopulationRateMonitor) and __value_recorded(
                var_name, monitors
            ):
                states = value.get_states()
                keys = np.array(list(states.keys()))
                keys = keys[keys != "N"]
                tmp = pd.DataFrame(columns=keys, index=states["t"])
                from style.plot_trace import SMOOTH_RATE_ARGS

                tmp["smooth_rate"] = value.smooth_rate(**SMOOTH_RATE_ARGS)
                for key in keys:
                    try:
                        tmp[key] = states[key]
                    except Exception as err:
                        logger.error(f"{var_name} -> {key} \t {err}")
                        logger.debug(f"shape = {states[key].shape}")
                        logger.debug(f"states[key] = {states[key]}")
                        logger.debug(f"value = {value}")
                tmp.to_hdf(path, key=var_name, complevel=7, complib="blosc:zstd")
            elif np.iterable(value) and not isinstance(value, VariableOwner):
                if isinstance(value, dict):
                    if isinstance(list(value.values())[0], VariableOwner):
                        # dict contains references to VariableOwner objects
                        continue
                    var_name += "_dict"
                elif isinstance(value[0], VariableOwner):
                    # iterable object (e.g. list) contains VariableOwner objects
                    continue
                to_save[var_name] = value
        numpy_path = path.replace(".h5", ".npz")

        np.savez_compressed(numpy_path, **to_save)

        self.var_df.to_hdf(path, key="var_df", complevel=7, complib="blosc:zstd")
        logger.debug(f"saved {to_save.keys()} to {numpy_path}")

    def load_variables(self, path=None, save_dest=None):
        if path is None:
            path = os.path.join("temp", self.sim_name + ".h5")
        if not os.path.exists(path):
            return False
        if save_dest is None:
            # save to object itself
            save_dest = self.__dict__
        keys = pd.HDFStore(path, "r").keys()
        for key in keys:
            df: pd.DataFrame = pd.read_hdf(path, key)
            clean_key = key.replace("/", "")
            df.name = clean_key
            if key == "/df":
                continue
            elif key == "/var_df":
                for name, value in df.iteritems():
                    save_dest[name] = value
            else:
                if "smooth_rate" in df.columns:
                    vals = df.smooth_rate.values
                    del df["smooth_rate"]
                    df.smooth_rate = lambda width, window: vals
                save_dest[clean_key] = df
            logger.debug(f"{key} loaded from cache {path}")
        if os.path.exists(path.replace(".h5", ".npz")):
            for key, value in np.load(
                path.replace(".h5", ".npz"), allow_pickle=True
            ).items():
                if key.endswith("_dict"):
                    save_dest[key.replace("_dict", "")] = value.item()
                elif key.endswith("_SpkMon"):
                    save_dest[key.replace("_SpkMon", "")] = SpkMon(*value)
                else:
                    save_dest[key] = value
            
            # see if a file ends with _SpkMon.npz
            files = glob.glob(path.replace(".h5", "*_SpkMon.npz"))
            for file in files:
                key = os.path.basename(file).replace("_SpkMon.npz", "")
                save_dest[key] = SpkMon.load(file)
            
        return True

    # noinspection PyUnresolvedReferences
    def _internalise_objects(self, results=None):
        """Take results and make each (key,value)s part of the object (i.e. accessible via `self.key`"""
        if results is not None:
            for key in results.keys():
                self.__dict__[key] = results[key]
        if "synapse_mon" in self.__dict__["__monitors"]:
            self.synapse_monitors = [
                self.synapse_mon_cee,
                self.synapse_mon_cie,
                self.synapse_mon_cei,
                self.synapse_mon_cii,
            ]
            self._create_helper_labels()

    def _create_helper_labels(self):
        for s_i, synapse_mon in enumerate(self.synapse_monitors):
            if synapse_mon is None:
                return
            s_name = (
                synapse_mon.name
                if type(synapse_mon) is pd.DataFrame
                else synapse_mon.source.name
            )
            s_label = "${}$".format(
                s_name.replace("_", "").replace("C", "C_{", 1) + "}"
            )
            self.s_labels.append(s_label)
            self.s_labels_dict[s_name] = s_label

    def plot(self, run=False, **run_kwargs):
        """Plots results. If no results are in the object, call self.run()"""
        if self.results is None or run:
            self.run(**run_kwargs)
        return self

    def save_figure(
        self,
        file_formats=("pdf", "jpg"),
        use_args=False,
        figs=None,
        close=True,
        **kwargs,
    ):
        if self.results is not None:
            logger.info("saving figures")
            figs = figs or [self.fig]
            save_time_0 = time.time()
            if use_args:
                file_name = self.sim_name.replace(".", "pt").replace(" * ", "")
            else:
                file_name = self.fig_name.replace(".", "pt").replace(" * ", "")
            plot_save(
                [f"output/{file_name}.{file_format}" for file_format in file_formats],
                figs=figs,
                close=False,
                **kwargs,
            )
            if close:
                for fig in figs:
                    plt.close(fig)
            save_time = time.time()
            save_dt = save_time - save_time_0
            logger.info(f"took {save_dt:.2f}s to save")

    def create_df(self, variables, pops=("E", "I", "all"), subsample=1):
        from brian2 import Quantity, ms

        d = {}
        p = {}
        smooth_rate_args = dict(window="flat", width=10.1 * ms)
        if "E" in pops:
            # extract recorded EGABA values for exc and inh populations
            p["E"] = list(filter(lambda x: x < self.N_E, self.state_mon.record))
        if "I" in pops:
            p["I"] = list(filter(lambda x: x >= self.N_E, self.state_mon.record))
        p["all"] = self.state_mon.record
        for pop in pops:
            for var in variables:
                if type(var) is list or type(var) is tuple:
                    var, var_name = var
                    var_name = f"{var_name}_{pop}"
                elif type(var) is dict:
                    var, var_name = list(var.items())[0]
                    var_name = f"{var_name}_{pop}"
                elif type(variables) is dict:
                    var_name = variables[var]
                    var_name = f"{var_name}_{pop}"
                else:
                    var_name = f"{var}_{pop}"
                if var == "r":
                    arr = getattr(self, f"r_{pop}").smooth_rate(**smooth_rate_args)[
                        ::subsample
                    ]
                else:
                    arr = getattr(self.state_mon[p[pop]], var).mean(axis=0)[::subsample]
                if isinstance(arr, Quantity):
                    if var.startswith("I"):
                        arr /= pA
                    elif var.startswith("g"):
                        arr /= nS
                    else:
                        arr /= arr.get_best_unit()

                d[var_name] = arr

        _df = pd.DataFrame(
            d, index=(self.r_all.get_states()["t"] / time_unit)[::subsample]
        )
        _df.sort_index(axis="columns", inplace=True)
        return _df


@hashable
class MultiRunFigure(LRDFigure):
    """For figures that rely on running sim multiple times, different caching strategies are required"""

    monitors = {
        "sp_all": False,
        "r_E": True,
        "r_I": True,
        "r_all": True,
        "state_mon": ["E_GABA", "g_GABA", "I_GABA_rec"],
        "synapse_mon": False,
    }

    # save memory by dropping unused columns, even if the values were recorded
    ignore = ["g_GABA_all"]

    def __init__(
        self, product_map: OrderedDict, seeds=(None,), default_params=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.seeds = seeds
        self.var_idx = {}
        self.var_names = []
        self.iterables = []
        self.default_params = default_params or {}
        if type(product_map) is not OrderedDict:
            # dicts do not guarantee ordering so sort by key for OrderedDict
            product_map = OrderedDict(sorted(product_map.items(), key=lambda x: x[0]))
        self.product_map = product_map
        self.df: pd.DataFrame = None

        for var_name, var_dict in product_map.items():
            if type(var_dict) is str:
                # use same range+title as another key without increasing the number of combinations
                self.var_idx[var_name] = self.var_names.index(var_dict)
            else:
                var_range = var_dict["range"] if "range" in var_dict else var_dict
                self.iterables.append(var_range)
                var_text = var_dict["title"] if "title" in var_dict else var_name
                self.var_names.append(var_text)
                self.var_idx[var_name] = len(self.iterables) - 1
                # convert range to tuple for caching
                product_map[var_name] = {"range": tuple(var_range), "title": var_text}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{json.dumps(dict(self.product_map), sort_keys=True)}, "
            f"seeds={self.seeds})"
        )

    def run(self, subsample=10, save_vars=False, use_vaex=False, **kwargs):
        full_hash = self.hash_extra(
            f"subsample={subsample}, " f"{json.dumps(kwargs, sort_keys=True)}"
        )
        fname = os.path.join("temp", f"{full_hash}.h5")
        vaex_multi_fname = os.path.join("temp", full_hash, f"{full_hash}_*.hdf5")
        vaex_fname = fname.replace(".h5", ".hdf5")

        seeds = self.seeds
        run_idxs = list(range(len(seeds)))
        self.save_dests = save_dests = {}
        sim_files = {}
        for vars_vals in itertools.product(*self.iterables):
            for i, run_seed in enumerate(seeds):
                full_kwargs = {
                    **self.default_params,
                    **dict(__monitors=self.monitors, run_seed=run_seed),
                    **kwargs,
                }
                for var, idx in self.var_idx.items():
                    full_kwargs[var] = vars_vals[idx]
                run_key = (*vars_vals, i)
                self.sim_name = sim.get_sim_name(**full_kwargs)
                try:
                    os.makedirs("temp")
                except IOError:
                    pass
                file_name = os.path.join("temp", self.sim_name + ".h5")
                logger.debug(f"{self.sim_name}")
                if not os.path.exists(file_name):
                    if save_vars:
                        variable_save_monitors = {
                            "monitors": {"sp_all": "sp_all" in self.monitors}
                        }
                    else:
                        variable_save_monitors = False
                    super().run(cache=variable_save_monitors, **full_kwargs)
                    _df = self.create_df(subsample=subsample)
                    _df.to_hdf(file_name, key="df", complevel=7, complib="blosc:zstd")
                    logger.debug("\t dataframe saved to cache")
                sim_files[run_key] = self.sim_name
        logger.debug("loading from cache")
        if save_vars:
            for run_key, sim_name in sim_files.items():
                file_name = os.path.join("temp", sim_name + ".h5")
                save_dests.setdefault(run_key, {})
                self.load_variables(file_name, save_dest=save_dests[run_key])
        if os.path.isfile(vaex_fname) and use_vaex:
            import vaex  # noqa

            self.results = self.df = vaex.open(vaex_fname)
            logger.info(f"loaded {self.df.column_names} from cache {vaex_fname}")
            return self
        if os.path.isfile(fname):
            self.results = self.df = pd.read_hdf(fname, key="df")
            logger.info(f"loaded {self.df.columns} from cache {fname}")
            return self

        if use_vaex:
            try:
                os.makedirs(os.path.split(vaex_multi_fname)[0])
            except OSError:
                pass

        for run_key, sim_name in sim_files.items():
            file_name = os.path.join("temp", sim_name + ".h5")
            logger.debug(f"{sim_name}")
            vaex_file_name = vaex_multi_fname.replace("*", f"{run_key}")
            try:
                _df = pd.read_hdf(file_name, key="df")
            except (KeyError, FileNotFoundError):
                _df = pd.read_hdf(file_name.replace(".h5", ""), key="df")

            if "duration" in kwargs:
                from brian2 import defaultclock

                if kwargs["duration"] * time_unit / defaultclock.dt == _df.shape[0]:
                    _df = _df.iloc[::subsample]
            _df.drop(labels=self.ignore, axis=1, inplace=True)
            _df.index = index = pd.Index(np.round(_df.index, 5), name="Time")

            logger.debug("\t loaded from cache")
            if use_vaex:
                import vaex

                # dataframe converted to long-form as vaex doesn't support multiindex
                vaex.from_pandas(
                    pd.DataFrame(
                        dict(
                            zip(
                                [*_df.columns, *self.var_names, "run_idx"],
                                [*_df.values.T, *run_key],
                            )
                        ),
                        index=index,
                    ).reset_index(),
                    name=str(run_key),
                ).export(vaex_file_name)
                logger.debug("\t converted to vaex")
            else:
                if self.df is None:
                    with warnings.catch_warnings():
                        from tables import NaturalNameWarning, PerformanceWarning

                        warnings.simplefilter("ignore", NaturalNameWarning)
                        warnings.simplefilter("ignore", PerformanceWarning)
                        columns = pd.MultiIndex.from_product(
                            [*self.iterables, run_idxs, _df.columns],
                            names=[*self.var_names, "run_idx", "var"],
                        )
                        self.df = pd.DataFrame(columns=columns, index=_df.index)
                self.df[run_key] = _df
                logger.debug("\t assigned to DataFrame")

        if use_vaex:
            logger.debug("loading vaex DataFrames and converting to single file")
            self.df = vaex.open(vaex_multi_fname, convert=vaex_fname)
            import shutil

            try:
                shutil.rmtree(os.path.split(vaex_multi_fname)[0])
            except OSError as err:
                logger.error(
                    f"Error occured while deleteing {os.path.split(vaex_multi_fname)[0]}, please dleete "
                    f"manually. Error: \n {err}"
                )
        else:
            logger.debug("sorting")
            self.df.sort_index(
                axis="columns", inplace=True
            )  # speed up access (unsorted is O(n), but sorted is O(1))
            # save complete df
            logger.debug(f"saving {self.df.columns}")
            self.df.to_hdf(fname, key="df")
            logger.debug("saved")
        self.results = self.df
        return self

    def create_df(self, subsample=1):
        from brian2 import Quantity, ms

        def with_best_unit(_arr):
            if isinstance(_arr, Quantity):
                if var.startswith("I"):
                    unit = pA
                elif var.startswith("g"):
                    unit = nS
                elif var.startswith("r"):
                    unit = Hz
                else:
                    unit = _arr.get_best_unit()
                _arr = _arr / unit
            return _arr

        df_builder = {}
        pop_idxs = {}
        smooth_rate_args = dict(window="flat", width=10.1 * ms)
        # extract recorded values for exc and inh populations
        pop_idxs["E"] = list(filter(lambda x: x < self.N_E, self.state_mon.record))
        pop_idxs["I"] = list(filter(lambda x: x >= self.N_E, self.state_mon.record))
        pop_idxs["all"] = self.state_mon.record

        for var, value in self.monitors.items():
            if value:
                var_name = var
                if var.startswith("r"):
                    arr = getattr(self, var).smooth_rate(**smooth_rate_args)[
                        ::subsample
                    ]
                    df_builder[var_name] = with_best_unit(arr)
                elif var == "state_mon":
                    for pop, idxs in pop_idxs.items():
                        if len(idxs):
                            for state_var in value:
                                arr = getattr(self.state_mon[idxs], state_var).mean(
                                    axis=0
                                )[::subsample]
                                if state_var == "I_GABA_rec":
                                    state_var = "I_GABA"
                                df_builder[f"{state_var}_{pop}"] = with_best_unit(arr)
                elif var == "synapse_mon":
                    for conn in ["cee", "cie", "cei", "cii"]:
                        pop = f"{var}_{conn}"
                        monitor = getattr(self, pop)
                        for state_var in value:
                            arr = getattr(monitor, state_var).mean(axis=0)[::subsample]
                            df_builder[f"{state_var}_{pop}"] = with_best_unit(arr)
                elif var == "Mg2":
                    arr = getattr(self, "mg2_mon").mean(axis=0)[::subsample]
                    df_builder["Mg2"] = with_best_unit(arr)
                elif type(value) is bool and value:
                    obj = getattr(self, var)
                    from brian2 import EventMonitor

                    if isinstance(obj, EventMonitor):
                        # spkmon = SpkMon(**{k: v for k, v in obj.get_states().items() if k != 'N'})
                        # file_name = os.path.join("temp", self.sim_name + ".npz")
                        # # from brian2 import defaultclock
                        # # t_index = np.round(np.arange(0, self.duration/time_unit, defaultclock.dt), 4)
                        # # d = {}
                        # # for i, train in obj.spike_trains().items():
                        # #     ts = np.zeros(len(t_index))
                        # #     for t in train:
                        # #         ts[np.argmin(np.abs(t_index - t/time_unit))] = 1
                        # #     d[i] = pd.arrays.SparseArray(ts)
                        # # pd.DataFrame(d)
                        # # from scipy.sparse import csr_matrix
                        # # i, t = obj.it
                        # # spks = np.arange(len(t))
                        # # sp_df = pd.DataFrame.sparse.from_spmatrix(csr_matrix((t, (i, spks))),
                        # #                                           # index=t_index,
                        # #                                           columns=np.arange(obj.N))
                        # np.savez_compressed(file_name, sp_all=spkmon)
                        # logger.debug(f"saved {var} to {file_name}")
                        continue
                    arr = getattr(self, var).mean(axis=0)[::subsample]
                    df_builder[var] = with_best_unit(arr)
                elif np.iterable(value):
                    monitor = getattr(self, var)
                    for pop, idxs in pop_idxs.items():
                        if len(idxs):
                            for state_var in value:
                                arr = getattr(monitor[idxs], state_var).mean(axis=0)[
                                    ::subsample
                                ]
                                df_builder[f"{state_var}_{pop}"] = with_best_unit(arr)

        _df = pd.DataFrame(
            df_builder,
            index=np.round(self.r_all.get_states()["t"] / time_unit, 5)[::subsample],
        )
        _df.sort_index(axis="columns", inplace=True)
        return _df

    def process_data(self):
        pass


class SpkMon(namedtuple("SpkMon", "i t count")):
    """Helper class for a stored SpikeMonitor"""

    __slots__ = ()

    @property
    def it(self):
        return self.i, self.t

    # save npz
    def save(self, file_name):
        np.savez_compressed(file_name, i=self.i, t=self.t, count=self.count)

    @staticmethod
    def load(file_name):
        data = np.load(file_name)
        return SpkMon(**data)
