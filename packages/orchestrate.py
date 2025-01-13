from typing import Any
import polars as pl
import dill
import datetime 
from collections import OrderedDict
import loguru
import os
import json

CACHE_DIR = "cache"
logger = loguru.logger

def write_metadata(experiment_name, step_name, step_version, 
                 step_kwargs, start_time: str):
    # TODO: fill in
    os.makedirs(f"outputs/{experiment_name}/{step_name}", exist_ok=True)
    with open(f"outputs/{experiment_name}/{step_name}/cache_metadata.json", 'w') as f:
        json.dump({
            "version": step_version,
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "start_time": start_time,
            "end_time": datetime.datetime.now().strftime("%H:%M:%S"),
            "kwargs": step_kwargs
        }, f) 

def cache_result(step_name, step_version, step_kwargs, result: Any):
    # TODO: add cache folder
    cache_name = _get_step_cache_name(step_name, step_version, step_kwargs)

    os.makedirs(CACHE_DIR, exist_ok=True)
    # with open(cache_name, 'wb') as f:
    with open(os.path.join(CACHE_DIR, cache_name), 'wb') as f:
        dill.dump(result, f)
    # return the cache path
    return os.path.join(CACHE_DIR, cache_name)

def load_from_cache(step_name, step_version, step_kwargs):
    # TODO: add cache folder.
    cache_name = _get_step_cache_name(step_name, step_version, step_kwargs)

    with open(os.path.join(CACHE_DIR, cache_name), 'rb') as f:
        return dill.load(f)

def _get_step_cache_name(step_name, step_version, step_kwargs):
    kwargs = step_kwargs.copy()
    # these should always be in there, unless we're doing step
    if "version" in kwargs:
        kwargs.pop("version")
    if "step_name" in kwargs:
        kwargs.pop("step_name")
    # remove k-v pairs where the key ends in "_ignore"
    for k in list(kwargs.keys()):
        if k.endswith("_ignore"):
            kwargs.pop(k)
    kwarg_str = "-".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
    return f"{step_name}-{step_version}-{kwarg_str}.dill" if kwarg_str else f"{step_name}-{step_version}.dill"

def _get_cacheable_cache_name(cacheable_name, cacheable_version, cacheable_kwargs):
    kwargs = cacheable_kwargs.copy()
    # these should always be in there, unless we're doing step
    if "version" in kwargs:
        kwargs.pop("version")
    if "cacheable_name" in kwargs:
        kwargs.pop("cacheable_name")
    kwarg_str = "-".join([f"{k}={v}" for k, v in sorted(kwargs.items())]) # NOTE: in the future, we may want to hash this instead of using the string.
    return f"{cacheable_name}-{cacheable_version}-{kwarg_str}.dill" if kwarg_str else f"{cacheable_name}-{cacheable_version}.dill"

# Steps have different behaviour, they're discrete units with their own metadata 
def meta_step(steps): 
    def step(step_func):
        def wrapper(*args, **kwargs):
            step_name = kwargs["step_name"]
            logger.info(f"Running step {step_name}")
            step_version = kwargs["version"]
            cache_name = _get_step_cache_name(step_name, step_version, kwargs)
            fcache_name = os.path.join(CACHE_DIR, cache_name)

            if os.path.exists(fcache_name):
                logger.info(f"Step {step_name} is cached at {fcache_name}, continuing.")
                return fcache_name
                # with open(fcache_name, 'rb') as f:
                #     return fcache_name, dill.load(f)
            else:
            #### DAG input logic goes here. ####:
                all_steps = steps.keys()
                original_kwargs = kwargs.copy()
                for key, value in kwargs.items(): 
                    if value == step_name:
                        continue
                    if value in all_steps: # substitute the value with the result of the step.
                        value_step_name = value
                        kwargs[key] = load_from_cache(value_step_name, steps[value_step_name][1]['version'], steps[value_step_name][1])
                result = step_func(*args, **kwargs)
                cache_path = cache_result(step_name, step_version, original_kwargs, result)
                return cache_path
        return wrapper
    return step

# cacheables can only be used within a step. They cannot have dependencies on previous steps, only the current step.
    # they also are only stored in the cache dir and not sim linked to the experiment folder 
def cacheable(func):
    def wrapper(*args, **kwargs):
        # TODO: we need to know the name of the step, the version of the step, and the kwargs of the step.
        # TODO: check if the function is cached. If it is, load the result and return it.
        assert "cacheable_name" in kwargs, "cacheable_name must be in kwargs"
        assert "version" in kwargs, "version must be in kwargs"
        cacheable_name = kwargs["cacheable_name"]
        logger.info(f"Running cacheable {cacheable_name}")
        cacheable_version = kwargs["version"]
        cache_name = _get_cacheable_cache_name(cacheable_name, cacheable_version, kwargs)
        fcache_name = os.path.join(CACHE_DIR, cache_name)
        if os.path.exists(fcache_name):
            logger.info(f"Cacheable {cacheable_name} is cached, loading.")
            with open(fcache_name, 'rb') as f:
                return dill.load(f)
        else:
            result = func(*args, **kwargs)
            os.makedirs(CACHE_DIR, exist_ok=True)
            # with open(cache_name, 'wb') as f:
            with open(os.path.join(CACHE_DIR, cache_name), 'wb') as f:
                dill.dump(result, f)
            return result
    return wrapper
        

def step_collate_predictions_across_models(version: str, step_name: str, **kwargs):
    """

    Args:
        version (str): _description_
        step_name (str): _description_
    """
    prediction_frames = []
    for k, v in kwargs.items():
        if k.startswith("predictions_"):
            prediction_frames.append(pl.DataFrame(v))
    return pl.concat(prediction_frames)

def conduct(experiment_steps: OrderedDict, experiment_name: str, latest_symlink_dir: str):
    step_wrapper = meta_step(experiment_steps)
    for step_name in experiment_steps:
        # step_name, step_fn, step_kwargs = step
        step_fn = step_wrapper(experiment_steps[step_name][0])
        step_kwargs = experiment_steps[step_name][1]
        step_version = step_kwargs["version"]
        start_time = datetime.datetime.now().strftime("%H:%M:%S")
        step_kwargs["step_name"] = step_name
        result_cache_path = step_fn(**step_kwargs)
        write_metadata(experiment_name, step_name, step_version, step_kwargs, start_time)