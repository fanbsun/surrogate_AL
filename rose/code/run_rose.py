import os, sys
import asyncio
import yaml

from rose.learner import Learner
import radical.pilot as rp

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import RadicalExecutionBackend

async def main():
    
    engine = await RadicalExecutionBackend({
        'resource': 'ncsa.delta_gpu_a40',
        'runtime': 60,
        'access_schema':'interactive',
        'project' : "bdyk",
        'cores'   : 16,
        'gpus'    : 1
        })
    asyncflow = await WorkflowEngine.create(engine)
    learner = Learner(asyncflow)

    code_path = f'{sys.executable} {os.getcwd()}'

    #FIXME!! I am utility task!!
    @learner.simulation_task
    async def bootstrap(*args, pipeline_dir, input_data_dir, seed):
        return f'{code_path}/bootstrap.py --pipeline_dir {pipeline_dir} --input_data_dir {input_data_dir} --seed {seed}'
    
    #FIXME!! For this two tasks, need to specify if using GPU, and other variables!!
    @learner.training_task
    async def training(*args, config_file, iteration, pipeline_dir, task_description={}):
        return f'{code_path}/train.py --config {config_file} --iteration {iteration} --pipeline_dir {pipeline_dir}'

    @learner.active_learn_task
    async def active_learning(*args, config_file, iteration, pipeline_dir, num_new_samples, task_description={}):
        return f'{code_path}/active.py --config {config_file} --iteration {iteration} --pipeline_dir {pipeline_dir} --n_new_samples {num_new_samples}'

    async def teach_single_pipeline(input_data_dir, config_file, pipeline_dir, num_iter, num_new_samples, seed):
        # Need to load conf before hand. This is because resource for different model will be different
        cfg = yaml.safe_load(open(config_file))
        model=cfg["model"]

        iter_id = 1
        print("Start doing bootstrap (only once!)")

        bs = await bootstrap(pipeline_dir=pipeline_dir, input_data_dir=input_data_dir, seed=seed)

        while iter_id < num_iter:   # n iter means in total n traning and n-1 al
            print(f"Start doing iteration {iter_id}")
            train_task_description = {
                    'cores_per_rank'    : 4 if model == 'bnn' else 8,
                    'gpus_per_rank'     : 1 if model == 'bnn' else 0,
                    'gpu_type'          : rp.CUDA if model == 'bnn' else None,
                    #'threading_type'    : rp.OpenMP,
                    }
            train = training(
                    config_file=config_file, 
                    iteration=iter_id, 
                    pipeline_dir=pipeline_dir,
                    task_description=train_task_description)
            if iter_id == num_iter - 1:
                await train
            else:
                active_learn = await active_learning(
                        train, 
                        config_file=config_file, 
                        iteration=iter_id, 
                        pipeline_dir=pipeline_dir, 
                        num_new_samples=num_new_samples,
                        task_description=train_task_description)
            iter_id += 1

    async def teach():
        submitted_pipelines = []

        input_data_dir = "/u/fsun2/surrogate_data/nano-confinement/"

        seed_list = [42,46]
        conf_list           = ["/u/fsun2/surrogate_AL_rose/config/gpr.yaml", 
                               "/u/fsun2/surrogate_AL_rose/config/bnn.yaml",
                               ]
        pipeline_dir_list   = ["/u/fsun2/rose_experiments/test_2",
                               "/u/fsun2/rose_experiments/test_3",
                               ]
    
    
        for config_file, pipeline_dir, seed in zip(conf_list, pipeline_dir_list, seed_list):
            submitted_pipelines.append(
                    teach_single_pipeline(
                        input_data_dir=input_data_dir, 
                        config_file=config_file, 
                        pipeline_dir=pipeline_dir, 
                        num_iter=100, 
                        num_new_samples=10,
                        seed=seed))

        results = await asyncio.gather(*submitted_pipelines)

    await teach()
    await learner.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
