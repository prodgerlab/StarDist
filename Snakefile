# The main entry point of your workflow.
# After configuring, running snakemake -n in a clone of this repository should successfully execute a dry-run of the workflow.


# Allow users to fix the underlying OS via singularity.
#container: "docker://continuumio/miniconda3"

configfile: "config/config.yaml"

def get_all_indices(split):
    return expand('results/in_{chan}/{split}/{imtype}/image{i}.tif',
                    chan=config['in_images'].keys(),
                    i=config['split_set'][split],
                    split=split,
                    imtype=['images','masks'])


rule all_train:
    input: 
        model_zip = expand('results/models/multichan_{mask}_model.zip',mask=config['in_masks'].keys()),


rule all_predict:
    input: 
        labels = expand('results/predictions/multichan_{chan}/image{i}.tif',
                                        i=config['split_set']['test'],
                                        chan=config['in_masks'].keys(),
                                        allow_missing=True),


rule all_preproc:
    input:
        images = expand('results/in_data/training/multichan/image{i}.tif',
                                        i=config['split_set']['training'],
                                        chan=config['in_masks'].keys(),
                                        allow_missing=True),
        masks = expand('results/in_{chan}/training/masks/image{i}.tif',
                                        i=config['split_set']['training'],
                                        chan=config['in_masks'].keys(),
                                        allow_missing=True),
 

rule import_image:
    """just copies it"""
    input: lambda wildcards: config['in_images'][wildcards.chan]
    output: 'results/in_{chan}/{split}/images/image{i}.tif'
    shell: "cp '{input}' {output}"

rule import_multichan:
    """just copies it"""
    input: lambda wildcards: config['in_multichan']
    output: 'results/in_data/{split}/multichan/image{i}.tif'
    shell: "cp '{input}' {output}"


rule import_mask:
    """ runs connected components to get each separate instance"""
    input: lambda wildcards: config['in_masks'][wildcards.chan]
    output: 'results/in_{chan}/{split}/masks/image{i}.tif'
    shell: "c3d '{input}' -comp -o {output}"  


rule train_model:
    input: 
        images = expand('results/in_{chan}/training/images/image{i}.tif',
                                        i=config['split_set']['training'],
                                        allow_missing=True),
        masks = expand('results/in_{chan}/training/masks/image{i}.tif',
                                        i=config['split_set']['training'],
                                        allow_missing=True),
    output:
        model_dir = directory('results/models/{chan,[a-zA-Z0-9]+}')
    container: 'stardist.sif'
    threads: 8
    resources:
        gpus = 1,
        mem_mb = 32000,
        time = 360
    script: 'scripts/train_model.py'

rule train_model_multichan:
    input: 
        images = expand('results/in_data/training/multichan/image{i}.tif',
                                        i=config['split_set']['training'],
                                        allow_missing=True),
        masks = expand('results/in_{chan}/training/masks/image{i}.tif',
                                        i=config['split_set']['training'],
                                        allow_missing=True),
    output:
        model_dir = directory('results/models/multichan_{chan}')
    container: 'stardist.sif'
    threads: 8
    resources:
        gpus = 1,
        mem_mb = 32000,
        time = 360
    script: 'scripts/train_model.py'


rule export_model:
    input:
        model_dir = directory('results/models/multichan_{chan}')
    output:
        model_zip = 'results/models/multichan_{chan}_model.zip'
    container: 'stardist.sif'
    threads: 1
    script: 'scripts/export_model.py'

   
        

rule predict_test_labels:
    input:
        model_dir = directory('results/models/multichan_{chan}'),
        images = expand('results/in_data/test/multichan/image{i}.tif',
                                        i=config['split_set']['test'],
                                        allow_missing=True),
    output:
        labels = expand('results/predictions/multichan_{chan}/image{i}.tif',
                                        i=config['split_set']['test'],
                                        allow_missing=True),
        fiji_rois = expand('results/predictions/multichan_{chan}/image{i}_rois.zip',
                                        i=config['split_set']['test'],
                                        allow_missing=True),
    container: 'stardist.sif'
    script: 'scripts/predict.py'


