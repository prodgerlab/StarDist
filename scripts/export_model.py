#!/usr/bin/env python

from stardist.models import StarDist2D


model_dir = snakemake.input.model_dir
out_model_zip = snakemake.output.model_zip

model = StarDist2D(None, name='stardist', basedir=model_dir)
model.export_TF(fname=out_model_zip)



