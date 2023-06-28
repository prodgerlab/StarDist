## Instructions:

1. Create and activate a virtual environment and install the dependencies with `pip install .`

2. Unpack your raw data into the `raw_data` subfolder. Note: if you want to dry-run this workflow with 0-sized data files, you can extract the `raw_data/dry_run_test.tar`. 

3. Ensure all the file paths to your datasets are correctly named in the `config/config.yml`. Note: a dry-run will complain if any files cannot be found.

4. Run a dry-run with `snakemake -npr`

5. To train, use: `snakemake all_train`

6. To predict, use: `snakemake all_predict`
   

