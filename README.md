# MLWorkflow


Machine Learning Workflow Automation

--------------------------------------------------------------------------------
## Summary

- Preprocess list of geotiff files for machine learning algorithms
- support state-of-art machine learning algorithms including XGBoost, Random Forest, and SVR.
- perform feature ranking 
- perform hyper-parameter tuning with algorithms including Bayesian Search 
- create model analysis and visualizations 

Setting up on NCI:

    ssh <USERNAME>@gadi.nci.org.au
    
    # working dir
    export MLHOME=/g/data/ge3
    cd $MLHOME
    
    # load req packages
    module purge
    module load pbs
    module load python3-as-python
    module load gdal/3.0.2
    
    # get updated code 
    mkdir -p $MLHOME/github
    cd $MLHOME/github
    rm -rf MLWorkflow
    git clone git@github.com:GeoscienceAustralia/MLWorkflow.git
    
    # create python environment
    rm -rf $MLHOME/venvs/MLWorkflow
    python3 -m venv $MLHOME/venvs/MLWorkflow
    source $MLHOME/venvs/MLWorkflow/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r $MLHOME/github/MLWorkflow/requirements.txt
    chmod 0700 $MLHOME    
    
    # make scripts excuatable 
    cd $MLHOME/github/MLWorkflow
    sed -i -e 's/\r$//' run*
    chmod +x run*

	
--------------------------------------------------------------------------------
## How to Install the Software at NCI GADI

    ssh abc123@gadi.nci.org.au

    module purge
    module load pbs
    module load python3-as-python
    module load gdal/3.0.2
    
    export MLHOME=/g/data/ge3/$USER   # this can be pointed any dir with enough storage space. 
    # user-specific work space in which github code, virtual environment is located 
    
    mkdir -p $MLHOME/github
    cd $MLHOME/github
    rm -rf MLWorkflow
    git clone git@github.com:GeoscienceAustralia/MLWorkflow.git
    # to test bleeding edge features checkout develop branch 
    # git clone -b develop --single-branch git@github.com:GeoscienceAustralia/MLWorkflow.git

    cd MLWorkflow
    rm -rf $MLHOME/venvs/MLWorkflow
    python3 -m venv $MLHOME/venvs/MLWorkflow
    source $MLHOME/venvs/MLWorkflow/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r $MLHOME/github/MLWorkflow/requirements.txt
    pip install -e .
    chmod 0700 $MLHOME    
    sed -i -e 's/\r$//' run*
    chmod +x run*

    cd /g/data/ge3/$USER/github/MLWorkflow

    # Change the output folder in the configration file, so that a writable dir can be used
    ## outputfolder =  /g/data/ge3/sg4953/results/debugging
    ## outputfolder =  /g/data/ge3/fxz547/results/debugging

--------------------------------------------------------------------------------
## How to Run Jupyter Notebook at NCI GADI

    qsub run_jupyter.sh
    less 5_connection_strings.txt
    # copy string in local terminal 
    # navigate to http://localhost:8385 in browser 
    # choose the notebook you want to run

--------------------------------------------------------------------------------
## Running Software on NCI

    qsub -v inputConfigFile="$MLHOME//github/MLWorkflow/configurations_examples/default_configuration.ini" run_small_workflow.sh
    qsub -v inputConfigFile="/g/data/ge3/sg4953/github/MLWorkflow/configurations_examples/reference_configuration_6.ini" run_large_workflow.sh
    less 1_connection_strings.txt

--------------------------------------------------------------------------------
## Running Software on Local Linux Machine 

    # input datasets
    # refer to bin/download_datasets.sh to download sample input dataset 
    
    # download software 
    mkdir -p $MLHOME/projects
    cd $MLHOME/projects
    rm -rf MLWorkflow
    git clone git@github.com:GeoscienceAustralia/MLWorkflow.git -b develop

    git clone git@github.com:GeoscienceAustralia/MLWorkflow.git -b sheece-tests
    
    # setup python environment 
    cd $MLHOME
    rm -rf $MLHOME/venvs/MLWorkflow
    python3 -m venv $MLHOME/venvs/MLWorkflow
    source $MLHOME/venv/MLWorkflow/bin/activate
    rm -rf ~/.cache/pip
    pip install -r $MLHOME/MLWorkflow/requirements.txt
 
    cd $MLHOME/projects/MLWorkflow
    source $MLHOME/venvs/MLWorkflow/bin/activate
    
    ray start --head
    
    python -m mlwkf -c experements/small_test.ini 
     
     
--------------------------------------------------------------------------------     
### Remove Temporary Files after running the pipeline

    rm *setupRayWorkerNode.sh -f
    rm *connection_strings* -f
    rm mlflowpbs* -f
    rm core.ray:* -f
    rm core.raylet* -f


--------------------------------------------------------------------------------    
### Compile

    find . -name '*.py'
    python3 -m compileall .
    find . -name '*.pyc' -delete
    find . -name '*.py' -delete
    find . -name '__pycache__' -delete


--------------------------------------------------------------------------------    
### Testing 

    pytest -rx -s tests
