#!/bin/bash
#------------------------------------------------------------------------
#------------------------- define nci cluster size ----------------------
#------------------------------------------------------------------------

#------------------------------------------------------------------------
#------------------------- define task parameters -----------------------
#------------------------------------------------------------------------

MLHOME=/g/data/ge3/$USER  # where you have installed venvs/MLWorkflow, etc

iteration=1
jupyterPort=838$iteration
rayDashboardPort=848$iteration
rayPort=637$iteration

inputConfigFile=/g/data/ge3/axb562/github/My_MLFLOW/reference_configuration.ini

#------------------------------------------------------------------------
#------------------------- load nci modules -----------------------------
#------------------------------------------------------------------------

module purge
module load pbs
# module load python3-as-python
module load gdal/3.0.2

set -e
ulimit -s unlimited

echo "A01: " $MLHOME
echo "A02: " $inputConfigFile

# source $MLHOME/venvs/MLWorkflow/bin/activate
source /g/data/cb01/anaconda3/bin/activate mlflow
cd $MLHOME/github/MLWorkflow


#------------------------------------------------------------------------
#------------------------- setup ray cluster ----------------------------
#------------------------------------------------------------------------


echo "A03: " $PWD
cd $PWD
ncpu=`nproc`
echo "A04: " $ncpu

nodeDnsIps=`hostname`
echo "A05: " $nodeDnsIps
hostNodeDnsIp=`uname -n`
echo "A06: " $hostNodeDnsIp
hostNodeIp=`hostname -i`
echo "A07: " $hostNodeIp
rayDashboardPort=$rayDashboardPort
echo "A08: " $rayDashboardPort
rayPassword='5241590000000000'


cat > $PWD/${iteration}_setupRayWorkerNode.sh << 'EOF'
#!/bin/bash -l
set -e
ulimit -s unlimited
cd $PWD

hostNodeIp=${1}
rayPort=${2}
rayPassword=${3}
MLHOME=${4}
hostIpNPort=$hostNodeIp:$rayPort

module purge
module load pbs
# module load python3-as-python
module load gdal/3.0.2
# source $MLHOME/venvs/MLWorkflow/bin/activate
source /g/data/cb01/anaconda3/bin/activate mlflow
cd $MLHOME/github/MLWorkflow

echo "running node to ray cluster"
echo "A12: " `uname -n`
echo "A13: " `hostname -i`
echo "A14: " `ray start --address=$hostIpNPort --num-cpus=16 --redis-password='5241590000000000'  --block &`

EOF

chmod +x $PWD/${iteration}_setupRayWorkerNode.sh

echo "set up ray cluster......."
for nodeDnsIp in `echo ${nodeDnsIps}`
do
        if [[ ${nodeDnsIp} == "${hostNodeDnsIp}" ]]
        then
                echo "Starting ray cluster on head node ..."
                module purge
                module load pbs
                # module load python3-as-python
                module load gdal/3.0.2
                # source $MLHOME/venvs/MLWorkflow/bin/activate
                source /g/data/cb01/anaconda3/bin/activate mlflow
                cd $MLHOME/github/MLWorkflow
                ray start --head --num-cpus=${ncpu} --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=${rayDashboardPort} --port=${rayPort}
                sleep 10
        else
                echo "Starting ray cluster on worker node ..."
                ssh "${nodeDnsIp}" $PWD/${iteration}_setupRayWorkerNode.sh "${hostNodeIp}" "${rayPort}" "${rayPassword}" "${MLHOME}" &
                sleep 5
        fi
done



echo "Creating ray connection string ..."
echo "ssh -N -L ${rayDashboardPort}:${hostNodeDnsIp}:${rayDashboardPort} ${USER}@gadi.nci.org.au &" > ${PWD}/${iteration}_connection_strings.txt

#------------------------------------------------------------------------
#------------------------- setup jupyter notebook -----------------------
#------------------------------------------------------------------------


hostNodeDnsIp=`uname -n`
echo "A15: " $hostNodeDnsIp

echo "Starting Jupyter lab ..."
echo "A16: " $jupyterPort
jupyter notebook --no-browser  --port ${jupyterPort} --no-browser --ip=${hostNodeDnsIp} --NotebookApp.token='' --NotebookApp.password='' &

echo "Creating jupyter connection string ..."
echo "ssh -N -L ${jupyterPort}:${hostNodeDnsIp}:${jupyterPort} ${USER}@gadi.nci.org.au &" >> ${PWD}/${iteration}_connection_strings.txt

#------------------------------------------------------------------------
#------------------------- run ml workflow ------------------------------
#------------------------------------------------------------------------


cd $MLHOME/github/MLWorkflow
python -m mlwkf -c $inputConfigFile
# sleep infinity  # this allows the pbs nodes to persist until requested wall timeout, therefore you can run jupyter notebook and terminal in a browser

#------------------------------------------------------------------------
#------------------------- gracefully exit ------------------------------
#------------------------------------------------------------------------

# rm *setupRayWorkerNode.sh -f
# rm *connection_strings* -f
# rm mlflowpbs* -f
# rm core.ray:* -f
# rm core.raylet* -f
# rm core.store* -f

ray stop
