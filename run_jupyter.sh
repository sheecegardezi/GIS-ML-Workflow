#!/bin/bash
#------------------------------------------------------------------------
#------------------------- define nci cluster size ----------------------
#------------------------------------------------------------------------
#PBS -N mlflowpbs
#PBS -P ge3
#PBS -q normalbw
#PBS -l walltime=6:00:00
#PBS -l ncpus=112
#PBS -l mem=400GB
#PBS -l jobfs=100GB
#PBS -l storage=gdata/ge3+gdata/u46
#PBS -l other=hyperthread


#------------------------------------------------------------------------
#------------------------- define task parameters -----------------------
#------------------------------------------------------------------------

MLHOME=/g/data/ge3/$USER  # where you have installed venvs/MLWorkflow, etc

iteration=5
jupyterPort=838$iteration
rayDashboardPort=848$iteration
rayPort=637$iteration


#------------------------------------------------------------------------
#------------------------- load nci modules -----------------------------
#------------------------------------------------------------------------

module purge
module load pbs
module load python3-as-python
module load gdal/3.0.2

set -e
ulimit -s unlimited

echo $MLHOME

source $MLHOME/venvs/MLWorkflow/bin/activate
cd $MLHOME/github/MLWorkflow


#------------------------------------------------------------------------
#------------------------- setup ray cluster ----------------------------
#------------------------------------------------------------------------


cd $PBS_O_WORKDIR

nodeDnsIps=`cat $PBS_NODEFILE | uniq`
hostNodeDnsIp=`uname -n`
hostNodeIp=`hostname -i`
rayDashboardPort=$rayDashboardPort
rayPassword='5241590000000000'


cat > $PBS_O_WORKDIR/${iteration}_setupRayWorkerNode.sh << 'EOF'
#!/bin/bash -l
set -e
ulimit -s unlimited
cd $PBS_O_WORKDIR

hostNodeIp=${1}
rayPort=${2}
rayPassword=${3}
MLHOME=${4}
hostIpNPort=$hostNodeIp:$rayPort

module purge
module load pbs
module load python3-as-python
module load gdal/3.0.2
source $MLHOME/venvs/MLWorkflow/bin/activate
cd $MLHOME/github/MLWorkflow

echo "running node to ray cluster"
echo `uname -n`
echo `hostname -i`
echo `ray start --address=$hostIpNPort --num-cpus=56 --redis-password='5241590000000000'  --block &`

EOF

chmod +x $PBS_O_WORKDIR/${iteration}_setupRayWorkerNode.sh

echo "set up ray cluster......."
for nodeDnsIp in `echo ${nodeDnsIps}`
do
        if [[ ${nodeDnsIp} == "${hostNodeDnsIp}" ]]
        then
                echo "Starting ray cluster on head node ..."
                module purge
                module load pbs
                module load python3-as-python
                module load gdal/3.0.2
                source $MLHOME/venvs/MLWorkflow/bin/activate
                cd $MLHOME/github/MLWorkflow
                ray start --head --num-cpus=56 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=${rayDashboardPort} --port=${rayPort}
                sleep 10
        else
                echo "Starting ray cluster on worker node ..."
                pbs_tmrsh "${nodeDnsIp}" $PBS_O_WORKDIR/${iteration}_setupRayWorkerNode.sh "${hostNodeIp}" "${rayPort}" "${rayPassword}" "${MLHOME}" &
                sleep 5
        fi
done



echo "Creating ray connection string ..."
echo "ssh -N -L ${rayDashboardPort}:${hostNodeDnsIp}:${rayDashboardPort} ${USER}@gadi.nci.org.au &" > ${PBS_O_WORKDIR}/${iteration}_connection_strings.txt

#------------------------------------------------------------------------
#------------------------- setup jupyter notebook -----------------------
#------------------------------------------------------------------------


hostNodeDnsIp=`uname -n`

echo "Starting Jupyter lab ..."
jupyter notebook --no-browser  --port ${jupyterPort} --no-browser --ip=${hostNodeDnsIp} --NotebookApp.token='' --NotebookApp.password='' &

echo "Creating jupyter connection string ..."
echo "ssh -N -L ${jupyterPort}:${hostNodeDnsIp}:${jupyterPort} ${USER}@gadi.nci.org.au &" >> ${PBS_O_WORKDIR}/${iteration}_connection_strings.txt

#------------------------------------------------------------------------
#------------------------- run ml workflow ------------------------------
#------------------------------------------------------------------------


cd $MLHOME/github/MLWorkflow
sleep infinity  # this allows the pbs nodes to persist until requested wall timeout, therefore you can run jupyter notebook and terminal in a browser

#------------------------------------------------------------------------
#------------------------- gracefully exit ------------------------------
#------------------------------------------------------------------------

rm *setupRayWorkerNode.sh -f
rm *connection_strings* -f
rm mlflowpbs* -f
rm core.ray:* -f
rm core.raylet* -f
rm core.store* -f
