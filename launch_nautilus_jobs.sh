#!/bin/bash

echo "Launching Jobs"
echo "---------------------------------------------------------------------------------------------"

print_help(){
    echo -e "Usage: ./$0 [-t <exp_label>] [-d] [-p]"
    echo -e "\t\t -t \t exp_label: \tShort phrase to identify table name, value is prepend to the job_name"
    echo -e "\t\t -d \t dryrun: \tRun Helm in dryrun mode"
    echo -e "\t\t -p \t parallel: \tLaunch experiments in parallel"
    echo
}

exp_label="default"
dryrun=""
parallel=""
nshift=0;
while getopts t:hdp flag
do
    case "${flag}" in
        t) exp_label="${OPTARG}"
            ((nshift=nshift+1));;
        d) dryrun="--dry-run --debug";;
        p) parallel="&";;
        *)
            print_help;
            exit -1;
        ;;
    esac
    ((nshift=nshift+1));
done
shift $nshift

echo "exp_label: $exp_label"
echo "dryrun: $dryrun"
echo "parallel: $parallel"
echo 

dataset="fp302u"
task="feature_evolve"

wavelet=("haar" "db2" "db4" "db6" "db8" "db16" "bior2.2" "bior3.3" "bior4.4")
downsample=("LL" "LH" "HL" "HH" )

model_base="vgg16_dwt-lrelu"

for exp_id in $(seq 0 $((${#wavelet[@]}-1)))
do

    for _i in $(seq 0 $((${#downsample[@]}-1)))
    do
        model="${model_base}/${wavelet[$exp_id]}/${downsample[$_i]}"

        job_name="ImpWCNet.${dataset}.${exp_label}.$(echo $model_base | tr -s '_' '.').${wavelet[$exp_id]}.${downsample[$_i]}"
        release_name="impwcnet.${dataset}.$(echo $model_base | tr -s '_' '.').${wavelet[$exp_id]}.$(echo ${downsample[$_i]} |  tr '[:upper:]' '[:lower:]')"
        
        exp_name="${dataset}-${job_name}-${task}"
        echo "Start Exp ${dataset}-${job_name} ..."
        
        helm install "$release_name" chart/Improved-WaveCNet/ $dryrun \
            --set job.name="$job_name" \
            --set job.description="experimentName: $(printf '%q' "$exp_name")" \
            \
            --set ImpWCNet.seed=202211 \
            --set ImpWCNet.num_epoch=250 \
            --set ImpWCNet.batch_size=32 \
            --set ImpWCNet.lr0=0.01 \
            --set ImpWCNet.lr_a=0.001 \
            --set ImpWCNet.lr_b=0.75 \
            --set ImpWCNet.momentum=0.9 \
            --set ImpWCNet.w_decay=0.0005 \
            --set ImpWCNet.lbl_sm=0.01 \
            --set ImpWCNet.model="${model}" \
            --set ImpWCNet.wavelet="${wavelet[$exp_id]}" \
            --set ImpWCNet.dataset="${dataset}" \
            --set ImpWCNet.num_workers=4 \
            --set ImpWCNet.task="${task}" \
            --set ImpWCNet.datadir="data/" \
            --set ImpWCNet.logdir="log/" \
            --set ImpWCNet.ptdir="pretrain/" \
            --set ImpWCNet.log_filename="train" \
            --set ImpWCNet.init_weights=0 \
            --set ImpWCNet.resume_train=0 \
            --set ImpWCNet.exp_label="${exp_label}" \
            --set "requiredGPU={NVIDIA-A10,NVIDIA-GeForce-RTX-3090,NVIDIA-GeForce-RTX-3090,NVIDIA-TITAN-RTX,NVIDIA-RTX-A5000,Quadro-RTX-6000,Tesla-V100-SXM2-32GB,NVIDIA-A40,NVIDIA-RTX-A6000,Quadro-RTX-8000}" \
            --set ImpWCNet.gpu=0  $parallel
        # break;
    done
    # break;
done 
echo 
echo "Waiting for subprocesses to finish launching"
# Wait for background jobs to finish
wait
echo "Done!"