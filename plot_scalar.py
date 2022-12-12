from src.plot import tbplot_s

datasets=['cifar10'];
tasks=['feature_evolve'];
logfiles=['train'];
legend=['HH','LL','HL','LH'];
exp_id='waveeff';

save_path = "plots"
log_basepath = "log"
smooth=5;

xlabel="steps";
ylabel="Loss";

# wavelets=[]
#models=['lenet5-lrelu', 'lenet5_dwt-lrelu/db2/LL'];
#models = ['resnet50-lrelu', 'resnet50_dwt-lrelu/bior4.4/LL'];
models = ['vgg16_dwt-lrelu/bior4.4/HH','vgg16_dwt-lrelu/bior4.4/LL','vgg16_dwt-lrelu/bior4.4/HL','vgg16_dwt-lrelu/bior4.4/LH']
tbplot_s(exp_id,datasets,tasks,models,logfiles,save_path,smooth=smooth,xlabel=xlabel,ylabel=ylabel,log_basepath=log_basepath, legend = legend);
