#!/bin/zsh
# End-to-end run for one domain and seed:
#   1. generate stimuli for all set sizes          stimuli_<domain>_s<seed>/
#   2. per set size N:
#        verify patch masks (pixel check)
#        extract DINOv2 features                   features/<domain>_n<N>_s<seed>.pt
#        probe mean and SD (nested CV)             results_s<seed>/<domain>/n<N>_<target>_linear.{txt,json}
#
#   ./pipeline.sh ori                       # seed 42, 50 reps, N = 1 2 3 4 8 16 32
#   ./pipeline.sh hue 7 50 "8 16"           # domain seed reps "set sizes"
#   python3 plot_neurips.py --results_dir results_s42
set -e
cd "$(dirname "$0")"

DOMAIN=${1:-ori}
SEED=${2:-42}
REPS=${3:-50}
SIZES=(${=${4:-1 2 3 4 8 16 32}})

STIM=stimuli_${DOMAIN}_s${SEED}
RES=results_s${SEED}/${DOMAIN}
mkdir -p features $RES

echo "=== generate $DOMAIN stimuli: seed=$SEED reps=$REPS N=(${SIZES[*]}) ==="
python3 stimuli/generate.py $DOMAIN --seed $SEED --n_reps $REPS \
    --set_sizes ${(j:,:)SIZES} --out $STIM

for N in $SIZES; do
    FEATS=features/${DOMAIN}_n${N}_s${SEED}.pt
    echo "=== N=$N: verify masks ==="
    python3 verify_mask.py --metadata $STIM/metadata.json --n_bars $N
    echo "=== N=$N: extract features ==="
    python3 loader.py --metadata $STIM/metadata.json --n_bars $N --output $FEATS
    for TARGET in mean variance; do
        echo "=== N=$N: probe $TARGET ==="
        python3 probe.py --features $FEATS --target $TARGET --n_bars $N --seed $SEED \
            --save $RES/n${N}_${TARGET}_linear.json | tee $RES/n${N}_${TARGET}_linear.txt
    done
done
