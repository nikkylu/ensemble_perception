#!/bin/zsh
# End-to-end run over every stimulus set:
#   domains  ori, hue, size
#   modes    standard           mean + SD probes, N = all set sizes
#            discount_spatial   outlier in the opposite quadrant    } mean probe only (one SD
#            discount_mixed     outlier among the group             }  level), N ≥ 3
# For each (domain, mode): generate stimuli once, then per set size N
#   verify patch masks → extract DINOv2 features → probe.
#
#   ./pipeline.sh                              # everything, seed 42, 50 reps
#   ./pipeline.sh "ori hue" 7 50 "8 16"        # domains, seed, reps, set sizes
#   MODES=standard ./pipeline.sh size          # restrict modes
#
# Outputs (keyed by seed, never overwritten across seeds):
#   stimuli_<domain>[_discount_<layout>]_s<seed>/
#   features/<domain>_<mode>_n<N>_s<seed>.pt
#   results_s<seed>/<domain>/n<N>_<target>_linear.{txt,json}         standard (plot_neurips.py input)
#   results_s<seed>/<domain>_<mode>/n<N>_mean_linear.{txt,json}      discount
set -e
cd "$(dirname "$0")"

DOMAINS=(${=${1:-ori hue size}})
SEED=${2:-42}
REPS=${3:-50}
SIZES=(${=${4:-1 2 3 4 8 16 32}})
MODES=(${=${MODES:-standard discount_spatial discount_mixed}})
mkdir -p features

for DOMAIN in $DOMAINS; do
  for MODE in $MODES; do
    if [[ $MODE == standard ]]; then
      STIM=stimuli_${DOMAIN}_s${SEED}
      RES=results_s${SEED}/${DOMAIN}
      GEN_ARGS=()
      TARGETS=(mean variance)
      NS=($SIZES)
    else
      STIM=stimuli_${DOMAIN}_${MODE}_s${SEED}
      RES=results_s${SEED}/${DOMAIN}_${MODE}
      GEN_ARGS=(--mode discount --layout ${MODE#discount_})
      TARGETS=(mean)
      NS=(${SIZES:#(1|2)})                       # discount needs a group of ≥ 2
    fi
    (( ${#NS} )) || continue
    mkdir -p $RES

    echo "=== $DOMAIN / $MODE: generate (seed=$SEED reps=$REPS N=${NS[*]}) ==="
    python3 stimuli/generate.py $DOMAIN $GEN_ARGS --seed $SEED --n_reps $REPS \
        --set_sizes ${(j:,:)NS} --out $STIM

    for N in $NS; do
      FEATS=features/${DOMAIN}_${MODE}_n${N}_s${SEED}.pt
      echo "=== $DOMAIN / $MODE / N=$N: verify masks, extract features ==="
      python3 verify_mask.py --metadata $STIM/metadata.json --n_bars $N
      python3 loader.py --metadata $STIM/metadata.json --n_bars $N --output $FEATS
      for TARGET in $TARGETS; do
        echo "=== $DOMAIN / $MODE / N=$N: probe $TARGET ==="
        python3 probe.py --features $FEATS --target $TARGET --n_bars $N --seed $SEED \
            --save $RES/n${N}_${TARGET}_linear.json | tee $RES/n${N}_${TARGET}_linear.txt
      done
    done
  done
done
echo "plot: python3 plot_neurips.py --results_dir results_s${SEED}"
