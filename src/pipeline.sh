#!/bin/zsh
# End-to-end run over every stimulus set:
#   domains  ori, hue, size
#   modes    standard           mean + SD probes, N = all set sizes
#            discount_spatial   outlier in the opposite quadrant    } N ≥ 3: group-mean probe +
#            discount_mixed     outlier among the group             } discount analysis
# For each (domain, mode): generate stimuli once, then per set size N
#   verify patch masks → extract DINOv2 features → probe
#   (discount sets also: discount.py — mean readout trained on the standard
#    features of the same N, tested on the outlier displays; needs standard first).
#
#   ./pipeline.sh                              # everything, seed 42, 200 reps
#   ./pipeline.sh "ori hue" 7 50 "8 16"        # domains, seed, reps, set sizes
#   MODES=standard ./pipeline.sh size          # restrict modes
# Stimuli are reused when up to date (generate.py checks version/seed/reps and
# renders only missing reps); features are re-extracted when the stimuli are
# newer. Rerunning after a probe.py change skips straight to the probes.
# 200 reps/condition = where the bar_concat regression saturates.
# Standard-mode targets: mean / variance classification (labels = nearest
# realized level) and mean_reg / variance_reg (within-cell ridge readout of
# the realized statistic); variance targets need N ≥ 2.
#
# Outputs (keyed by seed, never overwritten across seeds):
#   stimuli_<domain>[_discount_<layout>]_s<seed>/
#   features/<domain>_<mode>_n<N>_s<seed>.pt
#   results_s<seed>/<domain>/n<N>_<target>_linear.{txt,json}         standard (plot_neurips.py input)
#   results_s<seed>/<domain>_<mode>/n<N>_mean_linear.{txt,json}      discount
#   results_s<seed>/<domain>_<mode>/n<N>_discount.{txt,json}         outlier weight β
set -e
cd "$(dirname "$0")"

DOMAINS=(${=${1:-ori hue size}})
SEED=${2:-42}
REPS=${3:-200}
SIZES=(${=${4:-1 2 3 4 8 16 32}})
MODES=(${=${MODES:-standard discount_spatial discount_mixed}})
mkdir -p features

for DOMAIN in $DOMAINS; do
  for MODE in $MODES; do
    if [[ $MODE == standard ]]; then
      STIM=stimuli_${DOMAIN}_s${SEED}
      RES=results_s${SEED}/${DOMAIN}
      GEN_ARGS=()
      TARGETS=(mean variance mean_reg variance_reg)
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
      if [[ ! -d $STIM/n$(printf %02d $N) ]]; then
        echo "=== $DOMAIN / $MODE / N=$N: not in $STIM (e.g. does not fit a quadrant), skipping ==="
        continue
      fi
      FEATS=features/${DOMAIN}_${MODE}_n${N}_s${SEED}.pt
      if [[ ! -f $FEATS || $STIM/metadata.json -nt $FEATS ]]; then
        echo "=== $DOMAIN / $MODE / N=$N: verify masks, extract features ==="
        python3 verify_mask.py --metadata $STIM/metadata.json --n_bars $N
        python3 loader.py --metadata $STIM/metadata.json --n_bars $N --output $FEATS
      fi
      for TARGET in $TARGETS; do
        [[ $N -eq 1 && $TARGET == variance* ]] && continue   # one element has no SD
        echo "=== $DOMAIN / $MODE / N=$N: probe $TARGET ==="
        python3 probe.py --features $FEATS --target $TARGET --n_bars $N --seed $SEED --per_token \
            --save $RES/n${N}_${TARGET}_linear.json | tee $RES/n${N}_${TARGET}_linear.txt
      done
      TRAIN=features/${DOMAIN}_standard_n${N}_s${SEED}.pt
      if [[ $MODE != standard && -f $TRAIN ]]; then
        echo "=== $DOMAIN / $MODE / N=$N: discount analysis ==="
        python3 discount.py --train $TRAIN --test $FEATS \
            --save $RES/n${N}_discount.json | tee $RES/n${N}_discount.txt
      fi
    done
  done
done
echo "plot: python3 plot_neurips.py --results_dir results_s${SEED}"
