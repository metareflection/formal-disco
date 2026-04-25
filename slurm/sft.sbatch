#!/bin/bash
#SBATCH --job-name=sft_formal_disco-verus-lemma
#SBATCH --account=kempner_poesia_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "$FORMAL_DISCO_ROOT" || { echo "FORMAL_DISCO_ROOT not found"; exit 1; }

source venv/bin/activate
# accelerate launch distill.py sft
# python distill.py sft # data=lemma-verus output_dir=/n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/qwen2.5-32B_verus-lemma

#python distill.py sft \
#       data=lemma-si \
#       output_dir=/n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/qwen2.5-32B_opus-distilled_si1-lemma \
#       language=dafny

#python distill.py sft \
#       data=lemma-verus \
#       output_dir=/n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/qwen2.5-32B_verus-lemma \
#       language=verus

python distill.py sft \
    data=verus_fixer \
    output_dir=/n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/qwen2.5-32B_opus-verus-distilled_si1-fixer \
    language=verus

# Sanity-check run: train verus fixer on Verus-Bench ground truths themselves
# Dataset produced with
#   python extract.py task=verus_fixer data=verusbench_fixer_train output_prefix=verus_sanitycheck_fixer
#python distill.py sft \
#       data=verus_sanitycheck \
#       output_dir=/n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/qwen2.5-32B_verus-fixer-sanitycheck \
#       language=verus \
#       num_train_epochs=10
