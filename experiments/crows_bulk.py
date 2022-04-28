import argparse
import json
import os

import transformers

from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.model import models
from bias_bench.util import _is_generative, _is_self_debias

parser = argparse.ArgumentParser(description="Runs SEAT benchmark for debiased models.")
parser.add_argument(
    "--model_dir",
    action="store",
    type=str,
    help="Path to directory where the model is stored. Example: /home/mullick/scratch/debiasing-language-models/models/downstream/race/debsize_4000/lm_0.02/model_files/",
)
parser.add_argument(
    "--output_dir",
    action="store",
    type=str,
    help="Path to directory to save the model in. Example: home/mullick/scratch/.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    model = "BertForMaskedLM"
    model_name_or_path = "bert-base-uncased"
    load_path = args.model_dir
    persistent_dir = args.output_dir
    thisdir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(thisdir, ".."))

    debias_type = args.model_dir.split("/")[-5]
    debias_size = args.model_dir.split("/")[-4].split("_")[-1]
    lm_fraction = args.model_dir.split("/")[-3].split("_")[-1]

    experiment_id = f"{debias_type}_{debias_size}_{lm_fraction}"

    print("Running SEAT benchmark:")
    print(f" - experiment_id: {experiment_id}")
    print(f" - persistent_dir: {persistent_dir}")
    print(f" - model: {model}")
    print(f" - model_name_or_path: {model_name_or_path}")
    print(f" - load_path: {load_path}")

    # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
    print("Loading model.")
    model = getattr(models, model)(load_path or model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    print("Running tests.")
    for bias_type in ["gender", "race", "religion"]:
        print(f"Test: {bias_type}")
        runner = CrowSPairsRunner(
            model=model,
            tokenizer=tokenizer,
            input_file=f"{data_dir}/data/crows/crows_pairs_anonymized.csv",
            bias_type=bias_type,
            is_generative=_is_generative(model),  # Affects model scoring.
            is_self_debias=_is_self_debias(model),
        )
        results = runner()

        os.makedirs(f"{persistent_dir}/crows", exist_ok=True)
        with open(f"{persistent_dir}/crows/{experiment_id}_{bias_type}-test.json", "w") as f:
            json.dump(results, f)
