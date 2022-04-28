import argparse
import json
import os

import transformers

from bias_bench.benchmark.seat import SEATRunner
from bias_bench.model import models

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
    n_samples = 100000
    parametric = True
    model = "BertModel"
    model_name_or_path = "bert-base-uncased"
    gender_tests = ["sent-weat6", "sent-weat6b", "sent-weat7", "sent-weat7b", "sent-weat8", "sent-weat8b"]
    race_tests = ["sent-angry_black_woman_stereotype", "sent-angry_black_woman_stereotype_b", "sent-weat3", "sent-weat3b", "sent-weat4", "sent-weat5", "sent-weat5b"]
    religion_tests = ["sent-religion1", "sent-religion1b", "sent-religion2", "sent-religion2b"]
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
    print(f" - gender_tests: {gender_tests}")
    print(f" - race_tests: {race_tests}")
    print(f" - religion_tests: {religion_tests}")
    print(f" - n_samples: {n_samples}")
    print(f" - parametric: {parametric}")
    print(f" - model: {model}")
    print(f" - model_name_or_path: {model_name_or_path}")
    print(f" - load_path: {load_path}")

    # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
    print("Loading model.")
    model = getattr(models, model)(load_path or model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    print("Running tests.")
    for debias_type, tests in {"gender-test": gender_tests, "race-test": race_tests, "religion-test": religion_tests}.items():
        print(f"Test: {debias_type}")
        runner = SEATRunner(
            experiment_id=experiment_id,
            tests=tests,
            data_dir=f"{data_dir}/data/seat",
            n_samples=n_samples,
            parametric=parametric,
            model=model,
            tokenizer=tokenizer,
        )
        results = runner()

        os.makedirs(f"{persistent_dir}/seat", exist_ok=True)
        with open(f"{persistent_dir}/seat/{experiment_id}_{debias_type}.json", "w") as f:
            json.dump(results, f)
