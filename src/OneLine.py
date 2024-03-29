import argparse
import os
import yaml


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--job_config", type=str, required=True)
    parser.add_argument("--test", type=int, default=0)

    args = parser.parse_args()

    # Read in YAML file
    yaml_path = os.path.join(".", "job_config", args.job_config)
    with open(yaml_path, "r") as f:
        job_config = yaml.safe_load(f)

    snippet_path = job_config['snippet_path']
    finished_path = job_config['finished_path']
    delete_finished = job_config["delete_snippets"]
    output_path = job_config['output_path']
    year = job_config['year']
    run_batches = job_config['run_batches']
    batch_size = job_config['batch_size']
    column_model_list = job_config['column_model_list']
    images_count = job_config['images_count']
    sub_name = args.job_config.split('.')[0]

    for column_dict in column_model_list:
        for column_name in column_dict:
            model_list = column_dict[column_name]
            input_path = os.path.join(snippet_path, column_name)
            os.makedirs(output_path, exist_ok=True)

            batch_folders = os.listdir(input_path)
            for batch_folder in batch_folders:

                batch_input_folder = os.path.join(input_path, batch_folder)
                batch_number = batch_folder.split("_")[-1].split(".")[0]

                for model in model_list:
                    model_output_name = model + '_' + column_name + "_" + batch_number + '.csv'
                    model_output_path = os.path.join(output_path, str(year), sub_name, column_name, model)
                    os.makedirs(model_output_path, exist_ok=True)
                    full_model_output = os.path.join(model_output_path, model_output_name)

                    command = "sbatch run_batch.sh " + " ".join((batch_input_folder, model + ".hdf5", full_model_output,
                                                                 str(delete_finished)))
                    os.system(command)
                    print(command)


if __name__ == "__main__":
    main()
