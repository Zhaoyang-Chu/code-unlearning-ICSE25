import os
import subprocess
import concurrent.futures
import logging
import argparse
import re
import hashlib


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to load")
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")
    parser.add_argument('--start', type=int, default=0, help="start id")
    parser.add_argument('--end', type=int, default=20000, help="end id")

    return parser.parse_args()


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return (output, error)


def save_output(output, file_path):
    with open(file_path, "w") as f:
        f.write(output)


if __name__ == '__main__':
    root_dir = './'
    args = parse_arguments()

    generated_folder = '{}-temp{}-len{}-k{}'.format(args.model, args.temperature, args.seq_len, args.top_k)

    # path to the training data & the clone detection tool
    tool_path = os.path.join(root_dir, 'clone/simian-2.5.10.jar')
    data_dir = os.path.join(root_dir, 'clone/save/codeparrot/codeparrot-clean-train')

    file_path = os.path.join(root_dir, 'extract/results/{}/all_{}-{}'.format(generated_folder, args.start, args.end - 1))
    print(os.path.abspath(file_path))
    assert os.path.exists(file_path), "File {} does not exist".format(file_path)

    # create log dir
    log_dir = os.path.join(root_dir, 'log/save/', generated_folder, "{}-{}".format(args.start, args.end - 1))
    logger.info(f'[save dict]: {log_dir}')
    os.makedirs(log_dir, exist_ok=True)

    # build the commands to run
    commands = []
    # the number of training data split is 53
    start_i = 0
    for i in range(53):
        data_path = os.path.join(data_dir, '{}'.format(start_i + i))
        assert os.path.exists(data_path), "File {} does not exist".format(data_path)
        commands.append(['java', '-jar', tool_path, data_path, file_path])

    logger.info("Batch {}-{} started".format(args.start, args.end - 1))
    # launch the commands in parallel using a process pool
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_command, command) for command in commands]

    # wait for all futures to complete and capture the output
    outputs = [future.result() for future in futures]

    # decode the output and error into strings (assuming utf-8 encoding)
    output_strs = [output.decode("utf-8") for output, error in outputs]
    error_strs = [error.decode("utf-8") for output, error in outputs]

    # save output to files in parallel
    logger.info("Saving output to {}".format(log_dir))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(save_output, output_str, 
            os.path.join(log_dir, "{}.log".format(start_i + i))) for i, output_str in enumerate(output_strs)]
        
    # free memory
    del outputs
    del output_strs
    del error_strs

    logger.info("Batch {}-{} finished".format(args.start, args.end - 1))
