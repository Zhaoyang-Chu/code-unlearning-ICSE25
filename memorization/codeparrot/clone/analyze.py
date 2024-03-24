'''
Analyze memroization
'''

import os
import logging
import json
from tqdm import tqdm
import multiprocessing
import argparse


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to load")
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")
    
    parser.add_argument('--generation_strategy', type=str, default="npg", choices=["npg", "tdg", "pcg", "tsg"], help="The strategy to generate outputs from large code models")
    parser.add_argument('--start', type=int, default=0, help="start id")
    parser.add_argument('--end', type=int, default=20000, help="end id")

    return parser.parse_args()


def process_file(log_path):
    memorization = {}
    data = None
    with open(log_path, 'r') as f:
        logger.info("Analyzing {}".format(log_path))
        lines = f.readlines()
        for line in lines:
            if 'duplicate lines with fingerprint' in line:
                # store the previous data
                if data:
                    if data['extract'] > 0 and data['train'] > 0:
                        # only store memorized data
                        memorization[fingerprint] = data

                # update the information
                suffix = line.split('fingerprint ')[1]
                fingerprint = suffix.split(' in')[0]
                prefix = line.split(' duplicate')[0]
                len = int(prefix.split('Found ')[1])

                try:
                    data = memorization[fingerprint]
                except:
                    data = {'train': 0, 'extract': 0, "len": len}

            # analyzing the clone information
            if 'clone' in line:
                data['train'] += 1
            if 'extract' in line:
                data['extract'] += 1
    
    return memorization


def merge_memorizations(memorizations):
    memorization = {}
    for m in memorizations:
        for fingerprint in m:
            try:
                memorization[fingerprint]['train'] += m[fingerprint]['train']
                memorization[fingerprint]['extract'] += m[fingerprint]['extract']
            except:
                memorization[fingerprint] = m[fingerprint]
    return memorization


if __name__ == '__main__':
    root_dir = './'

    # from which model the results to be analyzed
    args = parse_arguments()
    generated_folder = '{}-temp{}-len{}-k{}-{}'.format(args.model, args.temperature, args.seq_len, args.top_k, args.generation_strategy)

    memorizations = []
    previous_memorizations = {}

    log_dir = os.path.join(root_dir, 'log/save/', generated_folder, "{}-{}".format(args.start, args.end - 1))
    logs = [os.path.join(log_dir, log) for log in os.listdir(log_dir) if log.endswith('.log')]

    stats_path = os.path.join(log_dir, 'stats')
    os.makedirs(stats_path, exist_ok=True)
    memorization_path = os.path.join(stats_path, 'memorization.json')
    
    # multiprocessing
    with multiprocessing.Pool() as pool:
        for result in pool.map(process_file, logs):
            memorizations.append(result)

    # merge the memorizations
    memorization = merge_memorizations(memorizations)
    # save as json
    logger.info("Saving memorization to {}".format(memorization_path))
    with open(memorization_path, 'w') as f:
        json.dump(memorization, f, indent=4)

    # merge with previous memorizations
    previous_memorizations = merge_memorizations([previous_memorizations, memorization])

    # count the number of unique fingerprints
    count = 0
    for fingerprint in previous_memorizations:
        count += 1
    logger.info("Number of unique fingerprints: {}".format(count))

    # store as json
    with open(memorization_path, 'w') as f:
        json.dump(memorization, f, indent=4)

    '''Analyze the memorization'''
    # length distribution
    lens = {}
    count = 0
    for fingerprint in memorization:
        count += 1
        data = memorization[fingerprint]
        try:
            lens[data['len']] += 1
        except:
            lens[data['len']] = 1

    # number of unique fingerprints
    logger.info("Number of unique fingerprints: {}".format(count))
