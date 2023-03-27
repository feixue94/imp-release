import argparse
import yaml


def str2bool(v):
    return v.lower() in ("true", "1")


# Parse command line arguments.
parser = argparse.ArgumentParser(description='dump eval data.')
parser.add_argument('--config_path', type=str, default='configs/yfcc.yaml')


def get_dumper(name):
    mod = __import__('dumper.{}'.format(name), fromlist=[''])
    return getattr(mod, name)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, yaml.Loader)

    dataset = get_dumper(config['data_name'])(config)

    dataset.initialize()
    if config['extractor']['extract']:
        dataset.dump_feature()
    dataset.format_dump_data()
