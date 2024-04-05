import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["MKL_DEBUG_CPU_TYPE"] = '5'
import warnings
warnings.filterwarnings("ignore")
import os
import sys
from pathlib import Path
import argparse
from mmengine.config import Config, DictAction
import multiprocessing

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from finagent.registry import DOWNLOADER
from finagent.utils.misc import update_data_root

class StockDownloaderProcess(multiprocessing.Process):
    def __init__(self, stocks, downloader):
        super().__init__()
        self.stocks = stocks
        self.downloader = downloader
    def run(self):
        self.downloader.download(self.stocks)

def parse_args():
    parser = argparse.ArgumentParser(description='Download News')
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "downloader", "tools", "fmp_economic_exp.py"), help="download datasets config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--if_remove", action="store_true", default=False)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is None:
        args.cfg_options = dict()
    if args.root is not None:
        args.cfg_options["root"] = args.root
    if args.workdir is not None:
        args.cfg_options["workdir"] = args.workdir
    if args.tag is not None:
        args.cfg_options["tag"] = args.tag
    cfg.merge_from_dict(args.cfg_options)

    update_data_root(cfg, root=args.root)

    exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
    if args.if_remove is None:
        args.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {exp_path}? ") == 'y')
    if args.if_remove:
        import shutil
        shutil.rmtree(exp_path, ignore_errors=True)
        print(f"| Arguments Remove work_dir: {exp_path}")
    else:
        print(f"| Arguments Keep work_dir: {exp_path}")
    os.makedirs(exp_path, exist_ok=True)

    downloader = DOWNLOADER.build(cfg.downloader)

    print(f"| Check Downloading {cfg.tag}...")
    stocks = downloader.check_download()
    print(f"| Check Downloading {cfg.tag} Done! Failed: {len(stocks)} / {len(downloader.stocks)}")

    batch_size = cfg.batch_size if cfg.batch_size < len(stocks) else 5
    batch_size = min(len(stocks), batch_size)

    processes = []
    remaining_stocks = downloader.check_download()

    while remaining_stocks:
        batch = remaining_stocks[:batch_size]
        remaining_stocks = remaining_stocks[batch_size:]

        process = StockDownloaderProcess(batch, downloader)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    main()