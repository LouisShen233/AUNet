import sys
base_dir='/home/lyuyu/AUNet'
sys.path.append(base_dir)


from utils.argparser import parse_args, load_yaml
from utils.utils import DirManager
from test import test_all_model, test_update
from test_real import test_update_real
# from test_real_temp import test_update_real


def run():
    args = parse_args()
    cfg = load_yaml(args)
    dir_manager = DirManager(cfg)


    if cfg["TEST"].get("ENABLE", True):
        cfg["TEST"]["LOAD"] = True
        # test_all_model(cfg, dir_manager)
        # test_update(cfg, dir_manager)
        test_update_real(cfg, dir_manager)


if __name__ == '__main__':
    run()