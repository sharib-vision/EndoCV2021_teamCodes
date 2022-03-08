from detectron2.utils.logger import setup_logger
setup_logger ()
import work
from detectron2.engine import launch, default_argument_parser


if __name__ == "__main__":
    args = default_argument_parser ().parse_args (args=[])
    args.config_file = "./detectron2_repo/configs/myconfig/my_mask_rcnn_R_50_FPN_3x.yaml"
    # args.eval_only=True
    print ("Command Line Args:", args)
    launch (
        work.main,
        num_gpus_per_machine=2,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,)
    )

