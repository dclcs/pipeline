
# config,py

def add_common_parser(parser):
    parser.add_argument('--dataset_path', type=str,
                        default='/home/server/dataset/new_dataset/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--deform_in_feats', type=int, default=3)
    parser.add_argument('--deform_hidden_size', type=int, default=192)
    parser.add_argument('--deform_out_feats', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    

    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epoches', type=int, default=3000)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--checkpoint', type=str, default='geoloss')
    parser.add_argument('--category', type=str, default='Chair')
    parser.add_argument('--model_version', type=str, default='model_box')
    return parser


def add_test_parser(parser):
    parser = add_common_parser(parser)
    return parser


def add_dataprocess_paser(parser):
    parser = add_common_parser(parser)
    return parser


def add_train_paser(parser):
    parser = add_common_parser(parser)
    parser.add_argument('--is_pretrained', action='store_true', default=False)
    parser.add_argument('--img_lr', type=float, default=3e-4)
    parser.add_argument('--deform_lr', type=float, default=3e-4)
    parser.add_argument('--pts_w', type=float, default=3000.)
    parser.add_argument('--lap_w', type=float, default=2000.)
    parser.add_argument('--mov_w', type=float, default=100.)
    parser.add_argument('--norm_w', type=float, default=100.)
    parser.add_argument('--front_epoch', type=int, default=10)
    parser.add_argument('--edge_w', type=float, default=300.)
    parser.add_argument('--conn_w', type=float, default=150.)
    parser.add_argument('--clip_value', type=float, default=0.05)
    return parser
