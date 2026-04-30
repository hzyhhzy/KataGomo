import argparse
import os
import torch
import modelconfigs
from model_pytorch import Model

def main():
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('required')
    optional = parser.add_argument_group('optional')
    optional.add_argument('-h','--help',action='help',default=argparse.SUPPRESS)
    required.add_argument('-model-kind',required=True)
    required.add_argument('-pos-len',type=int,required=True)
    required.add_argument('-path',required=True)
    args = vars(parser.parse_args())

    model_kind = args['model_kind']
    pos_len = args['pos_len']
    path = args['path']

    model_config = modelconfigs.config_of_name[model_kind]
    model = Model(model_config,pos_len)
    model.initialize()
    model.to('cpu')

    state_dict = {
        'model': model.state_dict(),
        'config': model_config,
    }

    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)
    torch.save(state_dict,path)

if __name__ == '__main__':
    main()

