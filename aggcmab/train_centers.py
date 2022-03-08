import sys, json, os, time, argparse
import os.path as osp
from datetime import datetime
import operator
from tqdm import trange
import numpy as np
import torch
from models.get_model import get_arch
from utils.sam import SAM
from utils.model_saving_loading import load_model
from utils.dice_loss import BinaryDiceLoss, BinaryAreaCEDiceLoss
from utils.get_loaders import get_train_val_seg_loaders, modify_loader
from utils.model_saving_loading import save_model, str2bool
from utils.reproducibility import set_seeds
from skimage.filters import threshold_otsu as threshold
from utils.evaluation import dice_score


from torch.optim.lr_scheduler import CosineAnnealingLR

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--csv_train', type=str, default='data/train_1.csv', help='path to training data csv')
parser.add_argument('--model_name', type=str, default='fpnet_resnet18_W', help='architecture')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--sampling', type=str, default='class', help='sampling mode (instance, class, sqrt, prog)')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer choice')
parser.add_argument('--loss_fn', type=str, default='bce', help='loss choice')
parser.add_argument('--min_lr', type=float, default=1e-8, help='learning rate')
parser.add_argument('--max_lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cycle_lens', type=str, default='20/25', help='cycling config (nr cycles/cycle len)')
parser.add_argument('--metric', type=str, default='dice', help='which metric to use for monitoring progress (loss/dice)')
parser.add_argument('--im_size', help='delimited list input, could be 512, or 512,640', type=str, default='512,640')
parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=False, help='avoid saving anything')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers to launch '
                                                               'for data loading tasks (handled by pytorch) [default: %(default)s]')
parser.add_argument('--n_checkpoints', type=int, default=1, help='nr of best checkpoints to keep (defaults to 3)')
parser.add_argument('--resume_path', type=str, default='no', help='path to folder with previous checkpoint')


def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'dice':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run_one_epoch(loader, model, criterion, optimizer=None, scheduler=None, assess=False):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here
    if train:
        model.train()
    else:
        model.eval()
    if assess:
        act = torch.sigmoid if n_classes == 1 else torch.nn.Softmax(dim=0)
        dice_scores, center_lst = [], []

    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for i_batch, (inputs, labels, centers) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            if isinstance(logits, tuple):  # wnet
                logits_aux, logits = logits
                loss_aux = criterion(logits_aux, labels.float())
                loss = loss_aux + criterion(logits.float(), labels.float())
            else:  # not wnet
                loss = criterion(logits, labels.float())

            loss *= 10

            if train:  # only in training mode
                loss.backward()
                if isinstance(optimizer, SAM):
                    # sam requires a second pass
                    optimizer.first_step(zero_grad=True)
                    logits = model(inputs)
                    if isinstance(logits, tuple):  # wnet
                        logits_aux, logits = logits
                        loss_aux = criterion(logits_aux, labels.float())
                        loss = loss_aux + criterion(logits.float(), labels.float())
                    else:  # not wnet
                        loss = criterion(logits, labels.float())
                    loss *= 10

                    loss.backward()  # for grad_acc_steps=0, this is just loss
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if assess:
                # evaluation
                for i in range(len(logits)):
                    prediction = act(logits[i]).detach().cpu().numpy()[-1]
                    target = labels[i].cpu().numpy()
                    # try:
                    #     thresh = threshold(prediction)
                    # except:
                    #     thresh = 0.5
                    thresh = 0.5
                    bin_pred = prediction > thresh
                    dice = dice_score(target.ravel(), bin_pred.ravel())
                    # print(target.shape, len(np.unique(target)), dice)
                    dice_scores.append(dice)
                    center_lst.append(centers[i])

            # Compute running loss
            running_loss += loss.item() * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss = running_loss / n_elems
            if train:
                t.set_postfix(tr_loss_lr="{:.4f}/{:.6f}".format(float(run_loss), get_lr(optimizer)))
            else:
                t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()
    if assess: return np.array(dice_scores), np.array(center_lst), run_loss # return np.mean(dice_scores), np.std(dice_scores), run_loss, center_lst
    return None, None, None


def train_one_cycle(train_loader, model, sampling, criterion, optimizer=None, scheduler=None, cycle=0):
    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]
    for epoch in range(cycle_len):
        print('Cycle {:d} | Epoch {:d}/{:d}'.format(cycle + 1, epoch + 1, cycle_len))
        # Modify sampling
        mod_loader = modify_loader(train_loader, sampling, epoch, n_eps=cycle_len)
        if epoch == cycle_len - 1:
            assess = True  # only get logits/labels on last cycle
        else:
            assess = False
        tr_dice_scores, tr_centers, tr_loss = \
            run_one_epoch(mod_loader, model, criterion, optimizer=optimizer,
                          scheduler=scheduler, assess=assess)

    return tr_dice_scores, tr_centers, tr_loss


def train_model(model, sampling, optimizer, criterion, train_loader, val_loader, scheduler, metric, exp_path, n_checkpoints):
    n_cycles = len(scheduler.cycle_lens)
    best_dice, best_cycle, best_models = 0, 0, []
    all_tr_losses, all_vl_losses, all_tr_dices, all_vl_dices = [], [], [], []

    is_better, best_monitoring_metric = compare_op(metric)

    for cycle in range(n_cycles):
        print('Cycle {:d}/{:d}'.format(cycle + 1, n_cycles))
        # prepare next cycle:
        # reset iteration counter
        scheduler.last_epoch = -1
        # update number of iterations
        scheduler.T_max = scheduler.cycle_lens[cycle] * len(train_loader)

        # train one cycle
        _, _, _ = train_one_cycle(train_loader, model, sampling, criterion, optimizer, scheduler, cycle)

        with torch.no_grad():
            tr_dice_scores, tr_centers, tr_loss = run_one_epoch(train_loader, model, criterion, assess=True)
            vl_dice_scores, vl_centers, vl_loss = run_one_epoch(val_loader, model, criterion, assess=True)

        centers = np.unique(tr_centers)
        tr_avg_performances, tr_std_performances, vl_avg_performances,  vl_std_performances= [], [], [], []
        for c in centers:
            d = tr_dice_scores[tr_centers==c]
            tr_avg_performances.append(np.mean(d[d!=-1]))
            tr_std_performances.append(np.std(d[d!=-1]))
            d = vl_dice_scores[vl_centers == c]
            vl_avg_performances.append(np.mean(d[d!=-1]))
            vl_std_performances.append(np.std(d[d!=-1]))

        # we exclude last center from performance evaluation since train/val split in a sequence is not really that easy
        tr_mean_dice, tr_std_dice = np.mean(tr_avg_performances[:-1]), np.std(tr_avg_performances[:-1])
        vl_mean_dice, vl_std_dice = np.mean(vl_avg_performances[:-1]), np.std(vl_std_performances[:-1])

        text_file = osp.join(exp_path, 'performance_cycle_{}.txt'.format(str(cycle + 1).zfill(2)))
        with open(text_file, 'w') as f:
            for j in range(len(centers)):
                print('Center {}: Train||Val DICE: {:.2f}+-{:.2f}||{:.2f} +-{:.2f}'.format(centers[j], 100*tr_avg_performances[j], 100*tr_std_performances[j],
                                                                                             100*vl_avg_performances[j], 100*vl_std_performances[j]), file=f)
                print('Center {}: Train||Val DICE: {:.2f}+-{:.2f}||{:.2f} +-{:.2f}'.format(centers[j], 100*tr_avg_performances[j], 100*tr_std_performances[j],
                                                                                             100*vl_avg_performances[j], 100*vl_std_performances[j]))
            print('\nTrain||Val Loss: {:.4f}||{:.4f}  -- Train||Val DICE: {:.2f}+-{:.2f}||{:.2f} +- {:.2f}'.format(
                tr_loss, vl_loss, 100 * tr_mean_dice, 100 * tr_std_dice, 100 * vl_mean_dice, 100 * vl_std_dice), file=f)
            print('\nTrain||Val Loss: {:.4f}||{:.4f}  -- Train||Val DICE: {:.2f}+-{:.2f}||{:.2f} +- {:.2f}'.format(
                tr_loss, vl_loss, 100 * tr_mean_dice, 100 * tr_std_dice, 100 * vl_mean_dice, 100 * vl_std_dice))

        # check if performance was better than anyone before and checkpoint if so
        if metric == 'loss':
            monitoring_metric = vl_loss
        elif metric == 'dice':
            monitoring_metric = vl_mean_dice


        all_tr_losses.append(tr_loss)
        all_vl_losses.append(vl_loss)
        all_tr_dices.append(tr_mean_dice)
        all_vl_dices.append(vl_mean_dice)

        if n_checkpoints == 1:  # only save best val model
            if is_better(monitoring_metric, best_monitoring_metric):
                print('Best {} attained. {:.2f} --> {:.2f}'.format(metric, 100 * best_monitoring_metric,
                                                                   100 * monitoring_metric))
                best_loss, best_dice, best_cycle = vl_loss, vl_mean_dice, cycle + 1
                best_monitoring_metric = monitoring_metric
                if exp_path is not None:
                    print(15 * '-', ' Checkpointing ', 15 * '-')
                    save_model(exp_path, model, optimizer)
            else:
                print('Best {} so far {:.2f} at cycle {:d}'.format(metric, 100 * best_monitoring_metric, best_cycle))

        else:  # SAVE n best - keep deleting worse ones
            from operator import itemgetter
            import shutil
            if exp_path is not None:
                s_name = 'cycle_{}_DICE_{:.2f}'.format(str(cycle + 1).zfill(2), 100 * vl_mean_dice)
                best_models.append([osp.join(exp_path, s_name), vl_mean_dice])

                if cycle < n_checkpoints:  # first n_checkpoints cycles save always
                    print('-------- Checkpointing to {}/ --------'.format(s_name))
                    save_model(osp.join(exp_path, s_name), model, optimizer)
                else:
                    worst_model = sorted(best_models, key=itemgetter(1), reverse=True)[-1][0]
                    if s_name != worst_model:  # this model was better than one of the best n_checkpoints models, remove that one
                        print('-------- Checkpointing to {}/ --------'.format(s_name))
                        save_model(osp.join(exp_path, s_name), model, optimizer)
                        print('----------- Deleting {}/ -----------'.format(worst_model.split('/')[-1]))
                        shutil.rmtree(worst_model)
                        best_models = sorted(best_models, key=itemgetter(1), reverse=True)[:n_checkpoints]

    del model
    torch.cuda.empty_cache()
    return best_dice, best_cycle, all_tr_losses, all_vl_losses, all_tr_dices, all_vl_dices


if __name__ == '__main__':

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    n_classes = 1

    # gather parser parameters
    model_name = args.model_name
    optimizer_choice = args.optimizer
    sampling = args.sampling
    max_lr, min_lr, bs = args.max_lr, args.min_lr, args.batch_size

    cycle_lens, metric = args.cycle_lens.split('/'), args.metric
    cycle_lens = list(map(int, cycle_lens))

    if len(cycle_lens) == 2:  # handles option of specifying cycles as pair (n_cycles, cycle_len)
        cycle_lens = cycle_lens[0] * [cycle_lens[1]]

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    do_not_save = str2bool(args.do_not_save)
    if do_not_save is False:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_path = osp.join('experiments', save_path)
        args.experiment_path = experiment_path
        os.makedirs(experiment_path, exist_ok=True)

        config_file_path = osp.join(experiment_path, 'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else:
        experiment_path = None
    n_checkpoints = args.n_checkpoints

    csv_train = args.csv_train
    csv_val = csv_train.replace('train', 'val')

    print('* Instantiating a {} model'.format(model_name))
    model, mean, std = get_arch(model_name, n_classes=n_classes)
    print('* Creating Dataloaders, batch size = {}, workers = {}'.format(bs, args.num_workers))
    train_loader, val_loader = get_train_val_seg_loaders(csv_path_train=csv_train, csv_path_val=csv_val, batch_size=bs,
                                                         tg_size=tg_size, mean=mean, std=std,
                                                         num_workers=args.num_workers)

    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=max_lr)
    elif optimizer_choice == 'sgd_sam':
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(model.parameters(), base_optimizer, lr=max_lr, momentum=0, weight_decay=0)
    else:
        sys.exit('please choose between adam or sgd optimizers')

    if args.resume_path != 'no':
        try:
            model, stats, optimizer_state_dict = load_model(model, args.resume_path, device=device, with_opt=True)
            optimizer.load_state_dict(optimizer_state_dict)
        except:
            sys.exit('Pretrained weights not compatible for this model')
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = max_lr
            param_group['initial_lr'] = max_lr

    scheduler = CosineAnnealingLR(optimizer, T_max=cycle_lens[0] * len(train_loader), eta_min=min_lr)

    setattr(optimizer, 'max_lr', max_lr)  # store it inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)

    if args.loss_fn == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.loss_fn == 'dice':
        criterion = BinaryDiceLoss()
    elif args.loss_fn == 'area_ce_dice':
        criterion = BinaryAreaCEDiceLoss()

    print('* Instantiating loss function', str(criterion))
    print('* Starting to train\n', '-' * 10)
    start = time.time()
    best_dice, best_cycle, all_tr_losses, all_vl_losses, all_tr_dices, all_vl_dices = \
                         train_model(model, sampling, optimizer, criterion, train_loader, val_loader, scheduler, metric, experiment_path, n_checkpoints)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))
    print('Test Dice: %f' % best_dice)
    print('Best Cycle: %d' % best_cycle)
    if do_not_save is False:
        with open(osp.join(experiment_path, 'val_metrics.txt'), 'w') as f:
            print('Best DICE = {:.2f}\nBest cycle = {}'.format(100 * best_dice, best_cycle), file=f)
            n_cycles = len(all_tr_losses )
            for i in range(n_cycles):
                print('Cycle {} ------------ DICE = {:.2f}/{:.2f}, Loss = {:.4f}/{:.4f}'.format(str(i+1).zfill(2),
                                                                                                100 * all_tr_dices[i],
                                                                                                100 * all_vl_dices[i],
                                                                                                all_tr_losses[i], all_vl_losses[i]), file=f)
            print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)
