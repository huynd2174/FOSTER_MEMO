import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.foster import FOSTER, _KD_loss
from utils.inc_net import FOSTERNet
from utils.toolkit import count_parameters, tensor2numpy


class MEMO_FOSTER(FOSTER):
    def __init__(self, args):
        super().__init__(args)
        # Extra knobs for MEMO freezing and compression KD
        self.memo_freeze_until = args.get('memo_freeze_until', 'layer2')
        self.kd_temperature = args.get('kd_temperature', args.get('T', 2))
        self.kd_alpha = args.get('kd_alpha', 1.0)

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self._cur_task > 1:
            self._network = self._snet
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Freeze shallow layers on the newest convnet branch (MEMO-style) only for incremental tasks
        if self._cur_task >= 1:
            try:
                latest = self._network_module_ptr.convnets[-1]
                if hasattr(latest, 'freeze_until'):
                    latest.freeze_until(self.memo_freeze_until)
            except Exception as e:
                logging.info(f"freeze_until skipped: {e}")

        if self._cur_task > 0:
            # Also freeze the first branch and old fc as in FOSTER teacher
            for p in self._network.convnets[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source='train',
            mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"],
                                       shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"],
                                      shuffle=False, num_workers=self.args["num_workers"]) 

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        # Unwrap only if actually wrapped
        if hasattr(self._network, "module"):
            self._network = self._network.module

    def _feature_boosting(self, train_loader, test_loader, optimizer, scheduler):
        # Same as FOSTER, but conceptually treat logits = F_current + F_new (already done in FOSTERNet)
        super()._feature_boosting(train_loader, test_loader, optimizer, scheduler)

    def _feature_compression(self, train_loader, test_loader):
        # Override to add KD alpha/temperature knobs if provided
        self._snet = FOSTERNet(self.args['convnet_type'], False)
        self._snet.update_fc(self._total_classes)
        if len(self._multiple_gpus) > 1:
            self._snet = nn.DataParallel(self._snet, self._multiple_gpus)
        if hasattr(self._snet, "module"):
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet
        self._snet.to(self._device)
        self._snet_module_ptr.convnets[0].load_state_dict(self._network_module_ptr.convnets[0].state_dict())
        self._snet_module_ptr.copy_fc(self._network_module_ptr.oldfc)

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._snet.parameters()),
                              lr=self.args["lr"], momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["compression_epochs"]) 

        self._network.eval()
        prog_bar = tqdm(range(self.args["compression_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._snet.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                stud_logits = self._snet(inputs)["logits"]
                with torch.no_grad():
                    teach_out = self._network(inputs)
                    teach_logits = teach_out["logits"]
                # KD + optional CE
                loss_kd = _KD_loss(stud_logits, teach_logits, self.kd_temperature)
                loss_ce = F.cross_entropy(stud_logits, targets)
                loss = self.kd_alpha * loss_kd + loss_ce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(stud_logits[:targets.shape[0]], dim=1)
                correct += preds.eq(targets.expand_as(preds)).sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = 'SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["compression_epochs"], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["compression_epochs"], losses/len(train_loader),  train_acc)
            prog_bar.set_description(info)
            logging.info(info)

        if len(self._multiple_gpus) > 1:
            self._snet = self._snet.module
        self._snet.eval()
        # Swap to student
        self._snet_module_ptr = self._snet
        self._network = self._snet
        self._network_module_ptr = self._snet_module_ptr

