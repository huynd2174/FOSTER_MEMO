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
        # MEMO+FOSTER: các tham số điều khiển
        # - memo_freeze_until: đóng băng tầng nông (MEMO-style) ở nhánh convnet mới
        # - kd_temperature, kd_alpha: nhiệt độ và trọng số KD cho giai đoạn nén (compression)
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

        # MEMO: Freeze shallow layers ở nhánh convnet mới (chỉ áp dụng cho incremental tasks)
        #  - CIFAR: 'stage_2' (conv_1_3x3, bn_1, stage_1, stage_2)
        #  - ImageNet: 'layer2' (conv1, bn1, layer1, layer2)
        if self._cur_task >= 1:
            try:
                latest = self._network_module_ptr.convnets[-1]
                if hasattr(latest, 'freeze_until'):
                    latest.freeze_until(self.memo_freeze_until)
            except Exception as e:
                logging.info(f"freeze_until skipped: {e}")

        if self._cur_task > 0:
            # Khóa nhánh nền đầu tiên và oldfc (vai trò teacher trong FOSTER)
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

        device_ids = [d.index for d in self._multiple_gpus if isinstance(d, torch.device) and d.type == 'cuda' and d.index is not None and d.index < torch.cuda.device_count()]
        if len(device_ids) > 1:
            self._network = nn.DataParallel(self._network, device_ids=device_ids)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        # Unwrap only if actually wrapped
        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module

    def _feature_boosting(self, train_loader, test_loader, optimizer, scheduler):
        # FOSTER Boosting (CE + KD): dùng triển khai của FOSTER
        #  - logits = F_current + F_new (FOSTERNet ghép đặc trưng đa nhánh)
        #  - loss = CE(logits) + CE(fe_logits) + lambda_okd * KD_T(logits[:,:K_old], old_logits)
        super()._feature_boosting(train_loader, test_loader, optimizer, scheduler)

    def _feature_compression(self, train_loader, test_loader):
        # FOSTER Compression (KD + CE): nén teacher -> student
        #  - KD với kd_temperature để học "soft targets" từ teacher
        #  - CE với labels thật để neo student, cải thiện ổn định
        #  - loss = kd_alpha * KD + CE
        self._snet = FOSTERNet(self.args['convnet_type'], False)
        self._snet.update_fc(self._total_classes)
        device_ids = [d.index for d in self._multiple_gpus if isinstance(d, torch.device) and d.type == 'cuda' and d.index is not None and d.index < torch.cuda.device_count()]
        if len(device_ids) > 1:
            self._snet = nn.DataParallel(self._snet, device_ids=device_ids)
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

        if isinstance(self._snet, nn.DataParallel):
            self._snet = self._snet.module
        self._snet.eval()
        # Swap to student
        self._snet_module_ptr = self._snet
        self._network = self._snet
        self._network_module_ptr = self._snet_module_ptr

