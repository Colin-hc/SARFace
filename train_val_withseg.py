import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.core import LightningModule
from torch.nn import CrossEntropyLoss
import evaluate_utils
import head
import net
import numpy as np
import utils
from models.swiftformerUnet import create_swiftformer_sr_model, SwiftFormer_width, SwiftFormer_depth
from parsing_for_vgg import face_split
from models.reinforce import REINFORCE, PolicyNetwork
import random
from torch.distributions import Bernoulli
torch.autograd.set_detect_anomaly(True)
from torchvision.utils import save_image
indices_to_keep = [0, 2, 3,4,5,6,7,8,9,10,11,12,13,14,17,19,20,21,22]
iter = 0
# torch.cuda.synchronize()
class Trainer(LightningModule):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__()
        self.save_hyperparameters()  # sets self.hparams
        self.automatic_optimization = False
        self.class_num = utils.get_num_class(self.hparams)
        print('classnum: {}'.format(self.class_num))
        self.iter = 0
        self.seg_model = create_swiftformer_sr_model(SwiftFormer_depth['l1'], SwiftFormer_width['l1'], SwiftFormer_depth['l1'], SwiftFormer_width['l1'])
        self.model = net.build_model(model_name=self.hparams.arch)
        self.head = head.build_head(head_type=self.hparams.head,
                                     embedding_size=512,
                                     class_num=self.class_num,
                                     m=self.hparams.m,
                                     h=self.hparams.h,
                                     t_alpha=self.hparams.t_alpha,
                                     s=self.hparams.s,
                                     )

        self.cross_entropy_loss = CrossEntropyLoss()

        # Degment Selection Policy
        self.policy = PolicyNetwork(12)
        self.agent = REINFORCE(12, self.policy)
        self.gamma = 0.99

        if self.hparams.start_from_model_statedict:
            # ckpt_model = torch.load('weight/epoch=11-step=73548.ckpt')
            # ckpt = torch.load(self.hparams.start_from_model_statedict)
            # ckpt = torch.load('/home/cdzs/Code/AdaFace-master/experiments/ir50_policyuseingseg_adaface_08-15_1/epoch=2-step=35542.ckpt')
            # ckpt = torch.load('weight/epoch=11-step=73548.ckpt')
            # ckpt = torch.load('/home/cdzs/Code/AdaFace-master/experiments/ir50_vgg_newseg_10-18_2/epoch=9-step=235217.ckpt')

            ckpt['state_dict']= {key: value for key, value in ckpt['state_dict'].items() if 'seg_model' not in key}
            self.model.load_state_dict({key.replace('model.', ''):val
                                        for key,val in ckpt['state_dict'].items() if 'model.' in key})
            self.head.load_state_dict({key.replace('head.', ''):val
                                        for key,val in ckpt['state_dict'].items() if 'head.' in key})
            self.policy.load_state_dict({key.replace('policy.', ''):val
                                        for key,val in ckpt['state_dict'].items() if 'policy.' in key})
            self.seg_model.load_state_dict(torch.load('weight/swiftl1_fp_128_only.pth'))
    def get_current_lr(self):
        scheduler = None
        if scheduler is None:
            try:
                # pytorch lightning >= 1.8
                scheduler = self.trainer.lr_scheduler_configs[0].scheduler
            except:
                pass

        if scheduler is None:
            # pytorch lightning <=1.7
            try:
                scheduler = self.trainer.lr_schedulers[0]['scheduler']
            except:
                pass

        if scheduler is None:
            raise ValueError('lr calculation not successful')

        if isinstance(scheduler, lr_scheduler._LRScheduler):
            lr = scheduler.get_last_lr()[0]
        else:
            lr = scheduler.get_epoch_values(self.current_epoch)[0]
        return lr


    def forward(self, images, labels, eval=False):
        # 获取分割信息，获得112*112单通道图。
        img = torch.nn.functional.interpolate(images,(128,128),mode='bilinear')
        self.seg_model = self.seg_model.eval()
        # if next(self.seg_model.parameters()).device != next(self.agent.policy.parameters()).device:
        #     self.agent.policy.to(next(self.seg_model.parameters()).device)
        seg = torch.argmax(self.seg_model(img),dim=1)
        seg = face_split(seg)
        # 分割信息，变成b*24*112*112 的特征，每一个112*112代表对应一个部位的mask
        one_hot = torch.zeros((img.size(0),24,128,128)).to(seg.device)
        one_hot = one_hot.scatter(1, seg.unsqueeze(1), 1)
        seg = torch.nn.functional.interpolate(one_hot,(112,112), mode='nearest')

        if eval:
            self.model.eval()
            self.head.eval()

        embeddings, norms = self.model(images)
        cos_thetas, softmax = self.head(embeddings, norms, labels)

        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            labels = labels.clone()
            labels[bad_grad.squeeze(-1)] = -100 # ignore_index
        return cos_thetas, norms, embeddings, labels, softmax, seg

    def get_epsilon(self, batch_idx):
        """
        Calculate epsilon based on the current batch index.
        Epsilon decreases linearly from 0.5 to 0.05 over the first 1000 batches, then stays at 0.05.
        """
        if batch_idx < 15000:
            return 0.5 - (0.45 * batch_idx / 15000)
        else:
            return 0.05
        # if batch_idx < 60000:
        #     return 0.5 - (0.45 * batch_idx / 15000)
        # else:
        #     return 0.05


    def batch_episode(self, seg, images, labels,eval=False, batch_idx=0, use_policy_only=False):
        b = seg.shape[0]
        if eval:
            self.model.eval()
            self.head.eval()
        idx = torch.arange(b, device=seg.device)
        # combine = torch.cat((seg, images), dim=1)
        if use_policy_only:
            # Directly use actions generated by the policy network without randomness
            seg_img = torch.cat([images, seg], dim=1)
            action, log_prob = self.agent.select_actions(seg_img)
        else:
            self.iter = self.iter + 1
            # Use epsilon-greedy strategy for action selection
            epsilon = 0.2 # self.get_epsilon(self.iter)
            # action, log_prob = self.epsilon_greedy_action(seg, epsilon)
            seg_img = torch.cat([images, seg], dim=1)
            action, log_prob = self.agent.select_actions(seg_img, eps=epsilon, add_noise=True)

        # if random.random() < 0.5:
        #     action = torch.ones_like(action)
        ones = torch.ones_like(action).to(action.device)
        action = torch.cat([ones, action], dim=1)
        select_region = seg * action.unsqueeze(-1).unsqueeze(-1)
        prcessed_images = select_region.sum(dim=1,keepdim=True).repeat(1, 3, 1, 1) * images
        embeddings_mask, norms_mask = self.model(prcessed_images)
        cos_thetas_mask, softmax_mask = self.head(embeddings_mask, norms_mask, labels)
        # print(softmax_mask.shape, labels.shape)
        mask_acc = softmax_mask[idx, labels]

        if isinstance(cos_thetas_mask, tuple):
            cos_thetas_mask, bad_grad = cos_thetas_mask
            labels = labels.clone()
            labels[bad_grad.squeeze(-1)] = -100 # ignore_index
        return prcessed_images, log_prob, mask_acc, idx, cos_thetas_mask, labels, action, softmax_mask

    def epsilon_greedy_action(self, seg, epsilon):
        """
        Select actions using epsilon-greedy strategy.
        """
        if random.random() < epsilon:
            # Select a random action
            action = torch.randint(0, 2, (seg.size(0), 24), device=seg.device, dtype=torch.float32)
            m = Bernoulli(action)
            action = m.sample()
            log_prob = m.log_prob(action)
            # log_prob = None  # No log_prob for random action
        else:
            # Select the best action based on the policy network
            action, log_prob = self.agent.select_actions(seg)

        # action = torch.randint(0, 2, (seg.size(0), 24), device=seg.device, dtype=torch.float32)
        # m = Bernoulli(action)
        # action = m.sample()
        # log_prob = m.log_prob(action)
        return action, log_prob


    def training_step(self, batch, batch_idx):
        images, labels = batch
        op_model, op_policy = self.optimizers()
        cos_thetas, norms, embeddings, labels, softmax, seg = self.forward(images, labels, eval=True)

        pre_processed_images, log_probs, mask_accs, idx, cos_thetas_mask, _, action, mask_softmax = self.batch_episode(seg, images, labels, eval=True,batch_idx=batch_idx)
        acc = softmax[idx, labels]
        # pred_classes = torch.argmax(softmax, dim=1)
        # correct_pred = (pred_classes == labels)
        # incorrect_pred = ~correct_pred
        #
        # err_enhance_factor = 5
        # corr_diminish_factor = 0.1
        # adjusted_rewards = torch.ones_like(acc).to(acc.device)
        # adjusted_rewards[incorrect_pred] = err_enhance_factor
        # adjusted_rewards[correct_pred] = corr_diminish_factor
        # 计算预测准确率
        # predicted_classes = torch.argmax(softmax, dim=1)
        # acc = (predicted_classes == labels).sum()/images.size(0)
        # select_region = seg * action.unsqueeze(-1).unsqueeze(-1)
        # region_size = select_region.sum(dim=(1,2,3))
        # sparsity_reward = 1.0 / (region_size + 1e-5)
        #
        higher = mask_accs > acc
        # sample_weights = torch.where(higher, torch.tensor(100.0).to(higher.device), torch.tensor(0.1).to(higher.device))
        #
        # rewards = (sparsity_reward * 100 + mask_accs - acc) * sample_weights.to(acc.device)# * adjusted_rewards
        pred_softmax = torch.argmax(softmax, dim=1)  # 全图预测类别
        pred_mask_softmax = torch.argmax(mask_softmax, dim=1)  # 掩码图预测类别

        # 比较全图和掩码的预测结果与真实标签
        correct_softmax = pred_softmax == labels  # 全图预测是否正确
        correct_mask_softmax = pred_mask_softmax == labels  # 掩码图预测是否正确

        # 根据逻辑分配奖励
        rewards = torch.zeros_like(acc)  # 初始化奖励为 0

        # 全图错误且掩码图正确 -> 奖励为 1
        rewards[(~correct_softmax) & correct_mask_softmax] = 1.0

        # 全图正确且掩码图错误 -> 奖励为 -1
        rewards[correct_softmax & (~correct_mask_softmax)] = -1.0

        # 计算稀疏性奖励
        select_region = seg * action.unsqueeze(-1).unsqueeze(-1)
        region_size = select_region.sum(dim=(1, 2, 3))
        sparsity_reward = 1.0 / (region_size + 1e-5)

        # 最终奖励 = 稀疏性奖励 + 分类奖励
        final_rewards = rewards + sparsity_reward * 0.05
        loss_policy = - (log_probs * final_rewards.unsqueeze(-1)).sum(dim=1).mean()
        entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
        loss_policy -= 0.01 * entropy
        prcessed_images = select_region.sum(dim=1, keepdim=True).repeat(1, 3, 1, 1) * images


        better_images = prcessed_images[higher]
        for i, img_tensor in enumerate(better_images):
            save_path = '/media/cdzs/94fe807a-5ba6-4f85-8ea3-f00bb3b97f4e/home/cdzs/data/select_better/'+ str(batch_idx) + '_' + str(i) + '.jpg'
            save_image(img_tensor[[2,1,0],:,:].cpu()*0.5+0.5, save_path)

        # 优化决策网络
        op_policy.zero_grad()
        self.manual_backward(loss_policy)
        # if (batch_idx+1) % 2 == 0 :
        op_policy.step()

        # 优化识别网络
        #
        _, _, _, _, cos_thetas_mask, labels, _, _ = self.batch_episode(seg, images, labels, eval=False, use_policy_only=True)
        loss_rec = self.cross_entropy_loss(cos_thetas_mask, labels)
        op_model.zero_grad()
        self.manual_backward(loss_rec)
        if (batch_idx+1) % 100 == 0:
            op_model.step()

        # print(loss_train.item())
        train_loss = loss_policy  # + loss_rec
        lr = self.get_current_lr()
        # log
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True)
        self.log('train_loss_policy', loss_policy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_loss_rec', loss_rec, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('seg_num', action.sum(dim=1).float().mean(), on_step=True, prog_bar=True)

        return train_loss

    def training_epoch_end(self, outputs):
        return None

    def validation_step(self, batch, batch_idx):
        images, labels, dataname, image_index = batch
        #添加分割人连操作
        # 获取分割信息，获得112*112单通道图。
        img = torch.nn.functional.interpolate(images, (128, 128), mode='bilinear')
        self.seg_model = self.seg_model.eval()
        seg = torch.argmax(self.seg_model(img), dim=1)
        seg = face_split(seg)
        # 分割信息，变成24*112*112 的特征，每一个112*112代表对应一个部位的mask
        one_hot = torch.zeros((images.size(0), 24, 128, 128)).to(seg.device)
        one_hot = one_hot.scatter(1, seg.unsqueeze(1), 1)
        seg = torch.nn.functional.interpolate(one_hot, (112, 112), mode='nearest')
        seg_img = torch.cat((images, seg), dim=1)

        action, log_prob = self.agent.select_actions(seg_img, add_noise=False)
        pre = torch.ones_like(action).to(action.device)
        action = torch.cat([pre, action], dim=1)
        select_region = seg * action.unsqueeze(-1).unsqueeze(-1)
        prcessed_images = select_region.sum(dim=1,keepdim=True).repeat(1, 3, 1, 1) * images

        embeddings, norms = self.model(images)

        fliped_images = torch.flip(images, dims=[3])
        flipped_embeddings, flipped_norms = self.model(fliped_images)
        stacked_embeddings = torch.stack([embeddings, flipped_embeddings], dim=0)
        stacked_norms = torch.stack([norms, flipped_norms], dim=0)
        embeddings, norms = utils.fuse_features_with_norm(stacked_embeddings, stacked_norms)

        if self.hparams.distributed_backend == 'ddp':
            # to save gpu memory
            return {
                'output': embeddings.to('cpu'),
                'norm': norms.to('cpu'),
                'target': labels.to('cpu'),
                'dataname': dataname.to('cpu'),
                'image_index': image_index.to('cpu')
            }
        else:
            # dp requires the tensor to be cuda
            return {
                'output': embeddings,
                'norm': norms,
                'target': labels,
                'dataname': dataname,
                'image_index': image_index
            }

    def validation_epoch_end(self, outputs):

        all_output_tensor, all_norm_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)

        dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
        val_logs = {}
        for dataname_idx in all_dataname_tensor.unique():
            dataname = idx_to_dataname[dataname_idx.item()]
            # per dataset evaluation
            embeddings = all_output_tensor[all_dataname_tensor == dataname_idx].to('cpu').numpy()
            labels = all_target_tensor[all_dataname_tensor == dataname_idx].to('cpu').numpy()
            issame = labels[0::2]
            tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(embeddings, issame, nrof_folds=10)
            acc, best_threshold = accuracy.mean(), best_thresholds.mean()

            num_val_samples = len(embeddings)
            val_logs[f'{dataname}_val_acc'] = acc
            val_logs[f'{dataname}_best_threshold'] = best_threshold
            val_logs[f'{dataname}_num_val_samples'] = num_val_samples

        val_logs['val_acc'] = np.mean([
            val_logs[f'{dataname}_val_acc'] for dataname in dataname_to_idx.keys() if f'{dataname}_val_acc' in val_logs
        ])
        val_logs['epoch'] = self.current_epoch

        for k, v in val_logs.items():
            # self.log(name=k, value=v, rank_zero_only=True)
            self.log(name=k, value=v)

        return None

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):

        all_output_tensor, all_norm_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)

        dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
        test_logs = {}
        for dataname_idx in all_dataname_tensor.unique():
            dataname = idx_to_dataname[dataname_idx.item()]
            # per dataset evaluation
            embeddings = all_output_tensor[all_dataname_tensor == dataname_idx].to('cpu').numpy()
            labels = all_target_tensor[all_dataname_tensor == dataname_idx].to('cpu').numpy()
            issame = labels[0::2]
            tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(embeddings, issame, nrof_folds=10)
            acc, best_threshold = accuracy.mean(), best_thresholds.mean()

            num_test_samples = len(embeddings)
            test_logs[f'{dataname}_test_acc'] = acc
            test_logs[f'{dataname}_test_best_threshold'] = best_threshold
            test_logs[f'{dataname}_num_test_samples'] = num_test_samples

        test_logs['test_acc'] = np.mean([
            test_logs[f'{dataname}_test_acc'] for dataname in dataname_to_idx.keys()
            if f'{dataname}_test_acc' in test_logs
        ])
        test_logs['epoch'] = self.current_epoch

        for k, v in test_logs.items():
            # self.log(name=k, value=v, rank_zero_only=True)
            self.log(name=k, value=v)

        return None

    def gather_outputs(self, outputs):
        if self.hparams.distributed_backend == 'ddp':
            # gather outputs across gpu
            outputs_list = []
            _outputs_list = utils.all_gather(outputs)
            for _outputs in _outputs_list:
                outputs_list.extend(_outputs)
        else:
            outputs_list = outputs

        # if self.trainer.is_global_zero:
        all_output_tensor = torch.cat([out['output'] for out in outputs_list], axis=0).to('cpu')
        all_norm_tensor = torch.cat([out['norm'] for out in outputs_list], axis=0).to('cpu')
        all_target_tensor = torch.cat([out['target'] for out in outputs_list], axis=0).to('cpu')
        all_dataname_tensor = torch.cat([out['dataname'] for out in outputs_list], axis=0).to('cpu')
        all_image_index = torch.cat([out['image_index'] for out in outputs_list], axis=0).to('cpu')

        # get rid of duplicate index outputs
        unique_dict = {}
        for _out, _nor, _tar, _dat, _idx in zip(all_output_tensor, all_norm_tensor, all_target_tensor,
                                                all_dataname_tensor, all_image_index):
            unique_dict[_idx.item()] = {'output': _out, 'norm': _nor, 'target': _tar, 'dataname': _dat}
        unique_keys = sorted(unique_dict.keys())
        all_output_tensor = torch.stack([unique_dict[key]['output'] for key in unique_keys], axis=0)
        all_norm_tensor = torch.stack([unique_dict[key]['norm'] for key in unique_keys], axis=0)
        all_target_tensor = torch.stack([unique_dict[key]['target'] for key in unique_keys], axis=0)
        all_dataname_tensor = torch.stack([unique_dict[key]['dataname'] for key in unique_keys], axis=0)

        return all_output_tensor, all_norm_tensor, all_target_tensor, all_dataname_tensor

    def configure_optimizers(self):
        # Optimizer for Sgement Selection Policy
        optimizer_policy = optim.Adam(self.agent.policy.parameters(), lr=1e-4)
        paras_policy = self.agent.policy.parameters()
        # paras_only_bn, paras_wo_bn = self.separate_bn_paras(self.model)
        paras_wo_bn, paras_only_bn = self.split_parameters(self.model)

        optimizer = optim.SGD([{
            'params': paras_wo_bn + [self.head.kernel],
            'weight_decay': 5e-4
        }, {
            'params': paras_only_bn
        }],
                                lr=self.hparams.lr,
                                momentum=self.hparams.momentum)

        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=self.hparams.lr_milestones,
                                             gamma=self.hparams.lr_gamma)

        return [optimizer,optimizer_policy], [scheduler]

    def split_parameters(self, module):
        params_decay = []
        params_no_decay = []
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
        return params_decay, params_no_decay

