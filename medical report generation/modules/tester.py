import os
from abc import abstractmethod

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import time
import torch
import pandas as pd
from numpy import inf
from torch.nn import DataParallel


class BaseTester(object):
    def __init__(self, model, args):
        self.args = args

        self.device = torch.device(args.gpu_index if torch.cuda.is_available() else "cpu")
        


        self.model = model.to(self.device)



        self.checkpoint_dir = args.save_dir

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _test_epoch(self, epoch):
        raise NotImplementedError

    def test(self):
        self._test_epoch(1)

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        # print(checkpoint)
        # self.start_epoch = checkpoint['epoch'] + 1
        # self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])

        # print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
 



class Tester(BaseTester):
    def __init__(self, model,args, val_dataloader,test_dataloader):
        super(Tester, self).__init__(model, args)


        
        self.model = self.model.to(self.device)

        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
    def _test_epoch(self, epoch):

        # resnet152 = self.model.visual_extractor  # 假设你的模型中ResNet152部分的命名是resnet152
        # # 保存ResNet152部分作为checkpoint
        # torch.save(resnet152.state_dict(), 'resnet152_checkpoint.pth')

        self.model.eval()
        with torch.no_grad():
            idlist,val_gts, val_res = [], [],[]
            for batch_idx, (images_id, images, reports_ids, reports_masks,backup) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                

                backup = backup.to(self.device)

                output = self.model(images,backup, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                # print(reports_ids[:, 1:])
                # print(images_id)
                idlist.append(images_id)
                val_res.extend(reports)
                val_gts.extend(ground_truths)

                # if len(idlist)>1:
                #     break



            # 将数组元素以字符串形式连接起来
            id_str = '\n'.join([str(item) for sublist in idlist for item in sublist])
            
            res_str = '\n'.join(val_res)
            gts_str = '\n'.join(val_gts)

            # 保存结果到文本文件
            with open('/data/gaowh/work/24process/use_R2gen_rewrite/prospect/szyk/val_idlist.txt', 'w') as file:
                file.write(id_str)
            with open('/data/gaowh/work/24process/use_R2gen_rewrite/prospect/szyk/val_res.txt', 'w') as file:
                file.write(res_str)

            with open('/data/gaowh/work/24process/use_R2gen_rewrite/prospect/szyk/val_gts.txt', 'w') as file:
                file.write(gts_str)



