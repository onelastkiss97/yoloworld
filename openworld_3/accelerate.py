"""Fixed-vocabulary YOLO-World deployment: cache text, FP16, fuse BN, CUDA Graph."""
import argparse
import copy
import json
import platform
import time
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.cnn import fuse_conv_bn
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.runner import load_checkpoint
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
CONFIG=ROOT/'openworld_2/runs/neck_5shot_85544/finetune.py'
CHECKPOINT=ROOT/'openworld_2/runs/neck_5shot_85544/best_coco_fixedwing_ap50_iter_400.pth'
TEXTS=['a multirotor drone','a helicopter','a fixed-wing aircraft']


class FixedVocabulary(torch.nn.Module):
    def __init__(self,model,text_features):
        super().__init__()
        self.image_encoder=model.backbone.image_model
        self.neck=model.neck
        self.head=model.bbox_head
        self.head.num_classes=3
        self.register_buffer('text_features',text_features.detach())

    def forward(self,x):
        features=self.image_encoder(x)
        features=self.neck(features,self.text_features)
        return self.head.forward(features,self.text_features,None)


class CapturedForward:
    def __init__(self,model,x):
        self.input=torch.empty_like(x)
        self.input.copy_(x)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(10):
                model(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output=model(self.input)

    def __call__(self,x):
        self.input.copy_(x)
        self.graph.replay()
        return self.output


class AlreadyClassified(torch.nn.Module):
    def forward(self,x,text):
        return x


class FixedAttention(torch.nn.Module):
    """Fixed text guidance expressed as a grouped 1x1 convolution."""
    def __init__(self,block,text):
        super().__init__()
        self.embed_conv=block.embed_conv
        self.project_conv=block.project_conv
        self.num_heads=block.num_heads
        self.classes=text.shape[1]
        guide=block.guide_fc(text).reshape(self.classes,block.num_heads,block.head_channels)
        weight=guide.permute(1,0,2).reshape(-1,block.head_channels,1,1)/(block.head_channels**0.5)
        self.guide_conv=torch.nn.Conv2d(block.num_heads*block.head_channels,
            block.num_heads*self.classes,1,groups=block.num_heads,bias=True).to(weight.device)
        self.guide_conv.weight.data.copy_(weight)
        self.guide_conv.bias.data.copy_(block.bias.repeat_interleave(self.classes))
        self.scale=block.scale

    def forward(self,x,guide):
        b,_,h,w=x.shape
        embed=self.embed_conv(x) if self.embed_conv is not None else x
        attention=self.guide_conv(embed).reshape(b,self.num_heads,self.classes,h,w).max(2)[0]
        attention=attention.sigmoid()*self.scale
        return (self.project_conv(x).reshape(b,self.num_heads,-1,h,w)*attention.unsqueeze(2)).reshape(b,-1,h,w)


def fold_attention(model):
    from yolo_world.models.layers.yolo_bricks import MaxSigmoidAttnBlock
    for parent in list(model.neck.modules()):
        for name,child in list(parent.named_children()):
            if isinstance(child,MaxSigmoidAttnBlock):
                setattr(parent,name,FixedAttention(child,model.text_features))
    return model


def fold_classifiers(model):
    """Compose 1x1 visual projection, frozen BN and fixed-text classification exactly."""
    head=model.head.head_module
    text=torch.nn.functional.normalize(model.text_features[0].float(),dim=-1)
    for i in range(3):
        conv=head.cls_preds[i][-1]
        contrast=head.cls_contrasts[i]
        bn=contrast.norm
        assert isinstance(bn,torch.nn.BatchNorm2d) and not bn.training
        assert conv.kernel_size==(1,1) and conv.groups==1
        linear=text*contrast.logit_scale.float().exp()
        alpha=bn.weight.float()/torch.sqrt(bn.running_var.float()+bn.eps)
        beta=bn.bias.float()-bn.running_mean.float()*alpha
        weight=(linear*alpha)@conv.weight.float()[:,:,0,0]
        bias=(linear*alpha)@conv.bias.float()+linear@beta+contrast.bias.float()
        merged=torch.nn.Conv2d(conv.in_channels,len(text),1,bias=True).to(weight.device)
        merged.weight.data.copy_(weight[:,:,None,None])
        merged.bias.data.copy_(bias)
        head.cls_preds[i][-1]=merged
        head.cls_contrasts[i]=AlreadyClassified()
    return model


def build():
    register_all_modules()
    cfg=Config.fromfile(str(CONFIG))
    model=MODELS.build(cfg.model).cuda().eval()
    load_checkpoint(model,str(CHECKPOINT),map_location='cpu')
    model.bbox_head.num_classes=3
    with torch.inference_mode():
        embeddings,_=model.backbone.forward_text([TEXTS])
    return cfg,model,FixedVocabulary(model,embeddings).eval()


def pipeline_for(cfg):
    transforms=copy.deepcopy(cfg.test_dataloader.dataset.pipeline)
    transforms=[p for p in transforms if p['type'] not in ('LoadAnnotations','LoadText')]
    transforms[0]=dict(type='mmdet.LoadImageFromNDArray')
    return Compose(transforms)


def prepare(model,pipeline,item):
    iid,path,img=item
    sample=pipeline(dict(img=img.copy(),img_id=iid,img_path=path,texts=TEXTS))
    return model.data_preprocessor(pseudo_collate([sample]),training=False)


def statistics(seconds):
    a=np.array(seconds)*1000
    return dict(mean_ms=float(a.mean()),median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a,95)),p99_ms=float(np.percentile(a,99)),
                max_ms=float(a.max()),fps=float(1000/a.mean()),samples=len(a),
                mean_under_10ms=bool(a.mean()<10),p99_under_10ms=bool(np.percentile(a,99)<10),
                all_observed_under_10ms=bool(a.max()<10))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--variant',default='fused_fp16_graph',choices=['cached_fp32','cached_fp16','fused_fp16','fused_fp16_graph','fused_fp16_channelslast_graph','reparam_fp16_channelslast_graph','reparam_neck_fp16_channelslast_graph'])
    parser.add_argument('--single',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    assert torch.cuda.is_available()
    cfg,model,cached=build()
    pipeline=pipeline_for(cfg)
    ann=json.loads(Path(cfg.test_evaluator.ann_file).read_text())
    image_root=Path(cfg.test_dataloader.dataset.dataset.data_root)/cfg.test_dataloader.dataset.dataset.data_prefix.img
    rng=np.random.default_rng(20260912)
    indices=rng.choice(len(ann['images']),100,replace=False)
    samples=[(ann['images'][int(i)]['id'],str(image_root/ann['images'][int(i)]['file_name']),
              mmcv.imread(str(image_root/ann['images'][int(i)]['file_name']))) for i in indices]
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'artifacts').mkdir(exist_ok=True)
    with torch.inference_mode():
        first=prepare(model,pipeline,samples[0])
        reference=model._forward(first['inputs'],first['data_samples'])
        outputs=cached(first['inputs'])
        for xs,ys in zip(reference,outputs):
            for x,y in zip(xs,ys):
                torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
        print('Cached-text FP32 equivalence passed',flush=True)
        variants=['cached_fp32','cached_fp16','fused_fp16','fused_fp16_graph'] if not (args.evaluate or args.single) else [args.variant]
        results={}
        for variant in variants:
            engine=copy.deepcopy(cached)
            if variant.startswith('reparam'):
                engine=fold_classifiers(engine)
                if 'neck' in variant:
                    engine=fold_attention(engine)
                composed=engine(first['inputs'])
                for xs,ys in zip(reference,composed):
                    for x,y in zip(xs,ys):
                        torch.testing.assert_close(x,y,atol=1e-4,rtol=1e-4)
                print('Composed classifier FP32 equivalence passed',flush=True)
            if variant.startswith(('fused','reparam')):
                engine=fuse_conv_bn(engine)
            if 'fp16' in variant:
                engine=engine.half()
            memory_format=torch.channels_last if 'channelslast' in variant else torch.contiguous_format
            engine=engine.to(memory_format=memory_format)
            dtype=next(engine.parameters()).dtype
            tensor=first['inputs'].to(dtype=dtype,memory_format=memory_format)
            forward=CapturedForward(engine,tensor) if variant.endswith('graph') else engine
            for _ in range(30):
                forward(tensor)
            torch.cuda.synchronize()
            if args.evaluate:
                evaluate(cfg,model,pipeline,engine,forward,dtype,ann,image_root,variant)
                break
            forward_times=[]
            e2e_times=[]
            for rnd in range(3):
                for sample in samples:
                    batch=prepare(model,pipeline,sample)
                    x=batch['inputs'].to(dtype=dtype,memory_format=memory_format)
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    forward(x)
                    torch.cuda.synchronize()
                    forward_times.append(time.perf_counter()-start)
                    start=time.perf_counter()
                    batch=prepare(model,pipeline,sample)
                    outs=forward(batch['inputs'].to(dtype=dtype,memory_format=memory_format))
                    # Decode and NMS in FP32, retaining the original model's evaluation settings.
                    outs=[[v.float() for v in group] for group in outs]
                    preds=engine.head.predict_by_feat(*outs,batch_img_metas=[s.metainfo for s in batch['data_samples']],rescale=True)
                    [p.cpu() for p in preds]
                    torch.cuda.synchronize()
                    e2e_times.append(time.perf_counter()-start)
            results[variant]=dict(forward=statistics(forward_times),end_to_end=statistics(e2e_times),
                                  forward_seconds=forward_times,end_to_end_seconds=e2e_times)
            print(variant,json.dumps({k:results[variant][k] for k in ['forward','end_to_end']}),flush=True)
            report=dict(gpu=torch.cuda.get_device_name(),hostname=platform.node(),torch=torch.__version__,
                        checkpoint=str(CHECKPOINT),resolution=1280,batch=1,images=100,rounds=3,
                        text_prompts=TEXTS,cached_text_equivalence='passed 1e-5',
                        scope='Forward: GPU tensor to raw head outputs. E2E: decoded RAM BGR to CPU detections, excluding disk/JPEG/model loading. CUDA synchronized wall time.',
                        results=results)
            report_name=f'latency_{variant}.json' if args.single else 'latency.json'
            (HERE/'results'/report_name).write_text(json.dumps(report,indent=2))
            if variant.startswith(('fused','reparam')):
                torch.save(dict(state_dict=engine.state_dict(),model_config=dict(cfg.model),
                                prompts=TEXTS,source_checkpoint=str(CHECKPOINT),precision='fp16',fused=True),
                           HERE/f'artifacts/{variant}.pth')
            del forward,engine
            torch.cuda.empty_cache()


def evaluate(cfg,model,pipeline,engine,forward,dtype,ann,image_root,variant):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    predictions=[]
    for index,item in enumerate(ann['images']):
        path=image_root/item['file_name']
        batch=prepare(model,pipeline,(item['id'],str(path),mmcv.imread(str(path))))
        outs=forward(batch['inputs'].to(dtype))
        outs=[[v.float() for v in group] for group in outs]
        pred=engine.head.predict_by_feat(*outs,batch_img_metas=[s.metainfo for s in batch['data_samples']],rescale=True)[0].cpu()
        for box,score,label in zip(pred.bboxes.tolist(),pred.scores.tolist(),pred.labels.tolist()):
            x1,y1,x2,y2=box
            predictions.append(dict(image_id=item['id'],category_id=label+1,bbox=[x1,y1,x2-x1,y2-y1],score=score))
        if (index+1)%200==0:
            print('Evaluated',index+1,flush=True)
    (HERE/f'results/{variant}_predictions.json').write_text(json.dumps(predictions))
    score_predictions(cfg,predictions,len(ann['images']),variant)


def score_predictions(cfg,predictions,image_count,variant):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO(cfg.test_evaluator.ann_file)
    dt=gt.loadRes(predictions)
    ev=COCOeval(gt,dt,'bbox')
    ev.params.maxDets=list(cfg.test_evaluator.get('proposal_nums',(100,300,1000)))
    ev.evaluate(); ev.accumulate(); ev.summarize()
    rows={}
    for k,cid in enumerate(ev.params.catIds):
        p=ev.eval['precision'][:,:,k,0,-1]
        mean=lambda a:float(a[a>=0].mean()) if (a>=0).any() else None
        rows[gt.loadCats([cid])[0]['name']]=dict(mAP=mean(p),AP50=mean(p[0]),AP75=mean(p[5]),
            mAP_s=mean(ev.eval['precision'][:,:,k,1,-1]),
            mAP_m=mean(ev.eval['precision'][:,:,k,2,-1]),
            mAP_l=mean(ev.eval['precision'][:,:,k,3,-1]))
    report=dict(variant=variant,test_images=image_count,metrics=rows,source_checkpoint=str(CHECKPOINT),max_dets=ev.params.maxDets)
    (HERE/f'results/{variant}_accuracy.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
