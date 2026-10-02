"""ONNX -> TensorRT FP16; actual raw-head timing and unchanged COCO evaluation."""
import argparse
import copy
import json
import hashlib
import platform
import time
from pathlib import Path
import mmcv
import numpy as np
import torch
import tensorrt as trt
from mmcv.cnn import fuse_conv_bn

from openworld_3.accelerate import (HERE, CHECKPOINT, TEXTS, build, pipeline_for,
                                  prepare, statistics, CapturedForward, evaluate, fold_classifiers)


class ExportRaw(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model

    def forward(self,x):
        cls,boxes=self.model(x)
        return tuple(cls)+tuple(boxes)


class TensorRTEngine:
    def __init__(self,path):
        self.logger=trt.Logger(trt.Logger.WARNING)
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        assert self.engine is not None
        self.context=self.engine.create_execution_context()
        self.outputs={}
        for i in range(self.engine.num_io_tensors):
            name=self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name)==trt.TensorIOMode.OUTPUT:
                dtype=torch.from_numpy(np.empty((),dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
                self.outputs[name]=torch.empty(tuple(self.engine.get_tensor_shape(name)),dtype=dtype,device='cuda')
                self.context.set_tensor_address(name,self.outputs[name].data_ptr())

    def __call__(self,x):
        assert tuple(x.shape)==(1,3,1280,1280) and x.dtype==torch.float32 and x.is_contiguous()
        self.context.set_tensor_address('images',x.data_ptr())
        ok=self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError('TensorRT execution failed')
        return ([self.outputs[f'cls_{i}'] for i in range(3)],
                [self.outputs[f'box_{i}'] for i in range(3)])


def build_engine(model):
    artifact=HERE/'artifacts'
    artifact.mkdir(exist_ok=True)
    onnx_path=artifact/'fixed_vocab_1280.onnx'
    engine_path=artifact/'fixed_vocab_1280_fp16.engine'
    manifest_path=artifact/'engine_manifest.json'
    manifest=dict(checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                  prompts=TEXTS,input_shape=[1,3,1280,1280],tensorrt=trt.__version__,
                  gpu=torch.cuda.get_device_name(),classifier_folded=True)
    if engine_path.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text())!=manifest:
            raise RuntimeError('Engine manifest mismatch: move the old engine/ONNX aside before rebuilding for a changed model/device.')
        print('Using existing engine',engine_path,flush=True)
        return engine_path
    import onnx
    x=torch.zeros(1,3,1280,1280,device='cuda')
    if not onnx_path.exists():
        torch.onnx.export(ExportRaw(model),x,str(onnx_path),input_names=['images'],
                          output_names=[f'cls_{i}' for i in range(3)]+[f'box_{i}' for i in range(3)],
                          opset_version=13,do_constant_folding=True)
        onnx.checker.check_model(str(onnx_path))
    logger=trt.Logger(trt.Logger.INFO)
    builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,2<<30)
    config.builder_optimization_level=3
    timing_path=artifact/'timing.cache'
    cache=config.create_timing_cache(timing_path.read_bytes() if timing_path.exists() else b'')
    config.set_timing_cache(cache,False)
    plan=builder.build_serialized_network(network,config)
    if plan is None:
        raise RuntimeError('TensorRT build failed')
    engine_path.write_bytes(bytes(plan))
    manifest_path.write_text(json.dumps(manifest,indent=2))
    timing_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    print('Saved',engine_path,flush=True)
    return engine_path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluate',action='store_true')
    p.add_argument('--stress',type=int,default=0)
    p.add_argument('--image',type=Path,help='Run one decoded image through the saved engine and original postprocessing')
    p.add_argument('--output',type=Path,default=HERE/'results/inference.json')
    p.add_argument('--latency-output',type=Path,default=HERE/'results/tensorrt_latency.json')
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    cfg,original,cached=build()
    cached=fuse_conv_bn(fold_classifiers(cached))
    pipeline=pipeline_for(cfg)
    ann=json.loads(Path(cfg.test_evaluator.ann_file).read_text())
    image_root=Path(cfg.test_dataloader.dataset.dataset.data_root)/cfg.test_dataloader.dataset.dataset.data_prefix.img
    (HERE/'results').mkdir(exist_ok=True)
    with torch.inference_mode():
        path=build_engine(cached)
        engine=TensorRTEngine(path)
        first_item=ann['images'][0]
        first=prepare(original,pipeline,(first_item['id'],str(image_root/first_item['file_name']),
                                        mmcv.imread(str(image_root/first_item['file_name']))))
        x=first['inputs'].contiguous()
        actual=engine(x)
        expected=cached(x)
        torch.cuda.synchronize()
        discrepancies=[]
        for ag,eg in zip(actual,expected):
            for a,e in zip(ag,eg):
                discrepancies.append(dict(max_absolute=float((a-e).abs().max()),mean_absolute=float((a-e).abs().mean())))
                assert torch.isfinite(a).all()
        print('Raw FP16 deviations',discrepancies,flush=True)
        graph=CapturedForward(engine,x)
        if args.image:
            image=mmcv.imread(str(args.image))
            if image is None:
                raise FileNotFoundError(args.image)
            data=prepare(original,pipeline,(0,str(args.image),image))
            outputs=graph(data['inputs'].contiguous())
            pred=cached.head.predict_by_feat(*outputs,batch_img_metas=[s.metainfo for s in data['data_samples']],rescale=True)[0].cpu()
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(dict(image=str(args.image),prompts=TEXTS,
                boxes_xyxy=pred.bboxes.tolist(),scores=pred.scores.tolist(),labels=pred.labels.tolist()),indent=2))
            print('Saved predictions',args.output,flush=True)
            return
        if args.evaluate:
            evaluate(cfg,original,pipeline,cached,graph,torch.float32,ann,image_root,'tensorrt_fp16_graph')
            return
        rng=np.random.default_rng(20260912)
        indices=rng.choice(len(ann['images']),100,replace=False)
        images=[]
        for idx in indices:
            im=ann['images'][int(idx)]
            file=image_root/im['file_name']
            images.append((im['id'],str(file),mmcv.imread(str(file))))
        results={}
        for name,forward in [('tensorrt_fp16',engine),('tensorrt_fp16_graph',graph)]:
            def e2e(item):
                data=prepare(original,pipeline,item)
                outputs=forward(data['inputs'].contiguous())
                predictions=cached.head.predict_by_feat(*outputs,batch_img_metas=[s.metainfo for s in data['data_samples']],rescale=True)
                return [pred.cpu() for pred in predictions]
            for i in range(30):
                forward(x)
                e2e(images[i%100])
            torch.cuda.synchronize()
            ft=[]; et=[]
            for _ in range(3):
                for item in images:
                    data=prepare(original,pipeline,item)
                    tensor=data['inputs'].contiguous()
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    forward(tensor)
                    torch.cuda.synchronize()
                    ft.append(time.perf_counter()-start)
                    start=time.perf_counter()
                    e2e(item)
                    torch.cuda.synchronize()
                    et.append(time.perf_counter()-start)
            results[name]=dict(forward=statistics(ft),end_to_end=statistics(et),forward_seconds=ft,end_to_end_seconds=et)
            print(name,json.dumps({k:results[name][k] for k in ('forward','end_to_end')}),flush=True)
        stress={}
        if args.stress:
            # Continuous warmed inference; no preprocessing/I/O in this diagnostic stress loop.
            values=[]
            for _ in range(args.stress):
                torch.cuda.synchronize()
                start=time.perf_counter()
                graph(x)
                torch.cuda.synchronize()
                values.append(time.perf_counter()-start)
            stress=dict(forward=statistics(values),seconds=values)
        report=dict(gpu=torch.cuda.get_device_name(),hostname=platform.node(),tensorrt=trt.__version__,torch=torch.__version__,
                    engine=str(path),checkpoint=str(CHECKPOINT),resolution=1280,batch=1,
                    engine_precision='FP16 tactics, FP32 input/output',cached_text_prompts=TEXTS,
                    images=100,rounds=3,warmup=30,raw_fp16_deviations=discrepancies,
                    results=results,continuous_stress=stress,
                    timing='CUDA-synchronized per-image wall time; pure forward from preprocessed FP32 GPU input. E2E decoded BGR RAM input through original decode/NMS to CPU boxes; excludes JPEG/disk/loading.')
        args.latency_output.parent.mkdir(parents=True,exist_ok=True)
        args.latency_output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
