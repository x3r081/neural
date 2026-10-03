"""Native CPU-only persistent decode identity and bounded microbenchmark.

No model/store read or inference. Four synthetic experts, original DLL loaded
as reference in the same process. Output AND materialized scratch bits compared;
capture byte copies verified before any timing result is accepted.
"""
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paths as NP

H, GU, GK, RB = 2880, 5760, 90, 1440
OFF_GS = (GU+H)*RB
RAW = OFF_GS + (GU+H)*GK
PACKED = OFF_GS + (GU+H)*46
VP = ctypes.c_void_p

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bind(path, persistent=False):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    lib.gptoss_experts_cap.argtypes = [ctypes.c_int]+[VP]*7+[ctypes.c_int, VP]
    lib.gptoss_experts_cap.restype = None
    lib.gptoss_set_fuse.argtypes = [ctypes.c_int]
    lib.gptoss_set_fuse.restype = None
    lib.gptoss_set_scale_layout.argtypes = [ctypes.c_int]
    lib.gptoss_set_scale_layout.restype = ctypes.c_int
    lib.gptoss_set_tuning.argtypes = [ctypes.c_int]*3
    lib.gptoss_set_tuning.restype = None
    lib.gptoss_set_tuning(0, 0, 0)
    lib.gptoss_set_fuse(1)
    if persistent:
        lib.gptoss_set_persistent.argtypes = [ctypes.c_int]
        lib.gptoss_set_persistent.restype = ctypes.c_int
        lib.gptoss_persistent_shutdown.restype = ctypes.c_int
        lib.gptoss_persistent_status.argtypes = [VP]
        lib.gptoss_persistent_status.restype = None
        lib.gptoss_persistent_fail_create_after.argtypes = [ctypes.c_int]
        lib.gptoss_persistent_fail_create_after.restype = ctypes.c_int
        lib.gptoss_team_probe_batch.argtypes = [ctypes.c_int]*4+[VP, VP]
        lib.gptoss_team_probe_batch.restype = ctypes.c_int
        lib.gptoss_team_fp_probe.argtypes = [ctypes.c_int]*2+[VP]
        lib.gptoss_team_fp_probe.restype = ctypes.c_int
    return lib

def status(lib):
    v=(ctypes.c_int*8)(); lib.gptoss_persistent_status(v)
    return {"raw":list(v), "math_jobs":(v[4]&0xffffffff)|((v[5]&0xffffffff)<<32),
            "fallback_calls":(v[6]&0xffffffff)|((v[7]&0xffffffff)<<32)}

def make_pool(seed=5931):
    rng=np.random.default_rng(seed)
    raw=np.empty((4, RAW), np.uint8)
    raw[:, :OFF_GS]=rng.integers(0,256,(4,OFF_GS),dtype=np.uint8)
    bases=rng.integers(113,120,(4, GU+H,1),dtype=np.uint8)
    deltas=rng.integers(0,12,(4, GU+H,GK),dtype=np.uint8)
    raw[:,OFF_GS:]=(bases+deltas).reshape(4,-1)
    packed=np.empty((4, PACKED),np.uint8)
    packed[:,:OFF_GS]=raw[:,:OFF_GS]
    s=packed[:,OFF_GS:].reshape(4,GU+H,46)
    s[:,:,0]=bases[:,:,0]; s[:,:,1:]=deltas[:,:,0::2]|(deltas[:,:,1::2]<<4)
    bgu=rng.normal(0,.08,(4,GU)).astype(np.float32)
    bdn=rng.normal(0,.08,(4,H)).astype(np.float32)
    inputs={"mixed":rng.normal(0,.8,H).astype(np.float32)}
    inputs["positive"]=np.abs(inputs["mixed"])
    inputs["negative"]=-inputs["positive"]
    inputs["signed_zero"]=np.zeros(H,np.float32); inputs["signed_zero"][1::2]=-0.
    return raw,packed,bgu,bdn,inputs

class Buffers:
    def __init__(self):
        self.scratch=np.zeros(H+4*(GU+3*H)+32,np.float32)
        self.out=np.empty(H,np.float32)
        self.capowners=[np.full(RAW+128,173,np.uint8) for _ in range(4)]
        self.caps=[]
        for owner in self.capowners:
            offset=(-owner.ctypes.data)%64
            self.caps.append(owner[offset:offset+RAW])

def run(lib,pool,bgu,bdn,x,weights,e,threads,capture_mask,buf,selection=None):
    sel=list(range(e)) if selection is None else selection
    slots=(VP*e)(*[pool[i].ctypes.data for i in sel])
    bg=(VP*e)(*[bgu[i].ctypes.data for i in sel])
    bd=(VP*e)(*[bdn[i].ctypes.data for i in sel])
    cp=(VP*e)(*[buf.caps[j].ctypes.data if capture_mask&(1<<j) else None for j in range(e)])
    lib.gptoss_experts_cap(e,slots,x.ctypes.data,bg,bd,weights.ctypes.data,
                         buf.out.ctypes.data,buf.scratch.ctypes.data,threads,cp)
    return slots, bg, bd, cp

def check_bits(a,b,label):
    if not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
        indices=np.flatnonzero(a.view(np.uint32)!=b.view(np.uint32))
        raise AssertionError(f"{label}: {len(indices)} mismatches, first{indices[0]}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--time",action="store_true")
    ap.add_argument("--iters",type=int,default=30)
    ap.add_argument("--repeats",type=int,default=4)
    args=ap.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    dll=ROOT/"gptoss_cpu_persistent.dll"; refpath=ROOT/"gptoss_cpu_cap2.dll"
    source=ROOT/"kernels/gptoss_cpu_persistent.c"
    original_hash=fingerprint(refpath)
    compiler=Path(NP.DEVKIT)/"gcc.exe"
    command=[str(compiler),"-O3","-march=native","-fopenmp","-shared","-o",str(dll),str(source)]
    buildenv=dict(os.environ,PATH=NP.DEVKIT+os.pathsep+os.environ.get("PATH",""))
    build=subprocess.run(command,capture_output=True,text=True,env=buildenv)
    if build.returncode: raise RuntimeError(build.stderr)
    os.environ.setdefault("GOMP_SPINCOUNT","20000000")
    dll_dir=os.add_dll_directory(NP.DEVKIT)
    ref,cand=bind(refpath),bind(dll,True)
    result={"label":"MEASURED CPU_ONLY SYNTHETIC_NOT_INFERENCE","pid":os.getpid(),
        "build_command":command,"build_stderr":build.stderr,"source_sha256":fingerprint(source),
        "included_cap2_sha256":fingerprint(ROOT/"kernels/gptoss_cpu_cap2.c"),
        "candidate_dll_sha256":fingerprint(dll),"reference_dll_sha256":original_hash,
        "fenv_scope":"default masked-exception nearest controls; nondefault candidate falls back",
        "pool_experts":4,"checks":0,"timings":[],"fp_controls":[],"errors":[]}
    try:
        assert cand.gptoss_set_persistent(1)==0
        # The same libgomp team supplies the original DLL and this probe.
        for n in (1,2,4,8):
            controls=[]
            for mode in (0,1):
                fp=(ctypes.c_uint*(2*n))()
                assert cand.gptoss_team_fp_probe(mode,n,fp)==0
                controls.append(list(fp))
            result["fp_controls"].append({"threads":n,"omp":controls[0],"persistent":controls[1]})
            for tid in range(n):
                assert controls[0][2*tid]&0xffc0 == controls[1][2*tid]&0xffc0
                assert controls[0][2*tid+1] == controls[1][2*tid+1]
        raw,packed,bgu,bdn,inputs=make_pool()
        a,b=Buffers(),Buffers()
        count_before=status(cand)
        caller_before=(ctypes.c_uint*2)()
        assert cand.gptoss_team_fp_probe(0,1,caller_before)==0
        for layout,pool in ((0,raw),(1,packed)):
            assert ref.gptoss_set_scale_layout(layout)==cand.gptoss_set_scale_layout(layout)==0
            for n in (1,2,4,8):
                for e in (1,2,3,4):
                    weights=np.arange(1,e+1,dtype=np.float32); weights/=weights.sum()
                    for name,x in inputs.items():
                        masks=range(1<<e) if name=="mixed" else (0,(1<<e)-1)
                        for mask in masks:
                            a.scratch.fill(91.); b.scratch.fill(91.)
                            run(ref,pool,bgu,bdn,x,weights,e,n,mask,a)
                            run(cand,pool,bgu,bdn,x,weights,e,n,mask,b)
                            assert np.isfinite(a.out).all() and np.isfinite(b.out).all()
                            check_bits(a.out,b.out,f"out layout{layout} n{n} e{e} {name} mask{mask}")
                            used=H+e*(GU+2*H)
                            check_bits(a.scratch[:used],b.scratch[:used],"materialized xe/xo/gu/he/ho/y")
                            for j in range(e):
                                if mask&(1<<j):
                                    assert np.array_equal(a.caps[j][:pool.shape[1]],pool[j])
                                    assert np.array_equal(b.caps[j][:pool.shape[1]],pool[j])
                            result["checks"]+=1
                print(f"PASS layout{layout} threads{n}: outputs, materialized scratch, allmixed capturemasks",flush=True)
        count_after=status(cand)
        result["identity_status_before"]=count_before; result["identity_status_after"]=count_after
        assert count_after["math_jobs"]-count_before["math_jobs"]==result["checks"]
        assert count_after["fallback_calls"]==count_before["fallback_calls"]
        caller_after=(ctypes.c_uint*2)()
        assert cand.gptoss_team_fp_probe(0,1,caller_after)==0
        result["caller_controls_before_after"]=[list(caller_before),list(caller_after)]
        assert caller_before[0]&0xffc0 == caller_after[0]&0xffc0
        assert caller_before[1] == caller_after[1]
        # Deliberate corruption must fail strict bits (signed zero preserved).
        negative=b.out.copy(); negative.view(np.uint32)[0]^=1
        try: check_bits(b.out,negative,"negative control")
        except AssertionError: result["negative_control"]=True
        else: raise AssertionError("negative control escaped")
        # Explicit partial-create failure, followed by a successful restart.
        result["failure_injection"]=[]
        for after in (0,1,3):
            assert cand.gptoss_persistent_fail_create_after(after)==0
            tt=(ctypes.c_double*2)(); checksum=ctypes.c_ulonglong()
            returned=cand.gptoss_team_probe_batch(1,8,2,0,tt,ctypes.byref(checksum))
            assert returned==-4 and status(cand)["raw"][1:3]==[0,0]
            result["failure_injection"].append({"after_workers":after,"status":returned})
        assert cand.gptoss_persistent_fail_create_after(-1)==0
        assert cand.gptoss_set_persistent(1)==0
        # Two callers own their output/scratch; library serializes shared pool.
        assert ref.gptoss_set_scale_layout(1)==cand.gptoss_set_scale_layout(1)==0
        weights=np.full(4,.25,np.float32)
        expected=[]
        for x in (inputs["positive"],inputs["negative"]):
            run(ref,packed,bgu,bdn,x,weights,4,8,0,a); expected.append(a.out.copy())
        def concurrent_case(index):
            local=Buffers()
            for _ in range(8):
                run(cand,packed,bgu,bdn,(inputs["positive"],inputs["negative"])[index],weights,4,8,0,local)
                check_bits(local.out,expected[index],"concurrent job")
            return True
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            assert all(executor.map(concurrent_case,(0,1)))
        result["concurrent_serialization_pass"]=True
        if args.time:
            for layout,pool in ((0,raw),(1,packed)):
                ref.gptoss_set_scale_layout(layout); cand.gptoss_set_scale_layout(layout)
                for e in (1,2,3,4):
                    weights=np.full(e,1/e,np.float32)
                    for capture in (False,True):
                        mask=(1<<e)-1 if capture else 0
                        samples={"reference":[],"persistent":[]}; legs=[]
                        for round_index in range(args.repeats):
                            for label,lib,buf in (("reference",ref,a),("persistent",cand,b)) if round_index%2==0 else (("persistent",cand,b),("reference",ref,a)):
                                for i in range(6): run(lib,pool,bgu,bdn,inputs["mixed"],weights,e,8,mask,buf,[(i+j)%4 for j in range(e)])
                                times=[]
                                for i in range(args.iters):
                                    begin=time.perf_counter_ns()
                                    run(lib,pool,bgu,bdn,inputs["mixed"],weights,e,8,mask,buf,[(i+j)%4 for j in range(e)])
                                    times.append((time.perf_counter_ns()-begin)/1e6)
                                samples[label].extend(times); legs.append({"round":round_index,"kind":label,"median_ms":statistics.median(times)})
                        med={label:statistics.median(values) for label,values in samples.items()}
                        timing={"layout":layout,"E":e,"threads":8,"capture":capture,"medians_ms":med,"legs":legs}
                        result["timings"].append(timing)
                        print(f"TIME layout{layout} E{e} cap{capture}: ref{med['reference']:.4f}ms persistent{med['persistent']:.4f}ms",flush=True)
        assert cand.gptoss_persistent_shutdown()==0
        result["shutdown_status"]=status(cand)
        assert result["shutdown_status"]["raw"][:3]==[0,0,0]
        assert fingerprint(refpath)==original_hash
        result["reference_unchanged"]=True; result["all_equal"]=True
        print(f"PASS {result['checks']} cases, negative control, failure injection, concurrency, FP controls, lifecycle",flush=True)
    except Exception as error:
        result["errors"].append(repr(error)); raise
    finally:
        cand.gptoss_persistent_shutdown()
        result["completed_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        dll_dir.close()

if __name__=="__main__": main()
