"""Protocol-matched from-scratch QMIX runner for coalition experiments."""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--env-root',type=Path,required=True); p.add_argument('--stable-root',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--device',default='cuda:0')
p.add_argument('--seed',type=int,default=0); p.add_argument('--total-steps',type=int,default=1536000)
p.add_argument('--critic-warmup-steps',type=int,default=0)  # accepted for common launcher; ignored by QMIX
p.add_argument('--direct-easy',action='store_true'); p.add_argument('--direct-medium',action='store_true'); p.add_argument('--direct-hard',action='store_true')
p.add_argument('--family',choices=('coalition','flow','concurrency'),default='coalition')
p.add_argument('--batch-size',type=int,default=64); p.add_argument('--buffer-capacity',type=int,default=50000)
p.add_argument('--replay-warmup',type=int,default=3200); p.add_argument('--train-every',type=int,default=4)
p.add_argument('--target-sync-every',type=int,default=1600); p.add_argument('--lr',type=float,default=5e-4)
p.add_argument('--gamma',type=float,default=.99); p.add_argument('--epsilon-end',type=float,default=.05)
args=p.parse_args()
for path in (args.env_root/'env',args.env_root/'assgrdims3a',args.stable_root,HERE):sys.path.insert(0,str(path))

from encoding import global_state_dim, global_state_to_array
from marl.common import make_env
from marl.vdn_qmix import QNetwork,QMixMixer,VDNMixer,JointReplayBuffer,epsilon_greedy_actions,_train_step
from marl_reference import _eval_local_model
from standards import official_instance_suite
from training_reward import sparse_delivery_reward

METHOD=os.environ.get('ASSEMBLYGRID_VALUE_DECOMP','qmix').lower()
if METHOD not in ('qmix','vdn'):raise ValueError(f'unsupported value decomposition method: {METHOD}')
FAMILY_SEED_OFFSET={'coalition':0,'flow':300_000,'concurrency':600_000}
PROTOCOL_ID=f'assemblygrid-{args.family}-{METHOD}-new-reward-scratch-v3'
def cell(index,difficulty):return next(x for x in official_instance_suite(index) if x['family']==args.family and x['difficulty']==difficulty)
def completed_operations(core):return sum(len(p.completed_ops) for p in core.products.values())
def means(rows):
    keys=('delivered','completion_rate','throughput','coalition_success_rate','team_formation_failures','team_proposal_churn','deadlock_ticks','robot_occupancy')
    return {k:float(np.mean([float(x.get(k,0) or 0) for x in rows])) for k in keys}
def evaluate(model,difficulty,role,indices):
    # VDN and QMIX have the same decentralized greedy Q-network interface at evaluation;
    # the mixer is training-only, so the benchmark's established QMIX evaluator is shared.
    rows=[_eval_local_model(model,cell(i,difficulty)['config'],'qmix',max_steps=cell(i,difficulty)['config'].horizon_T,device=args.device) for i in indices]
    return {'phase':role,'family':args.family,'difficulty':difficulty,'evaluation_role':'final_test' if role=='final_test' else 'validation','instance_indices':list(indices),'mean':means(rows),'episodes':rows}

if sum((args.direct_easy,args.direct_medium,args.direct_hard))!=1:raise ValueError('choose exactly one difficulty')
if args.total_steps<=0 or args.total_steps%5:raise ValueError('total steps must be positive and divisible by five matched realizations')
difficulty='hard' if args.direct_hard else ('medium' if args.direct_medium else 'easy')
torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed);dev=torch.device(args.device)
def training_cfg(realization):
    _o=FAMILY_SEED_OFFSET[args.family]
    return replace(cell(realization,difficulty)['config'],generation_seed=80_000+_o+args.seed*100+realization,execution_seed=90_000+_o+args.seed*100+realization,official_result=False)
cfg=training_cfg(0); per_instance=args.total_steps//5; realization=0
env=make_env(cfg,training_reward=sparse_delivery_reward()); agents=list(env.agents); n_agents=len(agents)
qnet=QNetwork(hidden=256).to(dev); target_qnet=QNetwork(hidden=256).to(dev);target_qnet.load_state_dict(qnet.state_dict())
mixer=(QMixMixer(n_agents,global_state_dim(env.core),mix_hidden=64,hyper_hidden=128) if METHOD=='qmix' else VDNMixer()).to(dev)
target_mixer=(QMixMixer(n_agents,global_state_dim(env.core),mix_hidden=64,hyper_hidden=128) if METHOD=='qmix' else VDNMixer()).to(dev);target_mixer.load_state_dict(mixer.state_dict())
opt=torch.optim.AdamW(list(qnet.parameters())+list(mixer.parameters()),lr=args.lr,eps=1e-5)
replay=JointReplayBuffer(args.buffer_capacity,n_agents);obs=env._last_obs;masks=dict(env._last_mask)
validation=range(200,205); final_indices=range(1000,1050); phases=[evaluate(qnet,difficulty,'random_initialization',validation)]
history=[];episodes=[];ep={'steps':0,'canonical':0.,'milestone':0.,'teams':0,'operations':0,'deliveries':0};started=time.perf_counter()
epsilon_decay=max(1,int(args.total_steps*.75))
for step in range(1,args.total_steps+1):
    if step>1 and (step-1)%per_instance==0:
        realization+=1;env=make_env(training_cfg(realization),training_reward=sparse_delivery_reward());obs=env._last_obs;masks=dict(env._last_mask)
        ep={'steps':0,'canonical':0.,'milestone':0.,'teams':0,'operations':0,'deliveries':0}
        if len(env.agents)!=n_agents:raise RuntimeError('matched realization changed the agent count')
        if METHOD=='qmix' and global_state_dim(env.core)!=mixer.hyper_b1.in_features:raise RuntimeError('matched realization changed QMIX tensor dimensions')
    epsilon=max(args.epsilon_end,1.-(1.-args.epsilon_end)*(step-1)/epsilon_decay)
    actions=epsilon_greedy_actions(qnet,obs,masks,epsilon,dev,agents);gstate=global_state_to_array(env.core)
    before_proposal={a:env.core._robot(env._agent_to_coord[a]).proposal_id for a in agents}
    bp=env.core.progress_events;bt=env.core.team_formation_count;bo=completed_operations(env.core);bd=env.core.delivered_count;bf=env.core.team_formation_failures;bc=env.core.team_proposal_churn
    next_obs,rewards,terms,truncs,infos=env.step(actions)
    dt=env.core.team_formation_count-bt;do=completed_operations(env.core)-bo;dd=env.core.delivered_count-bd;dp=env.core.progress_events-bp;df=env.core.team_formation_failures-bf;dc=env.core.team_proposal_churn-bc
    local_credit=float(np.mean([.05 if before_proposal[a] is None and env.core._robot(env._agent_to_coord[a]).proposal_id is not None else 0. for a in agents]))
    team_milestone=.10*dp+2.*dt+3.*do+10.*dd-.25*df-.05*dc-.001+local_credit
    reward=float(next(iter(rewards.values()),0.) if rewards else 0.)+team_milestone
    done=bool(next(iter(terms.values()),False) or next(iter(truncs.values()),False)) if terms or truncs else not env.agents
    obs_arr=np.stack([obs[a] for a in agents]);mask_arr=np.stack([masks[a] for a in agents]);act_arr=np.asarray([actions[a] for a in agents])
    ep['steps']+=1;ep['canonical']+=float(next(iter(rewards.values()),0.) if rewards else 0.);ep['milestone']+=team_milestone;ep['teams']+=dt;ep['operations']+=do;ep['deliveries']+=dd
    if done or not env.agents:
        reset_obs,reset_infos=env.reset();reset_masks={a:reset_infos[a]['action_mask'] for a in env.agents}
        next_obs_arr=np.stack([reset_obs[a] for a in agents]);next_mask_arr=np.stack([reset_masks[a] for a in agents]);next_gstate=gstate
    else:
        next_masks={a:infos[a]['action_mask'] for a in env.agents};next_obs_arr=np.stack([next_obs[a] for a in agents]);next_mask_arr=np.stack([next_masks[a] for a in agents]);next_gstate=global_state_to_array(env.core)
    replay.add(obs_arr,mask_arr,act_arr,reward,next_obs_arr,next_mask_arr,float(done),gstate,next_gstate)
    if done or not env.agents:
        episodes.append(dict(ep));ep={'steps':0,'canonical':0.,'milestone':0.,'teams':0,'operations':0,'deliveries':0};obs,masks=reset_obs,reset_masks
    else:obs,masks=next_obs,next_masks
    stats={}
    if len(replay)>=args.replay_warmup and step%args.train_every==0:stats=_train_step(qnet,target_qnet,mixer,target_mixer,opt,replay,args.batch_size,args.gamma,dev)
    if step%args.target_sync_every==0:target_qnet.load_state_dict(qnet.state_dict());target_mixer.load_state_dict(mixer.state_dict())
    if step%16000==0 or step==args.total_steps:
        recent=episodes[-40:]; row={'steps':step,'total_steps':args.total_steps,'stage':difficulty,'realization':realization,'epsilon':epsilon,'td_loss':stats.get('td_loss'),'mean_q_tot':stats.get('mean_q_tot'),'mean_teams':float(np.mean([x['teams'] for x in recent])) if recent else 0.,'mean_operations':float(np.mean([x['operations'] for x in recent])) if recent else 0.,'mean_deliveries':float(np.mean([x['deliveries'] for x in recent])) if recent else 0.,'canonical_reward':float(np.mean([x['canonical'] for x in recent])) if recent else 0.,'formation_reward':float(np.mean([x['milestone'] for x in recent])) if recent else 0.,'training_reward':float(np.mean([x['canonical']+x['milestone'] for x in recent])) if recent else 0.,'mean_episode_length':float(np.mean([x['steps'] for x in recent])) if recent else 0.,'elapsed_seconds':time.perf_counter()-started}
        history.append(row);print(json.dumps(row),flush=True)

phases.append(evaluate(qnet,difficulty,f'{difficulty}_complete',validation));final=evaluate(qnet,difficulty,'final_test',final_indices);phases.append(final)
args.output_dir.mkdir(parents=True,exist_ok=True);ckpt=args.output_dir/f'{METHOD}_final.pt';tmp=ckpt.with_suffix('.pt.tmp')
torch.save({'protocol_id':PROTOCOL_ID,'qnet':qnet.state_dict(),'mixer':mixer.state_dict(),'optimizer':opt.state_dict(),'steps':args.total_steps,'behavior_cloning_steps':0},tmp);tmp.replace(ckpt)
payload={'protocol_id':PROTOCOL_ID,'terminal_status':'complete','method':METHOD,'family':args.family,'target_difficulty':difficulty,'trained_from_scratch':True,'behavior_cloning_steps':0,'expert_trajectories':0,'critic_warmup_steps':None,'replay_warmup_steps':args.replay_warmup,'training_steps':args.total_steps,'phase_history':phases,'training_history':history,'training_seconds':time.perf_counter()-started,'checkpoint':{'path':str(ckpt),'sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()}}
out=args.output_dir/'aag042_result.json';tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(payload,indent=2),encoding='utf-8');tmp.replace(out);print(json.dumps({'result':str(out),'final_test':final['mean']}),flush=True)
