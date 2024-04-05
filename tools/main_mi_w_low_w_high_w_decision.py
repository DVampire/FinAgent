import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["MKL_DEBUG_CPU_TYPE"] = '5'
import warnings
warnings.filterwarnings("ignore")
import os
import sys
from pathlib import Path
import argparse
from mmengine.config import Config, DictAction

from dotenv import load_dotenv
load_dotenv(verbose=True)

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from finagent.registry import DATASET, ENVIRONMENT, PROVIDER, PROMPT, MEMORY, PLOTS
from finagent.asset import ASSET
from finagent.utils.misc import update_data_root
from finagent.utils import read_resource_file, save_json, load_json
from finagent.query import DiverseQuery
from finagent.prompt import (prepare_latest_market_intelligence_params,
                             prepare_low_level_reflection_params,
                             prepare_high_level_reflection_params,
                             prepared_tools_params)
from finagent.tools import StrategyAgents

def parse_args():
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "exp", "trading_mi_w_low_w_high_w_decision", "ETHUSD.py"), help="config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--if_remove", action="store_true", default=False)

    # train
    parser.add_argument("--checkpoint_start_date", type=str, default=None)
    parser.add_argument("--if_load_memory", action="store_true", default=True)
    parser.add_argument("--memory_path", type=str, default=None)
    parser.add_argument("--if_load_trading_record", action="store_true", default=True)
    parser.add_argument("--trading_record_path", type=str, default=None)
    parser.add_argument("--if_train", action="store_true", default=False)
    parser.add_argument("--if_valid", action="store_true", default=True)

    # valid
    # parser.add_argument("--if_load_memory", action="store_true", default=True)
    # parser.add_argument("--if_train", action="store_true", default=False)
    # parser.add_argument("--if_valid", action="store_true", default=True)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is None:
        args.cfg_options = dict()
    if args.root is not None:
        args.cfg_options["root"] = args.root

    args.cfg_options["checkpoint_start_date"] = args.checkpoint_start_date
    args.cfg_options["if_load_memory"] = args.if_load_memory
    args.cfg_options["memory_path"] = args.memory_path
    args.cfg_options["if_load_trading_record"] = args.if_load_trading_record
    args.cfg_options["trading_record_path"] = args.trading_record_path

    if args.if_train is not None:
        args.cfg_options["if_train"] = args.if_train
    if args.if_valid is not None:
        args.cfg_options["if_valid"] = args.if_valid
    cfg.merge_from_dict(args.cfg_options)

    update_data_root(cfg, root=args.root)

    exp_path = os.path.join(cfg.root, cfg.workdir, cfg.tag)
    if args.if_remove is None:
        args.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {exp_path}? ") == 'y')
    if args.if_remove:
        import shutil
        shutil.rmtree(exp_path, ignore_errors=True)
        print(f"| Arguments Remove work_dir: {exp_path}")
    else:
        print(f"| Arguments Keep work_dir: {exp_path}")
    os.makedirs(exp_path, exist_ok=True)

    cfg.dump(os.path.join(exp_path, 'config.py'))

    provider = PROVIDER.build(cfg.provider)

    dataset = DATASET.build(cfg.dataset)
    cfg.train_environment["dataset"] = dataset
    train_env = ENVIRONMENT.build(cfg.train_environment)
    cfg.valid_environment["dataset"] = dataset
    valid_env = ENVIRONMENT.build(cfg.valid_environment)

    plots = PLOTS.build(cfg.plots)

    cfg.memory["symbols"] = dataset.assets
    cfg.memory["embedding_dim"] = provider.get_embedding_dim()
    memory = MEMORY.build(cfg.memory)

    diverse_query = DiverseQuery(memory, provider, top_k=cfg.top_k)
    strategy_agents = StrategyAgents()

    if cfg.if_load_memory and cfg.memory_path is not None:
        print("load local memory...")
        memory_path = os.path.join(cfg.root, cfg.memory_path)
        memory.load_local(memory_path=memory_path)

    if cfg.if_train:
        train_records = run(cfg,
                            train_env,
                            plots,
                            memory,
                            provider,
                            diverse_query,
                            strategy_agents,
                            exp_path,
                            mode = "train")
        train_save_path = os.path.join(exp_path, "train_records.json")

        memory.save_local(memory_path=cfg.memory_path)
        save_json(train_records, train_save_path)

    if cfg.if_valid:
        valid_records = run(cfg,
                            valid_env,
                            plots,
                            memory,
                            provider,
                            diverse_query,
                            strategy_agents,
                            exp_path,
                            mode = "valid")
        valid_save_path = os.path.join(exp_path, "valid_records.json")
        save_json(valid_records, valid_save_path)

def run(cfg, env, plots, memory, provider, diverse_query, strategy_agents,  exp_path, mode = "train"):

    trading_records_path = os.path.join(exp_path, "trading_records")
    os.makedirs(trading_records_path, exist_ok=True)
    memory_path = os.path.join(exp_path, "memory_records")
    os.makedirs(memory_path, exist_ok=True)

    if cfg.if_load_trading_record and cfg.trading_record_path is not None:
        print("load trading records...")
        record_path = os.path.join(cfg.root, cfg.trading_record_path)
        trading_records = load_json(record_path)
    else:
        trading_records = {
            "symbol": [],
            "day": [],
            "value": [],
            "cash": [],
            "position": [],
            "ret": [],
            "date": [],
            "price": [],
            "discount": [],
            "kline_path": [],
            "trading_path": [],
            "total_profit": [],
            "total_return": [],
            "action": [],
            "reasoning": [],
        }

    state, info = env.reset()

    if cfg.checkpoint_start_date is not None:
        for action, date in zip(trading_records["action"], trading_records["date"]):
            if date <= cfg.checkpoint_start_date:
                action = env.action_map[action]
                state, reward, done, truncated, info = env.step(action)
            else:
                break

    while True:

        action = run_step(cfg,
                          state,
                          info,
                          plots,
                          memory,
                          provider,
                          diverse_query,
                          strategy_agents,
                          exp_path,
                          trading_records,
                          mode)

        assert action in env.action_map.keys(), f"Action {action} is not in the action map {env.action_map.keys()}"

        action = env.action_map[action]
        state, reward, done, truncated, info = env.step(action)

        if trading_records["action"][-1] != info["action"]:
            trading_records["action"][-1] = info["action"]

        if done:
            trading_records["total_profit"].append(info["total_profit"])
            trading_records["total_return"].append(info["total_return"])
            trading_records["date"].append(info["date"])
            trading_records["price"].append(info["price"])
            break

        memory_save_path = os.path.join(memory_path, f"memory_{str(info['date'])}")
        os.makedirs(memory_save_path, exist_ok=True)
        memory.save_local(memory_path=memory_save_path)

        save_json(trading_records, os.path.join(trading_records_path, f"trading_records_{str(info['date'])}.json"))

    return trading_records

def run_step(cfg, state, info, plots, memory, provider, diverse_query, strategy_agents, exp_path, trading_records, mode):

    params = dict()
    save_dir = "train" if mode == "train" else "valid"

    # plot kline
    kline_path = plots.plot_kline(state=state, info=info, save_dir=save_dir, mode=mode)
    params.update({
        "kline_path": kline_path
    })

    # tools
    tools_params = prepared_tools_params(state=state,
                                         info=info,
                                         params=params,
                                         memory=memory,
                                         provider=provider,
                                         diverse_query=diverse_query,
                                         strategy_agents=strategy_agents,
                                         cfg=cfg,
                                         mode=mode)
    params.update(tools_params)

    # latest market intelligence
    latest_market_intelligence_summary_template = read_resource_file(cfg.train_latest_market_intelligence_summary_template_path
                                                   if mode == "train" else cfg.valid_latest_market_intelligence_summary_template_path)
    latest_market_intelligence_summary = PROMPT.build(cfg.latest_market_intelligence_summary)
    latest_market_intelligence_summary_res = latest_market_intelligence_summary.run(state = state,
                                                                      info = info,
                                                                      params = params,
                                                                      template = latest_market_intelligence_summary_template,
                                                                      memory = memory,
                                                                      provider = provider,
                                                                      diverse_query = diverse_query,
                                                                      exp_path = exp_path,
                                                                      save_dir = save_dir,
                                                                      call_provider = False)

    # query past market intelligence
    prepared_latest_market_intelligence_params = prepare_latest_market_intelligence_params(state=state,
                                                                                        info=info,
                                                                                        params=params,
                                                                                        memory=memory,
                                                                                        provider=provider,
                                                                                        diverse_query=diverse_query)
    params.update(prepared_latest_market_intelligence_params)

    # add latest market intelligence to memory
    latest_market_intelligence_summary.add_to_memory(state=state,
                                                     info=info,
                                                     res=latest_market_intelligence_summary_res,
                                                     memory=memory,
                                                     provider=provider)

    # past market intelligence
    past_market_intelligence_summary_template = read_resource_file(cfg.train_past_market_intelligence_summary_template_path
                                                   if mode == "train" else cfg.valid_past_market_intelligence_summary_template_path)
    past_market_intelligence_summary = PROMPT.build(cfg.past_market_intelligence_summary)
    past_market_intelligence_summary_res = past_market_intelligence_summary.run(state=state,
                                                                                info=info,
                                                                                template=past_market_intelligence_summary_template,
                                                                                params=params,
                                                                                memory=memory,
                                                                                provider=provider,
                                                                                diverse_query=diverse_query,
                                                                                exp_path=exp_path,
                                                                                save_dir=save_dir,
                                                                                call_provider=False)

    # low level reflection
    low_level_reflection_template = read_resource_file(cfg.train_low_level_reflection_template_path
                                                       if mode == "train" else cfg.valid_low_level_reflection_template_path)
    low_level_reflection = PROMPT.build(cfg.low_level_reflection)
    low_level_reflection_res = low_level_reflection.run(state=state,
                                                        info=info,
                                                        template=low_level_reflection_template,
                                                        params=params,
                                                        memory=memory,
                                                        provider=provider,
                                                        diverse_query=diverse_query,
                                                        exp_path=exp_path,
                                                        save_dir=save_dir,
                                                        call_provider=False)

    # query past low level reflection
    prepared_low_level_reflection_params = prepare_low_level_reflection_params(state=state,
                                                                               info=info,
                                                                               params=params,
                                                                               memory=memory,
                                                                               provider=provider,
                                                                               diverse_query=diverse_query)
    params.update(prepared_low_level_reflection_params)

    # add low level reflection to memory
    low_level_reflection.add_to_memory(state=state,
                                       info=info,
                                       res=low_level_reflection_res,
                                       memory=memory,
                                       provider=provider)

    # plot trading
    if len(trading_records["date"]) <= 0:
        trading_path = None
    else:
        trading_path = plots.plot_trading(records=trading_records, info=info, save_dir=save_dir)
    params.update({
        "trading_path": trading_path
    })

    # high level reflection
    previous_date = trading_records["date"]
    previous_action = trading_records["action"]
    previous_reasoning = trading_records["reasoning"]
    params.update({
        "previous_date": previous_date,
        "previous_action": previous_action,
        "previous_reasoning": previous_reasoning,
    })
    high_level_reflection_template = read_resource_file(cfg.train_high_level_reflection_template_path
                                                        if mode == "train" else cfg.valid_high_level_reflection_template_path)
    high_level_reflection = PROMPT.build(cfg.high_level_reflection)
    high_level_reflection_res = high_level_reflection.run(state=state,
                                                          info=info,
                                                          template=high_level_reflection_template,
                                                          params=params,
                                                          memory=memory,
                                                          provider=provider,
                                                          diverse_query=diverse_query,
                                                          exp_path=exp_path,
                                                          save_dir=save_dir)

    # query past high level reflection
    prepared_high_level_reflection_params = prepare_high_level_reflection_params(state=state,
                                                                                 info=info,
                                                                                 params=params,
                                                                                 memory=memory,
                                                                                 provider=provider,
                                                                                 diverse_query=diverse_query)
    params.update(prepared_high_level_reflection_params)

    # add high level reflection to memory
    high_level_reflection.add_to_memory(state=state,
                                        info=info,
                                        res=high_level_reflection_res,
                                        memory=memory,
                                        provider=provider)

    # trader preference
    trader_preference = ASSET.get_trader(cfg.trader_preference)
    params.update({
        "trader_preference": trader_preference
    })

    # decision
    decision_template = read_resource_file(cfg.train_decision_template_path
                                           if mode == "train" else cfg.valid_decision_template_path)
    decision = PROMPT.build(cfg.decision)
    decision_res = decision.run(state=state,
                                info=info,
                                template=decision_template,
                                params=params,
                                memory=memory,
                                provider=provider,
                                diverse_query=diverse_query,
                                exp_path=exp_path,
                                save_dir=save_dir)

    # add records
    trading_records["symbol"].append(info["symbol"])
    trading_records["day"].append(info["day"])
    trading_records["value"].append(info["value"])
    trading_records["cash"].append(info["cash"])
    trading_records["position"].append(info["position"])
    trading_records["ret"].append(info["ret"])
    trading_records["date"].append(info["date"])
    trading_records["price"].append(info["price"])
    trading_records["discount"].append(info["discount"])
    trading_records["kline_path"].append(kline_path)
    trading_records["trading_path"].append(trading_path)
    trading_records["total_profit"].append(info["total_profit"])
    trading_records["total_return"].append(info["total_return"])
    trading_records["action"].append(decision_res["response_dict"]["action"])
    trading_records["reasoning"].append(decision_res["response_dict"]["reasoning"])

    action = decision_res["response_dict"]["action"]

    return action

if __name__ == '__main__':
    main()