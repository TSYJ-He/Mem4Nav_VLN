# type: ignore
import argparse
import torch
import os
import pprint 
import json 
import time
import logging 
try:
    from utils.config_parser import load_yaml_config, merge_configs, save_config_to_yaml, get_config_value #type: ignore
    from utils.logging_setup import setup_logger, get_logger
    from utils.general_helpers import set_random_seed, ensure_directory_exists, format_time_duration, count_trainable_parameters
except ImportError as e:
    # Basic logging if setup_logger hasn't run or fails
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Critical Error: Could not import utility modules: {e}. Ensure PYTHONPATH is correctly set and all util files exist.")
    exit(1)
try:
    from data_handling.base_loader import BaseVLNDataset 
    from data_handling.touchdown_loader import TouchdownDatasetLoader
    from data_handling.map2seq_loader import Map2SeqDatasetLoader
    from data_handling.graph_loader import VLNGraphLoader 
    # from data_handling.panorama_dataset import PanoramaDatasetForMAE 
    # from data_handling.synthetic_trajectory_dataset import SyntheticTrajectoryDatasetForLTM
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError as e:
    get_logger("Mem4NavApp.main").error(f"Critical Error: Could not import data_handling modules or transformers: {e}. Ensure dependencies are installed and PYTHONPATH is correct.")
    exit(1)
try:
    from agents.base_vln_agent import BaseVLNAgent # For type hinting
    from agents.modular_pipeline.agent import ModularAgent
    from agents.velma_integration.agent import VelmaMem4NavAgent
    from agents.flame_integration.agent import FlameMem4NavAgent
except ImportError as e:
    get_logger("Mem4NavApp.main").error(f"Critical Error: Could not import agent modules: {e}. Ensure agent files exist and are correct.")
    exit(1)
try:
    from training_utils.trainer import Trainer
    from evaluation_utils.evaluator import Evaluator, EnvironmentGraph
except ImportError as e:
    get_logger("Mem4NavApp.main").error(f"Critical Error: Could not import training_utils or evaluation_utils modules: {e}.")
    exit(1)
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Mem4Nav: Training and Evaluation Framework")
    parser.add_argument("--config_base", type=str, default="configs/base.yaml", #type: ignore 
                        help="Path to the base YAML configuration file. (Default: configs/base.yaml)") #type: ignore
    parser.add_argument("--config_experiment", type=str, required=True,
                        help="Path to the experiment-specific YAML configuration file (e.g., configs/modular_touchdown.yaml).") #type: ignore
    parser.add_argument("--run_mode", type=str, choices=["train", "evaluate"], required=True,
                        help="Mode to run: 'train' for training, or 'evaluate' for evaluation.")
    parser.add_argument("--agent_type", type=str, choices=["modular", "velma", "flame"], required=True,
                        help="Type of agent backbone to use: 'modular', 'velma', or 'flame'.")
    
    # Evaluation specific overrides
    parser.add_argument("--eval_split", type=str, default=None,
                        help="Dataset split for evaluation (e.g., 'dev', 'test_seen'). Overrides config if set for evaluation mode.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to a model checkpoint. Used for evaluation or resuming training if 'training.resume_from_checkpoint' is not set in config.")
    
    # General overrides for convenience
    parser.add_argument("--output_dir_override", type=str, default=None,
                        help="Override the root output directory (experiment.output_dir_root in config).")
    parser.add_argument("--device_override", type=str, choices=["cuda", "cpu"], default=None,
                        help="Override device specified in config (e.g., 'cuda', 'cpu').")
    parser.add_argument("--seed_override", type=int, default=None,
                        help="Override random seed specified in config.")
    parser.add_argument("--num_workers_override", type=int, default=None,
                        help="Override num_workers for DataLoader specified in config.")

    args = parser.parse_args()
    return args


def load_and_prepare_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads base and experiment configs, merges them, and applies CLI overrides."""
    try:
        base_cfg = load_yaml_config(args.config_base) #type: ignore
        exp_cfg = load_yaml_config(args.config_experiment) #type: ignore
    except FileNotFoundError as e:
        # Use basic logging if setup_logger hasn't run
        logging.error(f"Configuration file error: {e}")
        exit(1)
    except yaml.YAMLError as e:#type: ignore 
        logging.error(f"YAML parsing error: {e}")
        exit(1)
    
    config = merge_configs(base_cfg, exp_cfg)

    # Apply CLI overrides to the merged config
    if args.output_dir_override:
        config['experiment']['output_dir_root'] = args.output_dir_override
    if args.device_override:
        config['experiment']['device'] = args.device_override
    if args.seed_override is not None:
        config['experiment']['seed'] = args.seed_override
    
    if args.num_workers_override is not None:
        config['data_handling'] = config.get('data_handling', {})
        config['data_handling']['num_workers'] = args.num_workers_override

    if args.run_mode == "evaluate":
        config['evaluation'] = config.get('evaluation', {})
        if args.eval_split:
            config['evaluation']['eval_split_override'] = args.eval_split
        if args.checkpoint_path: # Prioritize CLI checkpoint for eval
            config['experiment']['load_checkpoint_path'] = args.checkpoint_path
    
    if args.run_mode == "train" and args.checkpoint_path:
        # If CLI checkpoint is provided for training, use it for resume if not already in config
        config['training'] = config.get('training', {})
        if not config['training'].get('resume_from_checkpoint'):
            config['training']['resume_from_checkpoint'] = args.checkpoint_path
        config['experiment']['load_checkpoint_path'] = args.checkpoint_path # Also set for initial load
    exp_name = get_config_value(config, "experiment.name", "unnamed_experiment")
    output_root = get_config_value(config, "experiment.output_dir_root", "./outputs_mem4nav")
    config['experiment']['full_output_dir'] = os.path.join(output_root, exp_name, time.strftime("%Y%m%d-%H%M%S"))
    ensure_directory_exists(config['experiment']['full_output_dir'])

    return config


def get_tokenizer(config: Dict[str, Any], agent_type: str) -> Optional[PreTrainedTokenizerBase]:   
    """Initializes and returns the tokenizer based on agent type and config."""
    logger = get_logger("Mem4NavApp.main.tokenizer")
    tokenizer_name_or_path: Optional[str] = None #type: ignore
    
    agent_base_config_path = f"agents.{agent_type}_agent" # e.g., "agents.flame_agent"
    agent_specific_config = get_config_value(config, agent_base_config_path, {})

    if agent_type == "flame":
        tokenizer_name_or_path = agent_specific_config.get('flame_model_path')
    elif agent_type == "velma":
        llm_model_name = agent_specific_config.get('llm_model')
        if llm_model_name and not llm_model_name.startswith("openai/"): # OpenAI API handles tokenization server-side
            tokenizer_name_or_path = llm_model_name
    elif agent_type == "modular":
        tokenizer_name_or_path = agent_specific_config.get('tokenizer_path', 
                                     get_config_value(config, "data_handling.default_tokenizer"))
                                     
    if tokenizer_name_or_path:
        logger.info(f"Attempting to load tokenizer: {tokenizer_name_or_path} for agent {agent_type}")
        try:
            # trust_remote_code might be needed for some newer models/tokenizers
            return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{tokenizer_name_or_path}': {e}")
            return None
    logger.info(f"No specific tokenizer path found for agent '{agent_type}'. Instruction tokenization might rely on dataset loader defaults or agent internals.")
    return None


def get_vln_dataloader(config: Dict[str, Any], dataset_name: str, split: str, tokenizer: Optional[PreTrainedTokenizerBase]) -> DataLoader:   
    """Creates and returns a DataLoader for a specific VLN dataset and split."""
    logger = get_logger("Mem4NavApp.main.dataloader")
    data_handling_config = get_config_value(config, "data_handling", required=True)
    
    is_train_split = 'train' in split.lower()
    batch_size_key = 'batch_size' if is_train_split else 'eval_batch_size'
    batch_size = data_handling_config.get(batch_size_key, 16 if is_train_split else 8)
    num_workers = data_handling_config.get('num_workers', 0)
    max_instr_len = data_handling_config.get('max_instruction_length', 128)

    dataset_instance: Optional[BaseVLNDataset] = None #type: ignore
    loader_config_key_in_dh = f"{dataset_name}_loader" # e.g., "touchdown_loader"
    if dataset_name.lower() == "touchdown":
        dataset_instance = TouchdownDatasetLoader(config, split, tokenizer, max_instr_len, name=loader_config_key_in_dh.capitalize())
    elif dataset_name.lower() == "map2seq":
        dataset_instance = Map2SeqDatasetLoader(config, split, tokenizer, max_instr_len, name=loader_config_key_in_dh.capitalize())
    else:
        logger.error(f"Unsupported dataset_name for VLN: {dataset_name}")
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
    logger.info(f"Created {dataset_name} dataset for split '{split}' with {len(dataset_instance)} instances.")   
    return DataLoader( #type: ignore
        dataset_instance,   
        batch_size=batch_size, 
        shuffle=is_train_split, 
        num_workers=num_workers,
        collate_fn=getattr(dataset_instance, 'collate_fn', None) # Use custom collate if defined by dataset
    )

def get_pretrain_dataloader(config: Dict[str, Any], phase_name: str, dataset_type_config_key: str,  #type: ignore
                            tokenizer: Optional[PreTrainedTokenizerBase]) -> Optional[DataLoader]:   
    logger = get_logger("Mem4NavApp.main.dataloader")
    phase_cfg = get_config_value(config, f"training.phases.{phase_name}", {})
    if not phase_cfg.get("enabled", False):
        logger.info(f"Pre-training phase '{phase_name}' is disabled in config.")
        return None

    dataset_name = phase_cfg.get("dataset_name_override", get_config_value(config, "data_handling.dataset_name"))
    split = phase_cfg.get("split", "train") # e.g., use VLN train set panoramas
    # dataset_type_config_key e.g., "panorama_dataset_for_mae" or "synthetic_trajectory_dataset_for_ltm"
    # This part needs actual Dataset classes like PanoramaDatasetForMAE or SyntheticTrajectoryDatasetForLTM
    
    logger.warning(f"Placeholder for pre-training dataloader '{dataset_type_config_key}' for phase '{phase_name}'. "
                   f"Actual implementation of specific Dataset class and loading logic needed.")

    mock_bs = get_config_value(config, "data_handling.batch_size", 16)
    return DataLoader(BaseVLNDataset(num_samples=100), batch_size=mock_bs)   


def initialize_agent(agent_type_str: str, config: Dict[str, Any], device: torch.device) -> BaseVLNAgent:
    logger = get_logger("Mem4NavApp.main.agent")
    agent: Optional[BaseVLNAgent] = None #type: ignore
    if agent_type_str == "modular":
        agent = ModularAgent(config, device)
    elif agent_type_str == "velma":
        agent = VelmaMem4NavAgent(config, device)
    elif agent_type_str == "flame":
        agent = FlameMem4NavAgent(config, device)
    else:
        logger.error(f"Unsupported agent_type: {agent_type_str}")
        raise ValueError(f"Unsupported agent_type: {agent_type_str}")
    
    agent.to(device)
    logger.info(f"Initialized agent: {agent_type_str} on device: {device}")
    logger.info(f"Agent trainable parameters: {count_trainable_parameters(agent):,}")
    return agent


def run_training(args: argparse.Namespace, config: Dict[str, Any], device: torch.device):#type: ignore
    """Initializes and runs the training process."""
    logger = get_logger("Mem4NavApp.train")
    logger.info("=== Starting Training Run ===")
    
    tokenizer = get_tokenizer(config, args.agent_type)
    agent = initialize_agent(args.agent_type, config, device)

    train_dataloaders: Dict[str, DataLoader] = {} #type: ignore
    val_dataloader_vln: Optional[DataLoader] = None #type: ignore
    
    training_config = get_config_value(config, "training", required=True)
    phases_config = get_config_value(training_config, "phases", required=True)

    if get_config_value(phases_config, "phase1_visual.enabled", False):
        dl = get_pretrain_dataloader(config, "phase1_visual", "panorama_dataset_for_mae", tokenizer)
        if dl: train_dataloaders["phase1_visual"] = dl

    if get_config_value(phases_config, "phase2_ltm.enabled", False):
        dl = get_pretrain_dataloader(config, "phase2_ltm", "synthetic_trajectory_dataset_for_ltm", tokenizer)
        if dl: train_dataloaders["phase2_ltm_synthetic"] = dl

    if get_config_value(phases_config, "phase3_e2e_nav.enabled", True):
        dataset_name = get_config_value(config, "data_handling.dataset_name", required=True)
        p3_train_split = get_config_value(phases_config, "phase3_e2e_nav.train_split", "train")
        p3_val_split = get_config_value(phases_config, "phase3_e2e_nav.val_split", "dev")
        
        try:
            train_dataloaders["phase3_vln"] = get_vln_dataloader(config, dataset_name, p3_train_split, tokenizer)
            val_dataloader_vln = get_vln_dataloader(config, dataset_name, p3_val_split, tokenizer)
        except Exception as e:
            logger.error(f"Failed to load dataloaders for Phase 3: {e}. Phase 3 might be skipped or fail.")
            if "phase3_vln" in train_dataloaders: del train_dataloaders["phase3_vln"]
            val_dataloader_vln = None
    trainer = Trainer(
        agent=agent,
        train_dataloaders=train_dataloaders,
        val_dataloader_vln=val_dataloader_vln,
        config=config,
        device=device
    )

    resume_checkpoint_path = get_config_value(config, "training.resume_from_checkpoint", None)
    if resume_checkpoint_path:
        trainer.load_checkpoint(resume_checkpoint_path)

    trainer.train()

    final_model_path = os.path.join(get_config_value(config, "experiment.full_output_dir"), "final_agent_model.pth")
    torch.save(agent.state_dict(), final_model_path)
    logger.info(f"Final trained agent saved to: {final_model_path}")
    logger.info("=== Training Run Finished ===")


def run_evaluation(args: argparse.Namespace, config: Dict[str, Any], device: torch.device):
    """Initializes and runs the evaluation process."""
    logger = get_logger("Mem4NavApp.eval")
    logger.info("=== Starting Evaluation Run ===")

    tokenizer = get_tokenizer(config, args.agent_type)
    agent = initialize_agent(args.agent_type, config, device)

    checkpoint_path = get_config_value(config, "experiment.load_checkpoint_path", None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading agent checkpoint for evaluation from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        agent_state_dict = state.get('agent_state_dict', state) # Handle both trainer and raw state_dicts
        agent.load_state_dict(agent_state_dict)
    elif checkpoint_path:
        logger.error(f"Specified checkpoint for evaluation not found: {checkpoint_path}. Evaluating with initial weights.")
    else:
        logger.warning("No checkpoint path for evaluation. Evaluating with initial/default agent weights.")
    agent.eval()

    dataset_name = get_config_value(config, "data_handling.dataset_name", required=True)
    eval_split = get_config_value(config, "evaluation.eval_split_override", 
                                  get_config_value(config, "evaluation.eval_split", "dev")) # Default eval split

    eval_dataloader = get_vln_dataloader(config, dataset_name, eval_split, tokenizer)
    
    # --- Load Environment Graph ---
    # This is critical for Evaluator and dataset-specific
    env_graph: Optional[EnvironmentGraph] = None  #type: ignore
    graph_data_dir_template = get_config_value(config, "data_handling.graph_dir_template", 
                                               os.path.join("{dataset_root}", "{dataset_specific_path_from_loader}", "graph"))
    
    loader_config_key = f"{dataset_name.lower()}_loader"
    dataset_specific_path_segment = get_config_value(config, f"data_handling.{loader_config_key}.specific_path", dataset_name)

    graph_data_dir = graph_data_dir_template.format(
        dataset_root=get_config_value(config, "data_handling.dataset_root_path"),
        dataset_specific_path_from_loader=dataset_specific_path_segment
    )
    
    logger.info(f"Attempting to load environment graph data from: {graph_data_dir}")
    if os.path.isdir(graph_data_dir):
        try:
            vln_graph_loader = VLNGraphLoader(graph_data_dir)
            nx_graph = vln_graph_loader.get_graph()
            node_positions = vln_graph_loader.get_node_positions()
            if nx_graph and nx_graph.number_of_nodes() > 0:
                env_graph = EnvironmentGraph(nx_graph=nx_graph)
                env_graph.node_positions = node_positions # Populate positions in our EnvironmentGraph wrapper
                logger.info(f"Successfully loaded environment graph with {nx_graph.number_of_nodes()} nodes.")
            else:
                logger.warning(f"Loaded graph from {graph_data_dir} is empty or invalid. Using mock graph in Evaluator.")
                env_graph = EnvironmentGraph() # Fallback to mock
                env_graph.add_mock_nodes_and_edges()
        except Exception as e:
            logger.error(f"Failed to load or process environment graph from {graph_data_dir}: {e}. Using mock graph.", exc_info=True)
            env_graph = EnvironmentGraph()
            env_graph.add_mock_nodes_and_edges()
    else:
        logger.warning(f"Environment graph directory not found: {graph_data_dir}. Using mock graph in Evaluator.")
        env_graph = EnvironmentGraph()
        env_graph.add_mock_nodes_and_edges()
    evaluator = Evaluator(
        agent=agent,
        dataloader=eval_dataloader,
        env_graph=env_graph,
        eval_config=get_config_value(config, "evaluation", {}),
        device=device
    )
    metrics = evaluator.evaluate()

    output_dir = get_config_value(config, "experiment.full_output_dir")
    metrics_file_name = f"metrics_{args.agent_type}_{dataset_name}_{eval_split}"
    if checkpoint_path:
        metrics_file_name += f"_ckpt_{os.path.splitext(os.path.basename(checkpoint_path))[0]}"
    metrics_file_name += ".json"
    
    metrics_file_path = os.path.join(output_dir, "results", metrics_file_name)
    ensure_directory_exists(os.path.dirname(metrics_file_path))
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation metrics saved to: {metrics_file_path}")
    logger.info(f"Final evaluation metrics: {pprint.pformat(metrics)}")
    logger.info("=== Evaluation Run Finished ===")


def main():
    """Main entry point for the Mem4Nav project."""
    args = parse_arguments()
    config = load_and_prepare_config(args)

    log_dir = os.path.join(config['experiment']['full_output_dir'], "logs")
    ensure_directory_exists(log_dir)
    log_file_name = get_config_value(config, "logging.log_file_name", f"{args.agent_type}_{args.run_mode}_{time.strftime('%m%d_%H%M')}.log")
    log_file_full_path = os.path.join(log_dir, log_file_name)
    
    logger_config_dict = get_config_value(config, "logging", {})
    main_logger = setup_logger(
        name="Mem4NavApp", 
        config={'logging': logger_config_dict}, 
        log_file=log_file_full_path,
        force_reconfigure=True # Ensure it uses the latest path
    )
    
    main_logger.info(f"Running with arguments: {pprint.pformat(vars(args))}")
    main_logger.info(f"Effective configuration saved to: {os.path.join(config['experiment']['full_output_dir'], 'effective_config.yaml')}") #type: ignore
    save_config_to_yaml(config, os.path.join(config['experiment']['full_output_dir'], "effective_config.yaml"))#type: ignore

    set_random_seed(get_config_value(config, "experiment.seed", 42))

    device_str = get_config_value(config, "experiment.device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        main_logger.warning("CUDA specified but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    main_logger.info(f"Using device: {device}")

    start_time = time.time()
    if args.run_mode == "train":
        run_training(args, config, device)
    elif args.run_mode == "evaluate":
        run_evaluation(args, config, device)
    else:
        main_logger.error(f"Invalid run_mode specified: {args.run_mode}")
        
    total_time_seconds = time.time() - start_time
    main_logger.info(f"Total execution time: {format_time_duration(total_time_seconds)}")
    main_logger.info(f"Outputs and logs saved in: {config['experiment']['full_output_dir']}")

if __name__ == "__main__":
    main()
