import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch

from rl.models import TransformerPolicy
from rl.envs import build_environment
from rl.algo import train_policy, evaluate_episode


DATA_DIR = "rl-project/data"  # Path to data directory containing train/test/val splits
START_DATE = None  # Start date for data filtering (YYYY-MM-DD) or None
END_DATE = None    # End date for data filtering (YYYY-MM-DD) or None

INITIAL_TRAIN_YEARS = 4  # Initial training period in years
VALIDATION_YEARS = 1     # Validation period in years
STEP_YEARS = 2         # Years to add to training in each step

# Grid search parameters - comprehensive search space
D_MODEL_OPTIONS = [32, 64, 128]  # Model dimension
NHEAD_OPTIONS = [2, 4, 8]        # Number of attention heads
NUM_LAYERS_OPTIONS = [1, 2, 3]   # Number of transformer layers
HIDDEN_OPTIONS = [64, 128, 256]  # Hidden layer size
DROPOUT_OPTIONS = [0.05, 0.1, 0.2]  # Dropout rates
ACTIVATION_OPTIONS = ["gelu", "relu"]  # Activation functions
NORM_FIRST_OPTIONS = [True, False]     # Pre/post normalization
FF_HIDDEN_MULT_OPTIONS = [2, 4]       # Feed-forward hidden multiplier
MAX_LEN = 5000

EPOCHS_PER_STEP_OPTIONS = [30, 50, 100]  # Epochs to train in each walk-forward step
TRAIN_LENGTH_OPTIONS = [8, 12, 16]    # Training window length options
LEARNING_RATE_OPTIONS = [0.000001, 0.00001, 0.0001]  # Learning rate options
RISK_FREE_RATE = 0.0    # Risk-free rate for Sharpe calculation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

RET_COL = "ret_next"
ID_COL = "ticker"
DATE_COL = "date"
WINDOW = 48
LENGTH = 24
MIN_OBS_IN_WINDOW = 1
STOCHASTIC = True
REBALANCE_PROB = 0.3
MAX_CHANGE = 2
MIN_PORTFOLIO_SIZE = 3
MAX_PORTFOLIO_SIZE = 15

# Output configuration
OUTPUT_DIR = "walk_forward_results"  # Output directory for results and checkpoints
SAVE_CHECKPOINTS = True  # Save model checkpoints after each step


def generate_parameter_combinations():
    """Generate all combinations of hyperparameters for grid search."""
    import itertools
    
    param_grid = {
        'd_model': D_MODEL_OPTIONS,
        'nhead': NHEAD_OPTIONS,
        'num_layers': NUM_LAYERS_OPTIONS,
        'hidden': HIDDEN_OPTIONS,
        'dropout': DROPOUT_OPTIONS,
        'activation': ACTIVATION_OPTIONS,
        'norm_first': NORM_FIRST_OPTIONS,
        'ff_hidden_mult': FF_HIDDEN_MULT_OPTIONS,
        'epochs_per_step': EPOCHS_PER_STEP_OPTIONS,
        'train_length': TRAIN_LENGTH_OPTIONS,
        'learning_rate': LEARNING_RATE_OPTIONS
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = []
    
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    print(f"Generated {len(combinations)} parameter combinations for grid search")
    return combinations


def get_date_ranges(data: pd.DataFrame, date_col: str, 
                   initial_train_years: int, validation_years: int, 
                   step_years: int) -> List[Tuple[str, str, str, str]]:
    """
    Generate date ranges for walk-forward validation.
    
    Args:
        data: DataFrame with date column
        date_col: Name of date column
        initial_train_years: Initial training period in years
        validation_years: Validation period in years
        step_years: Years to add to training in each step
        
    Returns:
        List of (train_start, train_end, val_start, val_end) tuples
    """
    data[date_col] = pd.to_datetime(data[date_col])
    min_date = data[date_col].min()
    max_date = data[date_col].max()
    
    ranges = []
    current_train_years = initial_train_years
    
    while True:
        # Calculate training period
        train_start = min_date
        train_end = min_date + pd.DateOffset(years=current_train_years)
        
        # Calculate validation period
        val_start = train_end
        val_end = val_start + pd.DateOffset(years=validation_years)
        
        # Check if we have enough data for validation
        if val_end > max_date:
            break
            
        ranges.append((
            train_start.strftime('%Y-%m-%d'),
            train_end.strftime('%Y-%m-%d'),
            val_start.strftime('%Y-%m-%d'),
            val_end.strftime('%Y-%m-%d')
        ))
        
        # Move to next step
        current_train_years += step_years
    
    return ranges


def load_and_prepare_data(data_dir: str, date_col: str, 
                         train_start: str, train_end: str,
                         val_start: str, val_end: str) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Load and prepare data for training and validation periods."""
    
    print(f"Training data: {train_start} to {train_end}")
    print(f"Validation data: {val_start} to {val_end}")
    
    # Build training environment with date filtering
    X_train, R_train, ids_train, dates_train, M_feat_train = build_environment(
        data_dir=data_dir,
        split="processed_weekly_panel.parquet",
        ret_col=RET_COL,
        id_col=ID_COL,
        date_col=DATE_COL,
        window=WINDOW,
        min_obs_in_window=MIN_OBS_IN_WINDOW,
        start_date=train_start,
        end_date=train_end,
        stochastic=False,
        rebalance_prob=0.3,
        max_change=2,
        min_portfolio_size=3,
        max_portfolio_size=15
    )
    
    # Build validation environment with date filtering
    X_val, R_val, ids_val, dates_val, M_feat_val = build_environment(
        data_dir=data_dir,
        split="processed_weekly_panel.parquet",
        ret_col=RET_COL,
        id_col=ID_COL,
        date_col=DATE_COL,
        window=WINDOW,
        min_obs_in_window=MIN_OBS_IN_WINDOW,
        start_date=val_start,
        end_date=val_end,
        stochastic=False,
        rebalance_prob=0.3,
        max_change=2,
        min_portfolio_size=3,
        max_portfolio_size=15
    )
    
    return (X_train, R_train, M_feat_train, ids_train,
            X_val, R_val, M_feat_val, ids_val, dates_train, dates_val)


def run_walk_forward_validation():
    """Run the complete walk-forward validation process with grid search."""
    
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Load data to get date ranges (use train_val split to determine date range)
    _, _, _, dates_temp, _ = build_environment(
        data_dir=DATA_DIR,
        split="processed_weekly_panel.parquet",
        ret_col=RET_COL,
        id_col=ID_COL,
        date_col=DATE_COL,
        window=WINDOW,
        min_obs_in_window=MIN_OBS_IN_WINDOW,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # Create a simple DataFrame for date range calculation
    if len(dates_temp) > 0:
        df = pd.DataFrame({'date': dates_temp})
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError("No data found in the specified data directory")
    
    # Get walk-forward date ranges
    date_ranges = get_date_ranges(
        df, DATE_COL,
        INITIAL_TRAIN_YEARS, VALIDATION_YEARS, STEP_YEARS
    )
    
    print(f"Found {len(date_ranges)} walk-forward steps")
    for i, (train_start, train_end, val_start, val_end) in enumerate(date_ranges):
        print(f"Step {i+1}: Train {train_start} to {train_end}, Val {val_start} to {val_end}")
    
    # Generate all parameter combinations
    param_combinations = generate_parameter_combinations()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize results storage
    all_grid_results = []
    
    # Run grid search
    for param_idx, params in enumerate(param_combinations):
        print(f"\n{'='*80}")
        print(f"GRID SEARCH: Parameter combination {param_idx + 1}/{len(param_combinations)}")
        print(f"Parameters: {params}")
        print(f"{'='*80}")
        
        # Initialize results for this parameter combination
        param_results = {
            "param_combination": param_idx + 1,
            "parameters": params,
            "walk_forward_steps": []
        }
        
        # Run walk-forward validation for this parameter combination
        for step, (train_start, train_end, val_start, val_end) in enumerate(date_ranges):
            print(f"\n--- Walk-Forward Step {step + 1}/{len(date_ranges)} ---")
            print(f"Training: {train_start} to {train_end}")
            print(f"Validation: {val_start} to {val_end}")
            
            # Load and prepare data
            (X_train, R_train, M_feat_train, ids_train,
             X_val, R_val, M_feat_val, ids_val, dates_train, dates_val) = load_and_prepare_data(
                DATA_DIR, DATE_COL,
                train_start, train_end, val_start, val_end
            )
            
            if len(X_train) == 0 or len(X_val) == 0:
                print(f"Skipping step {step + 1}: insufficient data")
                continue
            
            # Reset random seeds for each step to prevent state corruption
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            
            # Reinitialize model for each training period
            _, W, K = X_train[0].shape
            # Filter out non-model parameters
            training_params = {'epochs_per_step', 'train_length', 'learning_rate'}
            model_params = {k: v for k, v in params.items() if k not in training_params}
            policy = TransformerPolicy(
                K=K,
                **model_params,
                max_len=MAX_LEN
            )
            print(f"Created new model for training period {train_start} to {train_end}")
            
            # Train model
            epochs_per_step = params['epochs_per_step']
            train_length = params['train_length']
            learning_rate = params['learning_rate']
            print(f"Training for {epochs_per_step} epochs with length {train_length} and lr {learning_rate}...")
            logs = train_policy(
                policy,
                X_train, R_train,
                epochs=epochs_per_step,
                length=train_length,
                lr=learning_rate,
                device=DEVICE,
                seed=SEED,
                M_feat_list=M_feat_train,
                risk_free_rate=RISK_FREE_RATE,
            )
            
            # Debug: Check model state before evaluation
            print(f"Model weights contain NaN: {any(torch.isnan(p).any() for p in policy.parameters())}")
            
            # Detailed evaluation on validation set
            print("Evaluating on validation set...")
            val_port, val_ew, val_wealth_port, val_wealth_ew, val_sharpe, detailed_results = evaluate_episode(
                policy, X_val, R_val, t0=0, length=len(X_val), device=DEVICE, 
                M_feat_list=M_feat_val, risk_free_rate=RISK_FREE_RATE,
                ids_list=ids_val, dates_list=dates_val, detailed=True
            )
            
            # Debug: Check results
            print(f"Validation portfolio returns contain NaN: {np.isnan(val_port).any()}")
            print(f"Validation equal-weight returns contain NaN: {np.isnan(val_ew).any()}")
            print(f"Validation Sharpe: {val_sharpe}")
            
            # Store detailed results for this step
            step_results = {
                "step": step + 1,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "train_periods": len(X_train),
                "val_periods": len(X_val),
                "val_sharpe": float(val_sharpe),
                "final_wealth_port": float(val_wealth_port[-1]) if len(val_wealth_port) > 0 else 0.0,
                "final_wealth_ew": float(val_wealth_ew[-1]) if len(val_wealth_ew) > 0 else 0.0,
                "training_logs": logs,
                "detailed_portfolio_data": detailed_results  # Every portfolio weight and stock
            }
            
            param_results["walk_forward_steps"].append(step_results)
            
            # Print step results
            print(f"Step {step + 1} Results:")
            print(f"  Validation Sharpe: {val_sharpe:.4f}")
            print(f"  Final Wealth Portfolio: {val_wealth_port[-1]:.4f}")
            print(f"  Final Wealth Equal Weight: {val_wealth_ew[-1]:.4f}")
            
            # Save checkpoint if requested
            if SAVE_CHECKPOINTS:
                checkpoint_path = output_dir / f"param_{param_idx+1}_step_{step + 1}.pt"
                torch.save(policy.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        # Calculate summary statistics for this parameter combination
        if param_results["walk_forward_steps"]:
            val_sharpes = [step["val_sharpe"] for step in param_results["walk_forward_steps"]]
            
            param_results["summary"] = {
                "mean_sharpe": float(np.mean(val_sharpes)),
                "std_sharpe": float(np.std(val_sharpes)),
                "num_steps": len(val_sharpes)
            }
            
            print(f"\nParameter Combination {param_idx + 1} Summary:")
            print(f"  Mean Sharpe: {param_results['summary']['mean_sharpe']:.4f} ± {param_results['summary']['std_sharpe']:.4f}")
        
        all_grid_results.append(param_results)
        
        # Save individual parameter combination results
        param_filename = f"param_combination_{param_idx + 1:03d}.json"
        param_path = output_dir / "individual_results" / param_filename
        param_path.parent.mkdir(exist_ok=True)
        
        with open(param_path, 'w') as f:
            json.dump(param_results, f, indent=2, default=str)
        
        print(f"Saved individual results to: {param_path}")
        
        # Save intermediate summary results after each parameter combination
        summary_results = {
            "completed_combinations": len(all_grid_results),
            "total_combinations": len(param_combinations),
            "current_best": {
                "combination": max(range(len(all_grid_results)), 
                                 key=lambda i: all_grid_results[i].get("summary", {}).get("mean_sharpe", -float('inf'))) + 1,
                "mean_sharpe": max(result.get("summary", {}).get("mean_sharpe", -float('inf')) 
                                 for result in all_grid_results)
            }
        }
        
        summary_path = output_dir / "grid_search_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_results, f, indent=2, default=str)
    
    # Find best parameter combination
    best_params = None
    best_sharpe = -float('inf')
    
    for param_result in all_grid_results:
        if param_result.get("summary") and param_result["summary"]["mean_sharpe"] > best_sharpe:
            best_sharpe = param_result["summary"]["mean_sharpe"]
            best_params = param_result
    
    # Create comprehensive summary report
    summary_report = {
        "experiment_info": {
            "total_combinations": len(param_combinations),
            "total_steps": sum(len(result["walk_forward_steps"]) for result in all_grid_results),
            "date_ranges": len(date_ranges),
            "initial_train_years": INITIAL_TRAIN_YEARS,
            "validation_years": VALIDATION_YEARS,
            "step_years": STEP_YEARS
        },
        "best_parameters": {
            "combination_id": best_params["param_combination"] if best_params else None,
            "parameters": best_params["parameters"] if best_params else None,
            "mean_sharpe": best_sharpe if best_params else None,
            "std_sharpe": best_params["summary"]["std_sharpe"] if best_params else None,
            "num_steps": best_params["summary"]["num_steps"] if best_params else None
        },
        "top_10_combinations": sorted(
            [{"combination_id": result["param_combination"], 
              "mean_sharpe": result["summary"]["mean_sharpe"],
              "std_sharpe": result["summary"]["std_sharpe"],
              "parameters": result["parameters"]} 
             for result in all_grid_results if result.get("summary")],
            key=lambda x: x["mean_sharpe"],
            reverse=True
        )[:10],
        "parameter_ranges": {
            "d_model": D_MODEL_OPTIONS,
            "nhead": NHEAD_OPTIONS,
            "num_layers": NUM_LAYERS_OPTIONS,
            "hidden": HIDDEN_OPTIONS,
            "dropout": DROPOUT_OPTIONS,
            "activation": ACTIVATION_OPTIONS,
            "norm_first": NORM_FIRST_OPTIONS,
            "ff_hidden_mult": FF_HIDDEN_MULT_OPTIONS,
            "epochs_per_step": EPOCHS_PER_STEP_OPTIONS,
            "train_length": TRAIN_LENGTH_OPTIONS,
            "learning_rate": LEARNING_RATE_OPTIONS
        }
    }
    
    # Save final summary report
    summary_path = output_dir / "final_summary_report.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    # Save complete results (optional - for debugging)
    complete_results_path = output_dir / "complete_results.json"
    with open(complete_results_path, 'w') as f:
        json.dump(all_grid_results, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    
    if best_params:
        print(f"Best Parameter Combination: {best_params['param_combination']}")
        print(f"Best Parameters: {best_params['parameters']}")
        print(f"Best Mean Sharpe: {best_sharpe:.4f}")
    else:
        print("No valid results found")
    
    print(f"Total combinations tested: {len(param_combinations)}")
    print(f"Individual results saved to: {output_dir / 'individual_results'}")
    print(f"Summary report saved to: {summary_path}")
    print(f"Complete results saved to: {complete_results_path}")
    
    return summary_report


def main():
    """Main function."""
    results = run_walk_forward_validation()
    return results


if __name__ == "__main__":
    main()
