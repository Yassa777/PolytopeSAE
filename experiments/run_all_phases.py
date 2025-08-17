#!/usr/bin/env python3
"""
Run All Phases: Complete V2 Focused Experiment

This script runs all four phases of the V2 focused experiment sequentially:
1. Phase 1: Teacher Vector Extraction (2-3 hours)
2. Phase 2: Baseline H-SAE Training (8-10 hours)
3. Phase 3: Teacher-Initialized H-SAE Training (12-14 hours)
4. Phase 4: Evaluation & Steering (2-3 hours)

Total estimated runtime: 25-30 GPU hours
"""

import argparse
import logging
import yaml
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_phase(phase_script, config_path, additional_args=None):
    """Run a single phase script."""
    cmd = [sys.executable, phase_script, "--config", config_path]
    
    if additional_args:
        cmd.extend(additional_args)
    
    logger.info(f"🚀 Starting {Path(phase_script).stem}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"✅ {Path(phase_script).stem} completed successfully")
        logger.info(f"⏱️  Duration: {duration}")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.error(f"❌ {Path(phase_script).stem} failed with exit code {e.returncode}")
        logger.error(f"⏱️  Duration before failure: {duration}")
        
        return False, duration


def main():
    parser = argparse.ArgumentParser(description="Run all phases of the V2 focused experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip Phase 1 (teacher extraction)")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip Phase 2 (baseline H-SAE)")
    parser.add_argument("--skip-phase3", action="store_true", help="Skip Phase 3 (teacher H-SAE)")
    parser.add_argument("--skip-phase4", action="store_true", help="Skip Phase 4 (evaluation)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config to get experiment details
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("🧪 Starting V2 Focused Polytope Discovery Experiment")
    logger.info(f"📄 Config: {args.config}")
    logger.info(f"🖥️  Device: {args.device}")
    logger.info(f"⏰ Started at: {datetime.now()}")
    
    # Get experiment directory
    exp_base_dir = Path(config['logging']['save_dir'])
    exp_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Track experiment progress
    experiment_log = {
        'start_time': datetime.now().isoformat(),
        'config_path': args.config,
        'device': args.device,
        'phases': {}
    }
    
    total_start_time = datetime.now()
    
    # Define phases
    phases = [
        ("phase1_teacher_extraction.py", "Phase 1: Teacher Vector Extraction", args.skip_phase1),
        ("phase2_baseline_hsae.py", "Phase 2: Baseline H-SAE Training", args.skip_phase2),
        ("phase3_teacher_hsae.py", "Phase 3: Teacher-Initialized H-SAE Training", args.skip_phase3),
        ("phase4_evaluation.py", "Phase 4: Evaluation & Steering", args.skip_phase4),
    ]
    
    successful_phases = 0
    failed_phases = 0
    
    for phase_script, phase_name, skip_phase in phases:
        if skip_phase:
            logger.info(f"⏭️  Skipping {phase_name}")
            experiment_log['phases'][phase_script] = {
                'skipped': True,
                'timestamp': datetime.now().isoformat()
            }
            continue
        
        # Prepare additional arguments
        additional_args = []
        if args.device:
            additional_args.extend(["--device", args.device])
        if args.debug:
            additional_args.append("--debug")
        
        # Run phase
        phase_script_path = Path(__file__).parent / phase_script
        success, duration = run_phase(str(phase_script_path), args.config, additional_args)
        
        # Log results
        experiment_log['phases'][phase_script] = {
            'success': success,
            'duration_seconds': duration.total_seconds(),
            'duration_str': str(duration),
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            successful_phases += 1
        else:
            failed_phases += 1
            
            # Ask user if they want to continue
            logger.error(f"❌ {phase_name} failed!")
            response = input("Continue with remaining phases? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                logger.info("🛑 Experiment stopped by user")
                break
    
    # Final summary
    total_end_time = datetime.now()
    total_duration = total_end_time - total_start_time
    
    experiment_log['end_time'] = total_end_time.isoformat()
    experiment_log['total_duration_seconds'] = total_duration.total_seconds()
    experiment_log['total_duration_str'] = str(total_duration)
    experiment_log['successful_phases'] = successful_phases
    experiment_log['failed_phases'] = failed_phases
    
    # Save experiment log
    log_file = exp_base_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump(experiment_log, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("🏁 EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"⏰ Total duration: {total_duration}")
    logger.info(f"✅ Successful phases: {successful_phases}")
    logger.info(f"❌ Failed phases: {failed_phases}")
    logger.info(f"📊 Results saved to: {exp_base_dir}")
    logger.info(f"📝 Experiment log: {log_file}")
    
    if failed_phases == 0:
        logger.info("🎉 All phases completed successfully!")
        
        # Check if validation targets were met
        try:
            # Try to load Phase 1 validation results
            phase1_results_file = exp_base_dir / config['logging']['phase_1_log'] / "validation_results.json"
            if phase1_results_file.exists():
                with open(phase1_results_file, 'r') as f:
                    phase1_results = json.load(f)
                
                if phase1_results.get('passes_validation', False):
                    logger.info("✅ Phase 1 geometric validation: PASSED")
                else:
                    logger.warning("⚠️  Phase 1 geometric validation: FAILED")
            
            # Check for Phase 3 vs Phase 2 comparison
            # This would be in Phase 4 results
            logger.info("📊 Check individual phase results for detailed metrics")
            
        except Exception as e:
            logger.warning(f"Could not check validation results: {e}")
    else:
        logger.error(f"❌ Experiment incomplete - {failed_phases} phases failed")
    
    logger.info("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if failed_phases == 0 else 1)


if __name__ == "__main__":
    main()