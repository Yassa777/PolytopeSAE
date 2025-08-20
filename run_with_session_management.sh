#!/bin/bash

# 🛡️ ROBUST SESSION MANAGEMENT FOR RUNPOD
# Handles connection drops, automatic resume, and progress monitoring

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default values
CONFIG_FILE="configs/v2-focused.yaml"
DEVICE="cuda:0"
SESSION_NAME="polytope_hsae"
LOG_FILE="experiment.log"
NON_INTERACTIVE="--non-interactive"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --session)
            SESSION_NAME="$2"
            shift 2
            ;;
        --interactive)
            NON_INTERACTIVE=""
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config FILE     Config file (default: configs/v2-focused.yaml)"
            echo "  --device DEVICE   Device to use (default: cuda:0)"
            echo "  --session NAME    Session name (default: polytope_hsae)"
            echo "  --interactive     Enable interactive mode (default: non-interactive)"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "🚀 Starting Robust PolytopeSAE Experiment"
echo "Configuration:"
echo "  Config: $CONFIG_FILE"
echo "  Device: $DEVICE"
echo "  Session: $SESSION_NAME"
echo "  Log: $LOG_FILE"
echo ""

# Check if tmux is available, fallback to screen
if command -v tmux &> /dev/null; then
    SESSION_MANAGER="tmux"
    log_info "Using tmux for session management"
elif command -v screen &> /dev/null; then
    SESSION_MANAGER="screen"
    log_info "Using screen for session management"
else
    log_warning "No session manager found - running directly (risky for long experiments)"
    SESSION_MANAGER="none"
fi

# Function to check if session exists
session_exists() {
    if [ "$SESSION_MANAGER" = "tmux" ]; then
        tmux has-session -t "$SESSION_NAME" 2>/dev/null
    elif [ "$SESSION_MANAGER" = "screen" ]; then
        screen -list | grep -q "$SESSION_NAME"
    else
        return 1
    fi
}

# Function to create session and run experiment
run_experiment() {
    local cmd="python experiments/run_all_phases.py --config $CONFIG_FILE --device $DEVICE $NON_INTERACTIVE 2>&1 | tee $LOG_FILE"
    
    if [ "$SESSION_MANAGER" = "tmux" ]; then
        tmux new-session -d -s "$SESSION_NAME" "$cmd"
    elif [ "$SESSION_MANAGER" = "screen" ]; then
        screen -dmS "$SESSION_NAME" bash -c "$cmd"
    else
        eval "$cmd"
    fi
}

# Function to attach to session
attach_session() {
    if [ "$SESSION_MANAGER" = "tmux" ]; then
        log_info "Attaching to tmux session '$SESSION_NAME'"
        log_info "To detach safely: Ctrl+B then D"
        tmux attach-session -t "$SESSION_NAME"
    elif [ "$SESSION_MANAGER" = "screen" ]; then
        log_info "Attaching to screen session '$SESSION_NAME'"
        log_info "To detach safely: Ctrl+A then D"
        screen -r "$SESSION_NAME"
    fi
}

# Function to check experiment status
check_status() {
    if [ -f "$LOG_FILE" ]; then
        log_info "📊 Experiment Status:"
        echo "----------------------------------------"
        
        # Show last few lines
        echo "Recent activity:"
        tail -10 "$LOG_FILE"
        echo ""
        
        # Check for phase completion
        local phases_completed=$(grep -c "✅.*completed successfully" "$LOG_FILE" 2>/dev/null || echo "0")
        local phases_failed=$(grep -c "❌.*failed" "$LOG_FILE" 2>/dev/null || echo "0")
        local current_phase=$(grep -o "Phase [1-4]" "$LOG_FILE" | tail -1 2>/dev/null || echo "Unknown")
        
        echo "Progress:"
        echo "  Current phase: $current_phase"
        echo "  Phases completed: $phases_completed"
        echo "  Phases failed: $phases_failed"
        echo ""
        
        # Check for checkpoints
        if [ -d "runs" ]; then
            local checkpoints=$(find runs -name "*.pt" -type f | wc -l)
            echo "  Checkpoints saved: $checkpoints"
        fi
        
        # Check for W&B logs
        if grep -q "wandb" "$LOG_FILE" 2>/dev/null; then
            local wandb_url=$(grep -o "https://wandb.ai/[^[:space:]]*" "$LOG_FILE" | tail -1 2>/dev/null || echo "Not found")
            echo "  W&B URL: $wandb_url"
        fi
        
        echo "----------------------------------------"
    else
        log_warning "No log file found - experiment may not have started"
    fi
}

# Function to resume experiment if needed
resume_experiment() {
    log_info "🔄 Checking for resume capability..."
    
    # Check for existing checkpoints
    if [ -d "runs" ]; then
        local latest_checkpoint=$(find runs -name "hsae_step_*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$latest_checkpoint" ]; then
            log_success "Found checkpoint: $latest_checkpoint"
            log_info "Experiment can be resumed from this checkpoint"
            
            # Extract step number
            local step=$(echo "$latest_checkpoint" | grep -o "step_[0-9]*" | cut -d'_' -f2)
            log_info "Last checkpoint at step: $step"
            
            return 0
        fi
    fi
    
    log_info "No checkpoints found - will start from beginning"
    return 1
}

# Function to monitor experiment
monitor_experiment() {
    log_info "📱 Starting experiment monitoring..."
    log_info "Press Ctrl+C to stop monitoring (experiment will continue)"
    
    while true; do
        if [ -f "$LOG_FILE" ]; then
            # Check for recent activity (last 5 minutes)
            if [ $(find "$LOG_FILE" -mmin -5 2>/dev/null | wc -l) -eq 0 ]; then
                log_warning "⚠️  No recent activity in log file - experiment may have stalled"
            fi
            
            # Check for completion
            if grep -q "🎉.*experiment.*complete" "$LOG_FILE" 2>/dev/null; then
                log_success "🎉 Experiment completed successfully!"
                break
            fi
            
            # Check for failures
            if grep -q "CRITICAL\|FATAL\|Failed to.*after.*attempts" "$LOG_FILE" 2>/dev/null; then
                log_error "❌ Critical error detected in experiment"
                break
            fi
        fi
        
        sleep 60  # Check every minute
    done
}

# Main logic
main() {
    # Check if session already exists
    if session_exists; then
        log_info "🔄 Existing session '$SESSION_NAME' found"
        
        echo "Options:"
        echo "1. Attach to existing session"
        echo "2. Check status"
        echo "3. Kill existing session and start new"
        echo "4. Exit"
        
        if [ -n "$NON_INTERACTIVE" ]; then
            log_info "Non-interactive mode - attaching to existing session"
            attach_session
        else
            read -p "Choose option (1-4): " choice
            case $choice in
                1)
                    attach_session
                    ;;
                2)
                    check_status
                    ;;
                3)
                    if [ "$SESSION_MANAGER" = "tmux" ]; then
                        tmux kill-session -t "$SESSION_NAME"
                    elif [ "$SESSION_MANAGER" = "screen" ]; then
                        screen -S "$SESSION_NAME" -X quit
                    fi
                    log_info "Starting new experiment..."
                    run_experiment
                    attach_session
                    ;;
                4)
                    exit 0
                    ;;
                *)
                    log_error "Invalid choice"
                    exit 1
                    ;;
            esac
        fi
    else
        log_info "🚀 Starting new experiment session"
        
        # Check for resume capability
        resume_experiment
        
        # Set up environment
        log_info "Setting up environment..."
        
        # Ensure W&B is configured
        if [ -z "$WANDB_API_KEY" ]; then
            log_warning "WANDB_API_KEY not set - W&B logging may fail"
        fi
        
        # Start experiment
        log_info "Starting experiment in session '$SESSION_NAME'"
        run_experiment
        
        # Give it a moment to start
        sleep 2
        
        if [ "$SESSION_MANAGER" != "none" ]; then
            log_success "Experiment started in background session"
            log_info "To attach: $0 --config $CONFIG_FILE --device $DEVICE"
            log_info "To monitor: tail -f $LOG_FILE"
            
            # Ask if user wants to attach immediately
            if [ -z "$NON_INTERACTIVE" ]; then
                read -p "Attach to session now? (y/N): " attach_now
                if [[ $attach_now =~ ^[Yy]$ ]]; then
                    attach_session
                fi
            else
                # In non-interactive mode, start monitoring
                monitor_experiment
            fi
        fi
    fi
}

# Handle Ctrl+C gracefully
trap 'log_info "Monitoring stopped - experiment continues in background session"; exit 0' INT

# Run main function
main "$@"