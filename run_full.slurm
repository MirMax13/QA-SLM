#!/bin/bash -l
#SBATCH --job-name=tinyfridge_gen       
#SBATCH --account=p201106            
#SBATCH --partition=gpu              
#SBATCH --qos=default                
#SBATCH --nodes=1                    
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1                     
#SBATCH --time=24:00:00              
#SBATCH --output=logs/gen-%j.out     
#SBATCH --error=logs/gen-%j.err      

## Load software environment
module load GCC/13.3.0
module load Python/3.11.10-GCCcore-13.3.0

## Task execution
cd /mnt/tier2/users/u103585/my_files

# Активуємо віртуальне середовище
source venv/bin/activate

echo "🚀 Starting Python script..."
echo "Python version:"
python --version
echo "GPU info:"
nvidia-smi

python -u run_d.py