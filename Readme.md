Start up GPU and install tourchvision 

check if GPU is active `nvidia-smi`  

see which GPU available  `module avail cuda` 

load in GPU version you want (here 12.2)
 `module load cuda/12.2`

check it was loaded `nvcc --version` 

make sure you have the pyt loaded and you install + set variables

```(fall-2025-pyt)[mscs2024@scc-202 599_Project]$ pip install transformers[torch]


export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```


